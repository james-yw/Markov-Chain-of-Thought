from __future__ import annotations

import re
import argparse
from typing import List, Dict
from termcolor import colored

from vllm.outputs import RequestOutput
from timeout_decorator import timeout

from python_tool import PythonInterpreter




SFT_PROMPT = "<question>\n{question}\n</question>\n"
PRIMER = "<p>\n"
STOP = ["\n</code>", "</code>"]
CODE_LTAG = "<code>"
CODE_RTAG = "</code>"

def _python_ast_init():
    python = PythonInterpreter(globals=globals(), locals=None)
    return python


def tool_wrapper(tool):
    def _tool(query):
        return tool.run(query)
    return _tool


def no_action_wrapper(tool):
    def _tool(query):
        return "No action, no observation. Please continue to solve."
    return _tool

def action_execution(parser_results: List[Dict[str, str]]) -> str:

    @timeout(30)
    def _action_execution(parser_results: List[Dict[str, str]]) -> str:
        cur_action = parser_results[-1]["action"]
        tool_func = tools[cur_action]

        # first, execute historical action inputs with the same action, but not output
        for history_act in parser_results[:-1]:
            if history_act["action"] == cur_action:
                _ = tool_func(history_act["action_input"])
        
        # then, execute current action input, and return output
        observation = str(tool_func(parser_results[-1]["action_input"]))
        del tool_func
        return observation 
    
    try:
        observation = _action_execution(parser_results)
    except Exception as e:
        observation = "{}: {}".format(type(e).__name__, str(e))

    return observation

# We define a dummy tool, to implement multiple tools.
tools = {
    "None": no_action_wrapper(_python_ast_init()),
    "python_interpreter": tool_wrapper(_python_ast_init()),
}

class STEP(object):
    
    def __init__(self,
        text: str = "",
        action: str = "",
        action_input: str = "",
        observation: str = "",
        sub_question: str = "",
        final_answer: str = "",
        depth: int = 0,
    ):
        self.text = text
        self.action = action
        self.action_input = action_input
        self.observation = observation
        self.sub_question = sub_question
        self.final_answer = final_answer
        self.depth = depth

        self.next_step = None
        self.is_terminal = False



class Solver(object):
    def __init__(self, args, question: str):
        self.args = args
        self.question = question

        self.start_step = STEP()

        self.current_step = self.start_step
        self.step_texts = []
        self.step_actions = []

        self.all_sub_questions = []
    
    def step_generate_flag(self) -> bool:
        return not self.current_step.is_terminal and self.current_step.depth <= self.args.max_depth
    
    def terminal_flag(self) -> bool:
        return not self.current_step.is_terminal

    def get_llm_request(self) -> str:

        # create prompt
        if self.current_step.sub_question:
            prompt = SFT_PROMPT.format(question=self.current_step.sub_question)
        else:
            prompt = SFT_PROMPT.format(question=self.question)
        if self.args.verbose:
                print(colored(f"Prompt: {prompt}\n", "blue"))
        return prompt
    
    def get_question_request(self) -> str:
        if self.current_step.sub_question:
            prefix = SFT_PROMPT.format(question=self.current_step.sub_question)
        else:
            prefix = SFT_PROMPT.format(question=self.question)
        # if self.current_step.observation:
        #     prompt = f"{prefix}{self.current_step.text}\n"
        # else:
        #     raise ValueError("No observation to generate question.")
        prompt = f"{prefix}{self.current_step.text}\n"
        if self.args.verbose:
                print(colored(f"Prompt: {prompt}", "blue"))
        return prompt

    def update_question(self, output: RequestOutput) -> None:
        sampled_step_result = output.outputs[0].text.strip()
        if self.args.verbose:
            print(colored(f"Sampled Sub Question: {sampled_step_result}", "green"))
        # parsing code snippet
        step_result, parser_result = self.action_parser(sampled_step_result)
        self.process_step_result(step_result, parser_result, "Output", update_depth=False)

            
    
    def step_generate(self, output: RequestOutput) -> None:
        """process output from vllm
        e.g.,

        outputs = llm.generate(prompts, sampling_params)
        for output in outputs:
            step_generate(output)
        """
        # sampled_step_result = (PRIMER + output.outputs[0].text).strip()
        sampled_step_result = output.outputs[0].text.strip()
        if self.args.verbose:
            print(colored(f"Sampled Step: {sampled_step_result}", "green"))
        # parsing code snippet
        step_result, parser_result = self.action_parser(sampled_step_result)
        self.process_step_result(step_result, parser_result, "Output")

    def process_step_result(
        self, 
        step_result: str, 
        parser_result: Dict[str, str],
        observation_key: str,
        update_depth: bool = True
    ) -> None:
        
        #create new step
        new_step = STEP()
        if update_depth:
            new_step.depth = self.current_step.depth + 1
        else:
            new_step.depth = self.current_step.depth


        if parser_result is None:
            new_step.is_terminal = True
            new_step.text = step_result
            new_step.final_anser = "Cannot generate parsable text."
        elif parser_result["final_answer"]:
            new_step.is_terminal = True
            new_step.text = step_result
            new_step.final_answer = parser_result["final_answer"]
        elif parser_result["sub_question"]:
            new_step.text = step_result
            new_step.sub_question = parser_result["sub_question"]
            self.all_sub_questions.append(new_step.sub_question)
            
        elif parser_result["action"]:
            new_step.sub_question = self.current_step.sub_question
            new_step.action = parser_result["action"]
            new_step.action_input = parser_result["action_input"]
            # update step_actions
            self.step_actions.append(parser_result)

            # get observation
            observation = action_execution(self.step_actions).rstrip()
            new_step.observation = observation


            if self.args.verbose:
                print(colored(f"{observation_key}: {observation}", "yellow"))
            
            # new_step.text = f"{step_result}\n{observation_key}: {observation}"
            new_step.text = f"{step_result}\n<output>\n{observation}\n</output>"


        else:
            print("WARNING:")
            new_step.text = step_result

        # update step_texts
        self.step_texts.append(new_step.text)
        # update current step
        self.current_step.next_step = new_step
        self.current_step = new_step

    def action_parser(self, text: str):
        includes_answer = "Final Answer:" in text
        includes_sub_question = "Sub Question:" in text
        # includes_sub_question = "Sub Question:" in text or "Subproblem 1" in text or "Subproblem" in text 
        regex = r"{code_ltag}[\s]*(.*)".format(code_ltag=CODE_LTAG)
        code_match = re.search(regex, text, re.DOTALL)

        parser_result = { 
            "action": "",
            "action_input": "",
            "sub_question": "",
            "final_answer": "",
        }

        if code_match:
            if includes_answer:
                print(f"Warning: Incorrect format generated: `{text}`")
                return text, None
            
            text = f"{text}\n{CODE_RTAG}"
            code_snippet = code_match.group(1)
            parser_result["action"] = "python_interpreter"
            parser_result["action_input"] = code_snippet.strip(" ").strip('"')
            return text, parser_result
        
        
        if includes_answer:
            parser_result["final_answer"] = text.split("Final Answer:")[-1].split("</p>")[0].strip()
            return text, parser_result
        
        elif includes_sub_question:
            parser_result["sub_question"] = text.split("Sub Question:")[-1].split("</p>")[0].strip()
            # parser_result["sub_question"] = text.split(":")[-1].split("</p>")[0].strip()

            # parser_result["sub_question"] = re.search('Sub Question:(.*?)</p>', text).group(1).strip()
            return text, parser_result
        
        else:
            print(f"Warning: Could not parse LLM output: `{text}`")
            return text, None
        
def parse_args():
    args = argparse.ArgumentParser()

    args.add_argument('-c', '--checkpoint_dir', type=str, required=True, help="folder of model checkpoint.")
    args.add_argument('--max_depth', type=int, default=8, help="maximum step of solution")
    args.add_argument('--verbose', action="store_true", help="print intermediate result on screen")
    args.add_argument('--temperature', type=float, default=0, help="for sampling")

    args.add_argument('-q', '--question', type=str, default=None, help="question")
    args.add_argument('--seed', type=int, default=1234, help="random seed.")
    
    args = args.parse_args()
    return args

if __name__ == '__main__':
    # the following script shows an example to solve one single question.
    from vllm import LLM, SamplingParams

    args = parse_args()

    # init llm
    llm = LLM(model=args.checkpoint_dir, tensor_parallel_size=1, trust_remote_code=True, seed=args.seed)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=-1,
        top_p=1.0,
        use_beam_search=False,
        best_of=1,
        max_tokens=2048, 
        n=1, 
        stop=STOP,
    )

    # define question and solver
    if args.question:
        question = args.question
    else:
        # an example question
        question = "Given complex number $(a+i)(1-ai)=2,\;a \in \mathbb{R}$, find $a$."
        # question = "What is the smallest value of $x$ such that $|5x - 1| = |3x + 2|$? Express your answer as a common fraction."
        question = "If $2^8=4^x$, what is the value of $x$?"
        question = "Simplify\n\\[4 \\sin x \\sin (60^\\circ - x) \\sin (60^\\circ + x).\\]The answer will be a trigonometric function of some simple function of $x,$ like \"$\\cos (2x)$\" or \"$\\sin (x^3)$\""
        question = "If $\\sin x + \\cos x = \\frac{1}{5}$ and $0 < x < \\pi,$ find $\\tan x.$" 
        question = "Find the smallest positive integer solution to $\\tan{19x^{\\circ}}=\\dfrac{\\cos{96^{\\circ}}+\\sin{96^{\\circ}}}{\\cos{96^{\\circ}}-\\sin{96^{\\circ}}}$."
        question = "Given that the largest integer value of $r$ within the interval $(2 - 10\\sqrt{2}, 2 + 10\\sqrt{2})$ is 16, and using the formula for the total number of band members $r \\times (\\frac{r}{2} - 2) + 2$, what is the total number of band members when $r = 16$?"
        question = "What is the positive difference between $120\\%$ of 30 and $130\\%$ of 20?"
        question = "How many vertical asymptotes does the graph of $y=\\frac{2}{x^2+x-6}$ have?"
        question = "Find $x$ such that $\\lceil x \\rceil + x = \\dfrac{23}{7}$. Express $x$ as a common fraction."
        question = "Evaluate $i^5+i^{-25}+i^{45}$."
        question = "If $2^8=4^x$, what is the value of $x$?"
        question = "What is the 100th term of the arithmetic sequence 6, 10, 14, 18, ...?"
    
    
    solver = Solver(args, question)

    # run solver
    while solver.step_generate_flag():
        prompt = solver.get_llm_request()
        prompts = [prompt]
        # import pdb; pdb.set_trace()
        outputs = llm.generate(prompts, sampling_params)
        solver.step_generate(outputs[0])


        if solver.terminal_flag():
            prompt_2 = solver.get_question_request()
            prompts_2 = [prompt_2]
            outputs_2 = llm.generate(prompts_2, sampling_params)
            solver.update_question(outputs_2[0])


    
    # save solution
    full_solution = "\n".join(solver.step_texts)

    import json
    
    data = {}
    data["question"] = question
    data["react"] = json.dumps(full_solution, ensure_ascii=False)
    
    with open("inference_v2.jsonl", "a+") as writer:
        writer.write(json.dumps(data, ensure_ascii=False) + '\n')

    


