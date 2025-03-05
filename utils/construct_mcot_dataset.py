import os
import sys
import argparse
from tqdm import tqdm
import random
import json
import glob
from utils import load_jsonl, save_jsonl, write_jsonl, load_prompt, extract_text, deduplicate_jsonl

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def preprocess_multi_step_reasoning_dataset(original_file, dataset_file, tag = "math", waiting_generate_file = None):
    '''
    Utilize the original file (Multi-reasoning datasets) to construct dataset file (MCoT datasets) and waiting_generate_file


    original_file: str, path to the original file
    dataset_file: str, path to the dataset file
    waiting_generate_file: str, path to the waiting_generate file
    '''
   
    '''
    orginal_data_format:
    {
        "question": "",
        "solution":{
            "intermediate_steps":[
                {
                    "thought":"",
                    "action":"",
                    "action_input":"",
                    "observation":""
                },
                ...multi intermediate_steps...
                {
                    "thought":"",
                    "action":"",
                    "action_input":"",
                    "observation":""
                }
            ],
            "final answer":""
        }
        "source":""
        "idx":"",
    }
    '''

    original = list(load_jsonl(original_file))
    dataset = []
    waiting_generate = []
    for i, o in enumerate(original):
        steps = len(o["solution"]["intermediate_steps"])  # 1, 2, 3
        if steps == 0:
            raise ValueError(f"steps == 0: {o['id']}")
        # 1.1 Default: verification = True, add {question, intermediate_solution, final_answer} to dataset
        elif steps == 1:
            solution = o["solution"]["intermediate_steps"][0]
            solution["final_thought"] = ""
            all_sub_question = {
                "question": o["question"],
                "solution": solution,
                "sub_question": "",
                "final_answer": o["solution"]["final_answer"],
            }
            verification = "True"
            complete = "True"
      
        # 1.2 if steps == 2,  add {question, intermediate_solution_1, inermediate_solution_2, final_answer} to dataset
        # intermediate_solution_1 = {thought, action, action_input, observation}
        # intermediate_solution_2 = {thought}, key is "final_thought"
        elif steps == 2:
            solution = o["solution"]["intermediate_steps"][0]
            solution["final_thought"] = o["solution"]["intermediate_steps"][1]['thought']
            all_sub_question = {
                "question": o["question"],
                "solution": solution,
                "sub_question": "",
                "final_answer": o["solution"]["final_answer"],
            }
            verification = "True"
            complete = "True"
            next_step = "None"
        # 1.3 if steps > 2, add {question, intermediate_solution_1} to waiting_generate_file
        else:
            solution = o["solution"]["intermediate_steps"][0]
            solution["final_thought"] = ""
            all_sub_question = {
                "question": o["question"],
                "solution": solution,
                "sub_question": "",
                "final_answer": "",
            }
            verification = "True"
            complete = "False"
            next_step = "waiting_generate"
            
        all_sub_questions = []
        all_sub_questions.append(all_sub_question)
        data = {
                "original question": o["question"],
                "final_answer": o["solution"]["final_answer"],
                "source": o["source"],
                "idx": o["idx"],
                "math_tag": f"{tag}_{str(i)}",
                "all_sub_questions": all_sub_questions,
                "candidate_sub_question": all_sub_questions[-1]["sub_question"],
                "react": "",
                "react_tag": [],
                "verification": verification,
                "complete": complete,
                "next_step": next_step
            }
        dataset.append(data)
        if complete == "False":
            data["verification"] = ""
            waiting_generate.append(data)
            
    save_jsonl(dataset, dataset_file)
    
    if waiting_generate_file is not None:
        save_jsonl(waiting_generate, waiting_generate_file)
    print(f"Finish preprocess_multi_step_reasoning_dataset, dataset: {len(dataset)}, waiting_generate: {len(waiting_generate)}")


def construct_step_a(waiting_generate_file, candidate_file, dataset, start_idx = 0, prompt_type = "few_shot_prompt"):
    '''
    Generate new questions based on the waiting generate file using GPT4


    waiting_generate_file: str, path to the waiting_generate file
    candidate_file: str, path to the candidate file
    '''

    from gpt_api import GPTproblemGenerator
    from gpt_api import parse_args
    args = parse_args()

    args.temperature = 0.6
    args.seed = 1234
    args.dataset = dataset
    args.prompt = prompt_type

    args.dataset_file = waiting_generate_file
    args.save_file = candidate_file
    args.start_idx = start_idx

    print("args: {{{")
    for k, v in sorted(vars(args).items()):
        print(f"\t{k}: {v}")
    print("}}}\n")


    generator = GPTproblemGenerator(args)
    generator.generate()

    print(f"Finish construct_step_a: candidate_file: {candidate_file}")



def construct_step_b(candidate_file, msr_file, temp = 0, seed = 1234,  dataset = "math", checkpoint_dir = "your msr model path", start_idx = 0):
    '''
    Utilize the MSR model to verify the candidate file, and generate the intermediate steps and final answer by MSR

    candidate_file: str, path to the candidate file
    msr_file: str, path to the multi-step model result of solving the candidate file
    '''

    from batch_inference_msr import main, parse_args
    args = parse_args()
   
    args.temperature = temp
    args.seed = seed

    args.checkpoint_dir = checkpoint_dir
    args.question_file = candidate_file

    args.question_key = "sub_question"
    args.save_file = msr_file

    print("args: {{{")
    for k, v in sorted(vars(args).items()):
        print(f"\t{k}: {v}")
    print("}}}\n")
    main(args)

    print(f"Finish construct_step_b: msr_file: {msr_file}")

        
def construct_step_c(msr_file, verify_file):
    '''
    Utilize the math_evaluation libary to verify the msr file

    msr_file: str, path to the multi-step model result of solving the candidate file
    verify_file: str, path to the verify file
    '''
    from verify import main, parse_args
    args = parse_args()

    args.debug = False
    args.input_file = msr_file
    args.save_file = verify_file
    main(args)

    print(f"Finish construct_step_c: verify_file: {verify_file}")



def construct_step_d(verify_file, dataset_file, waiting_generate_file):
    '''
    Utilize the verify file to construct the dataset file and waiting generate file

    verify_file: str, path to the verify file
    dataset_file: str, path to the dataset file
    '''
    verifies = list(load_jsonl(verify_file))
    dataset = []
    waiting_generate = []

    for v in verifies:
        if v["verification"] == "True" and v["complete"] == "False":
            steps = len(v["react"]["intermediate_steps"])  # 1, 2, 3
            if steps == 0:
                raise ValueError(f"steps == 0: {v['id']}")
            elif steps == 1:
                solution = v["react"]["intermediate_steps"][0]
                solution["final_thought"] = ""
                new_sub_question = {
                    "question": v["all_sub_questions"][-1]["sub_question"],
                    "solution": solution,
                    "sub_question": "",
                    "final_answer": v["react"]["final_answer"],
                }
                verification = "True"
                complete = "True"
                next_step = "None"
            elif steps == 2:
                solution = v["react"]["intermediate_steps"][0]
                solution["final_thought"] = v["react"]["intermediate_steps"][1]['thought']
                new_sub_question = {
                    "question": v["all_sub_questions"][-1]["sub_question"],
                    "solution": solution,
                    "sub_question": "",
                    "final_answer": v["react"]["final_answer"],
                }
                verification = "True"
                complete = "True"
                next_step = "None"
            # 1.3 if steps > 2, add {question, intermediate_solution_1, sub_question} to candidates
            else:
                solution = v["react"]["intermediate_steps"][0]
                solution["final_thought"] = ""
                new_sub_question = {
                    "question": v["all_sub_questions"][-1]["sub_question"],
                    "solution": solution,
                    "sub_question": "",
                    "final_answer": "",
                }
                verification = "True"
                complete = "False"
                next_step = "waiting_generate"
                
            all_sub_questions = v["all_sub_questions"]
            all_sub_questions.append(new_sub_question)

            
            dataset.append({
                    "original question": v["original question"],
                    "final_answer": v["final_answer"],
                    "source": v["source"],
                    "idx": v["idx"],
                    "math_tag":v["math_tag"],
                    "all_sub_questions": all_sub_questions,
                    "candidate_sub_question": all_sub_questions[-1]["sub_question"],
                    "react": "",
                    "react_tag": v["react_tag"],
                    "verification": verification,
                    "complete": complete,
                    "next_step": next_step,
                    "math_eval": v['math_eval']
                }
                )
            if complete == "False":
                waiting_generate.append({
                    "original question": v["original question"],
                    "final_answer": v["final_answer"],
                    "source": v["source"],
                    "idx": v["idx"],
                    "math_tag": v["math_tag"],
                    "all_sub_questions": all_sub_questions,
                    "candidate_sub_question": all_sub_questions[-1]["sub_question"],
                    "react": "",
                    "react_tag": v["react_tag"],
                    "verification": "",
                    "complete": complete,
                    "next_step": next_step,
                    "math_eval": v['math_eval']
                }
                )
    
    save_jsonl(dataset, dataset_file)
    save_jsonl(waiting_generate, waiting_generate_file)
    print(f"Finish construct_step_d: dataset_file: {len(dataset)}, waiting_generate_file: {len(waiting_generate)}")

def construct_mcot_dataset(input_path, output_path):
    question_to_final_answer = []
    question_to_sub_question = []
    q_fa_file = os.path.join(output_path, "question_to_final_answer.jsonl")
    q_sq_file = os.path.join(output_path, "question_to_sub_question.jsonl")
    for file_name in os.listdir(input_path):
        file_name = os.path.join(input_path, file_name)
        # if "math_v3" in file_name:
        #     continue
        dataset = list(load_jsonl(file_name))
        for data in dataset:
            if data["complete"] == "True":
                num = len(data["all_sub_questions"])
                if num == 0:
                    raise ValueError(f"num == 0: {data['math_tag']}")
                elif num == 1:
                    question_to_final_answer.append(data["all_sub_questions"][0])
                else:
                    question_to_sub_question.extend(data["all_sub_questions"][:-1])
                    question_to_final_answer.append(data["all_sub_questions"][-1])
            else:
                num = len(data["all_sub_questions"]) - 1
                if num > 0:
                    question_to_sub_question.extend(data["all_sub_questions"][:-1])
    
    save_jsonl(question_to_final_answer, q_fa_file)
    save_jsonl(question_to_sub_question, q_sq_file)
    print(f"Finish extract_dataset: q_fa_file: {len(question_to_final_answer)}, q_sq_file: {len(question_to_sub_question)}")

    deduplicate_jsonl(q_fa_file, q_fa_file)
    deduplicate_jsonl(q_sq_file, q_sq_file)



class STEP(object):
    def __init__(self, prefix: dict, time: int):

        self.dataset_file = os.path.join(prefix['dataset'], f"dataset_{time}.jsonl")
        self.candidate_file = os.path.join(prefix['candidate'], f"candidate_{time}.jsonl")
        self.msr_file =os.path.join(prefix['msr'], f"msr_{time}.jsonl")
        self.verify_file = os.path.join(prefix['verify'], f"verify_{time}.jsonl")
        self.waiting_generate_file = os.path.join(prefix['waiting_generate'], f"waiting_generate_{time}.jsonl")  


if __name__ == '__main__':
    
    dataset = "math" # or "gsm8k"
    checkpoint = "deepseek"
    checkpoint_dir = "your multi-step reasoning model path"
    
    original_file = "./datasets/original/multi_reasoning_dataset_demo.jsonl" # original file path (Multi-reasoning dataset path)
    dataset_file_prefix = f"./datasets/markov_chain_of_thought/{dataset}/dataset"
    candidate_file_prefix = f"./datasets/markov_chain_of_thought/{dataset}/candidate"
    msr_file_prefix = f"./datasets/markov_chain_of_thought/{dataset}/msr"
    verify_file_prefix = f"./datasets/markov_chain_of_thought/{dataset}/verify"
    waiting_generate_file_prefix = f"./datasets/markov_chain_of_thought/{dataset}/waiting_generate"
    format_dataset_prefix = f"./datasets/markov_chain_of_thought/{dataset}/format_dataset"

    prefix = {
        "dataset": dataset_file_prefix,
        "candidate": candidate_file_prefix,
        "msr": msr_file_prefix,
        "verify": verify_file_prefix,
        "waiting_generate": waiting_generate_file_prefix,
        "format_dataset": format_dataset_prefix, 
    }

    for k, v in prefix.items():
        if not os.path.exists(v):
            os.makedirs(v)

    
    few_shot_prompt_suffixs = [".with_gpt4_few_shot_prompt.with_temp_0.5.jsonl", ".with_gpt4_few_shot_prompt.with_temp_0.6.jsonl", ".with_gpt4_few_shot_prompt.with_temp_0.7.jsonl"]
    seeds = [1518, 5200, 6824, 7370, 9786]
    temp = 0.6
    

    step_0 = STEP(prefix, 0)
    step_1 = STEP(prefix, 1)
    step_2 = STEP(prefix, 2)


    preprocess_multi_step_reasoning_dataset(original_file, step_0.dataset_file, tag = "math", waiting_generate_file = step_0.waiting_generate_file)
    
    

    # Round 1
    for few_shot_prompt_suffix in few_shot_prompt_suffixs:
        prompt_type = "few_shot_prompt"
        construct_step_a(step_0.waiting_generate_file, step_0.candidate_file + few_shot_prompt_suffix, dataset, start_idx = 0, prompt_type = prompt_type)
        
        for seed in seeds:
            suffix = f".with_{checkpoint}_7b_temp_{temp}_seed_{seed}.jsonl"
            construct_step_b(step_0.candidate_file + few_shot_prompt_suffix, step_0.msr_file + few_shot_prompt_suffix + suffix, temp, seed, dataset, checkpoint_dir)
            construct_step_c(step_0.msr_file + few_shot_prompt_suffix + suffix, step_0.verify_file + few_shot_prompt_suffix + suffix)
            construct_step_d(step_0.verify_file + few_shot_prompt_suffix + suffix, step_1.dataset_file + few_shot_prompt_suffix + suffix, step_1.waiting_generate_file + few_shot_prompt_suffix + suffix)
            

            # Round 2
            for few_shot_prompt_suffix in few_shot_prompt_suffixs:
                prompt_type = "few_shot_prompt"
                construct_step_a(step_1.waiting_generate_file + few_shot_prompt_suffix + suffix, step_1.candidate_file + few_shot_prompt_suffix + suffix, dataset, start_idx = 0, prompt_type = prompt_type)

                for seed in seeds:
                    new_suffix = f".with_{checkpoint}_7b_temp_{temp}_seed_{seed}.jsonl"
                    construct_step_b(step_1.candidate_file + few_shot_prompt_suffix + suffix, step_1.msr_file + few_shot_prompt_suffix + suffix + new_suffix, temp, seed, dataset, checkpoint_dir)
                    construct_step_c(step_1.msr_file + few_shot_prompt_suffix + suffix + new_suffix, step_1.verify_file + few_shot_prompt_suffix + suffix + new_suffix)
                    construct_step_d(step_1.verify_file + few_shot_prompt_suffix + suffix + new_suffix, step_2.dataset_file + few_shot_prompt_suffix + suffix + new_suffix, step_2.waiting_generate_file + few_shot_prompt_suffix + suffix + new_suffix)
        
    construct_mcot_dataset(dataset_file_prefix, format_dataset_prefix)