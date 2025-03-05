from __future__ import annotations

import os
import re
import argparse
import backoff 
import requests

from typing import Optional, List, Dict, Union
from termcolor import colored
from functools import partial

from utils import load_jsonl, save_jsonl, write_jsonl, extract_text

import pdb

import tqdm
import time

import logging

from few_shot_prompt import (
    custom_prefix,
    custom_suffix,
    gsm8k_examples,
    math_examples,
)

from zero_shot_prompt import zero_shot_prompt_prefix




import openai
if openai.__version__ >= "1.0.0":
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<GPT API Key>"))

    @backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_time=60)
    def completions_with_backoff(**kwargs):
        return client.chat.completions.create(**kwargs)

else:
    openai.api_key = os.environ.get("OPENAI_API_KEY", "<GPT API Key>")

    @backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_time=60)
    def completions_with_backoff(**kwargs):
        return openai.Completion.create(**kwargs)


def gpt(
    prompt: str, 
    model: str, 
    temperature: float = 0, 
    max_tokens: int = 4096, 
    n: int = 1, 
    stop: Union[Optional[str], List[str]] = None, 
    seed: int = None,
) -> List[str]:
    messages = [{"role": "user", "content": prompt}]
    response = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop, seed=seed)
    # return [choice["message"]["content"] for choice in response["choices"]]
    return [choice.message.content for choice in response.choices]


PRIMER = "<subquestion>"

def create_prompt(
    dataset: str,
    example: Dict[str, str],
) -> str:
    if dataset == "gsm8k":
        custom_examples = gsm8k_examples
    elif dataset == "math":
        custom_examples = math_examples
    else:
        raise ValueError("Invalid dataset")
    
    if len(custom_examples) > 1:
        example_prefix = "The following are %d demonstration examples." % len(custom_examples)
    elif len(custom_examples) == 1:
        example_prefix = "The following is a demonstration example."
    
    custom_question = custom_suffix.format(
            origin_problem = example["origin_problem"],
            thought = example["thought"],
            action = example["action"],
            observation = example["observation"],
        )

    prompt_template = "\n\n".join([custom_prefix, example_prefix, *custom_examples, custom_question])
    return prompt_template

class GPTproblemGenerator:
    def __init__(self, args):
        self.args = args
        self.llm = partial(gpt, model=args.gpt_model_id, temperature=args.temperature, seed=args.seed)


    def generate_one(self, data: dict):
        args = self.args
        
        example = {}
        example["origin_problem"] = data["question"]
        example["thought"] = data['solution']["thought"]
        example["action"] = data['solution']["action_input"]
        example["observation"] = data['solution']["observation"]

        simplified_question = self.get_parsable_samples(example, n=1)[0]
        
        max_retry = 5
        retry_cnt = 0
        while retry_cnt < max_retry:
            try:
                data['sub_question'] = simplified_question.strip()
                break
            except Exception as e:
                simplified_question = self.get_parsable_samples(example, n=1)[0]
        
        if args.verbose:
            print(colored(f"simplified_question: {simplified_question}\n", "red"))
            print(colored(f"intermediate_observation: {step['observation']}\n", "yellow"))
        
        return simplified_question



    def generate(self):
        args = self.args
        if getattr(args, "dataset", None) is None and getattr(args, "dataset_file", None) is None:
            raise ValueError("dataset should be provided.")
        
        # if getattr(args, "dataset", None) is not None:
        #     if args.dataset in ['math']:
        #         dataset = 'math_gpt_human' if not args.debug else 'math_demo'
        #     elif args.dataset in ['gsm8k']:
        #         dataset = 'gsm8k_gpt_human' if not args.debug else 'gsm8k_demo'
        #     else:
        #         raise ValueError("Invalid dataset")
        #     data_path = "./datasets/original/{}.jsonl".format(dataset)

        if getattr(args, "dataset_file", None) is not None:
            data_path = args.dataset_file
        
        if getattr(args, "save_file", None) is not None:
            save_file = args.save_file
        else:
            save_file = os.path.join(args.save_path, dataset + f"_simplified_questions_temp_{args.temperature}_seed_{args.seed}.jsonl")

        all_questions = []
        
        write_jsonl({"Start from": args.start_idx}, save_file) if args.start_idx > 0 else None

        for idx, all_data in enumerate(tqdm.tqdm(list(load_jsonl(data_path)))):

            data = all_data["all_sub_questions"][-1]
            if idx < args.start_idx:
                continue
            example = {}
            if args.verbose:
                    print(colored(f"original_question: {data['question']}\n", "blue"))
                    print(colored(f"intermediate_thought_step: {len(data['solution']['intermediate_steps'])}", "green"))
            
            question = {}
            question.update({"original_question": data["question"]})

                
            
            example["origin_problem"] = data["question"]
            example["thought"] = data['solution']["thought"]
            example["action"] = data['solution']["action_input"]
            example["observation"] = data['solution']["observation"]

            simplified_question = self.get_parsable_samples(example, n=1)[0]
            # simplified_question = self.get_parsable_samples(example, n=8)

            

            max_retry = 5
            retry_cnt = 0
            while retry_cnt < max_retry:
                try:
                    data['sub_question'] = simplified_question.strip()
                    # data['sub_question'] = {
                    #     "sub_question_1": simplified_question[0].strip(),
                    #     "sub_question_2": simplified_question[1].strip(),
                    #     "sub_question_3": simplified_question[2].strip(),
                    #     "sub_question_4": simplified_question[3].strip(),
                    #     "sub_question_5": simplified_question[4].strip(),
                    #     "sub_question_6": simplified_question[5].strip(),
                    #     "sub_question_7": simplified_question[6].strip(),
                    #     "sub_question_8": simplified_question[7].strip(),
                    # }
                    break
                except Exception as e:
                    simplified_question = self.get_parsable_samples(example, n=1)[0]
                    # simplified_question = self.get_parsable_samples(example, n=8)
            
            if args.verbose:
                print(colored(f"simplified_question: {simplified_question}\n", "red"))
                print(colored(f"intermediate_observation: {step['observation']}\n", "yellow"))

            all_data["all_sub_questions"][-1] = data
            write_jsonl(all_data, save_file)
            
    
    def get_parsable_samples(self, example, n: int = 1):
        args = self.args
        try:
            samples = self.get_llm_samples(example, n)
            # samples = self.get_llm_samples(example, n, 0.7)
            samples = [extract_text(sample) for sample in samples]
            return samples
        except Exception as e:
            max_retry = 5
            temperature = 0.8
            print(f"Exception: {e}. We will retry {max_retry} times by setting a larger temperature {temperature}.")
            retry_cnt = 0
            while retry_cnt < max_retry:
                try: 
                    samples = self.get_llm_samples(example, n, temperature)
                    samples = [extract_text(sample) for sample in samples]
                    return samples
                except Exception as e:
                    print(f"Exception: {e}. Retry {retry_cnt + 1} times.")
                    retry_cnt += 1
            print(f"Failed to get parsable samples after {max_retry} retries.")
            return None


    def get_llm_samples(self, example, n: int = 1, temperature: float = 0):
        args = self.args
        if args.prompt is None:
            raise ValueError("prompt should be provided.")
        elif args.prompt == "zero_shot_prompt":
            prompt = zero_shot_prompt_prefix.format(
            origin_problem = example["origin_problem"],
            thought = example["thought"],
            action = example["action"],
            observation = example["observation"],
        )
        elif args.prompt == "few_shot_prompt":
            prompt = create_prompt(args.dataset, example)
        
        if temperature is None: # default llm
            samples = self.llm(prompt, n=n)
        else:
            diverse_llm = partial(gpt, model=self.args.gpt_model_id, temperature=temperature)
            samples = diverse_llm(prompt, n=n)
        # if args.verbose:
        #     print(colored(gpt_usage(args.gpt_model_id), "blue"))
        return samples



def parse_args():
    args = argparse.ArgumentParser()

    args.add_argument(
        '-g', '--gpt_model_id', 
        type=str, 
        default="gpt-4-1106-preview",
        choices=["gpt-3.5-turbo-1106", "gpt-4-1106-preview"], 
        help="gpt model id",
    )

    args.add_argument('--temperature', type=float, default=0, help="for sampling")
    args.add_argument('--seed', type=int, default=1234, help="for sampling")

    args.add_argument('--dataset', type=str, default=None, choices=["gsm8k", "math"], help="which dataset to use")
    args.add_argument('--prompt', type=str, default="math_diversity", help="which prompt file to use")
    
    args.add_argument('--start_idx', type=int, default=0, help="start index of the dataset")
    
    args.add_argument('--dataset_file', type=str, default=None, help="path to the dataset")
    args.add_argument('--save_path', type=str, default="./datasets/simplified", help="path to save the simplified questions")
    args.add_argument('--save_file', type=str, default=None, help="file to save file.")

    args.add_argument('--verbose', action="store_true", help="print intermediate result on screen")
    args.add_argument('--debug', action="store_true", help="debug mode, using a demo dataset")

    args = args.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(filename=f'./logs/{args.dataset}_simplified_questions_temp_{args.temperature}_seed_{args.seed}.log', level=logging.INFO)
    logging.info(args)

    generator = GPTproblemGenerator(args)

    
    start = time.time() 
    generator.generate()
    logging.info(f"Time used: {time.time() - start}")

    