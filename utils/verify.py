# !/usr/bin/env python
# -*-coding:utf-8 -*-


import argparse
import codecs
import json
import sys

from tqdm import tqdm

import pkg_resources
math_evaluation_version = pkg_resources.get_distribution("math_evaluation").version
print(f"math_evaluation version: {math_evaluation_version}")
if math_evaluation_version == '0.0.1':
    from math_evaluation.core.evaluations import is_equiv
    from math_evaluation.core.preprocess import *
    from math_evaluation.core.metamath_util import is_equiv as metamath_is_equiv
elif "0.2" in math_evaluation_version:
    from math_evaluation import is_equiv
else:
    raise ImportError("No math_evaluation version found")


from utils import load_jsonl, save_jsonl, write_jsonl, load_prompt, extract_text

def parse_args():
    parser = argparse.ArgumentParser(description="String Match based Eval Math Result")
    parser.add_argument("--input-file", type=str, default="./datasets/formatted/simplified/math_gpt_human_simplified.prediction.jsonl", help="input file.")
    parser.add_argument("--save-file", type=str, default=None, help="save file.")
    parser.add_argument("--label-preprocess", type=str, default="gsm8k_label",
                        help="label should be preprocessed by function [label-preprocess]")
    parser.add_argument("--predicted-preprocess", type=str, default="gsm8k_label",
                        help="predicted should be preprocessed by function [predicted-preprocess]")
    parser.add_argument("--debug", type=bool, default=False, help="debug mode.")
    args = parser.parse_args()
    return args


def get_preprocess_func(args):
    try:
        return  globals()[args.label_preprocess], globals()[args.predicted_preprocess]
    except:
        print(f"label_preprocess: {args.label_preprocess}")
        print(f"predicted_preprocess: {args.predicted_preprocess}")
        raise

def main(args):
    print("args: {{{")
    for k, v in sorted(vars(args).items()):
        print(f"\t{k}: {v}")
    print("}}}\n")

    with open(args.input_file, 'r') as fin:
        data_list = fin.readlines()

    # label_preprocess, predicted_preprocess = get_preprocess_func(args)

    right_count = 0
    total_count = 0

    update_data_list = []

    no_sub_questions_count = 0
    for idx, item_json_string in enumerate(tqdm(data_list)):
        item = json.loads(item_json_string)
        total_count += 1
    
        try:
            ref_str = item["final_answer"]
        except:
            item["final_answer"] = item["answer"]
            ref_str = item["answer"]
        
        
        answer = item["react"].get("final_answer", "None")

        # if item["all_sub_questions"] != []:
        #     answer = item["all_sub_questions"][-1].get("final_answer", "None")
        # else:
        #     no_sub_questions_count += 1
        #     answer = "None"

        ref_answer = ref_str
        predicted_answer = answer

        whether_right = is_equiv(ref_answer, predicted_answer, verbose=args.debug)

        
        # whether_right = metamath_is_equiv(ref_answer, predicted_answer)

        # if args.debug:
        #     print(f"predict react: {item['react']}")
        #     print(f"ref_str: {ref_str} VS predicted_str: {predicted_answer} = {whether_right}")

        if whether_right:
            item['verification'] = "True"
            
            right_count += 1
            # if args.debug:
            #     print(f"question: {item['question']}")
            #     print(f"predict react: {item['react']}")
            #     print(f"predict answer: {item['react']['final_answer']}")
            # print("idx: ", idx)

            

        else:
            item['verification'] = "False"
        item['math_eval'] = str(math_evaluation_version)
        update_data_list.append(item)

    if args.save_file:
        save_jsonl(update_data_list, args.save_file)
    else:
        save_jsonl(update_data_list, args.input_file)
    


    right_rate = (right_count / total_count) * 100
    print(f"Total Questions: {total_count}")
    print(f"Right Answer Count: {right_count}")
    print(f"Right Rate: {right_rate:.2f}%")
    print(f"No sub_questions count: {no_sub_questions_count}")


if __name__ == "__main__":
   args = parse_args()
   main(args)
