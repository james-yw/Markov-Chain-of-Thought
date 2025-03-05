import os
import re
import json
import random
import json
import numpy as np
from pathlib import Path
from typing import Iterable, Union, Any

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                print("Error in loading:", line)
                exit()

def save_jsonl(samples, save_path, mode = "w"):
    # ensure path
    folder = os.path.dirname(save_path)
    os.makedirs(folder, exist_ok=True)

    with open(save_path, mode, encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    print("Saved to", save_path)

def write_jsonl(sample, save_path, mode="a"):
    # ensure path
    folder = os.path.dirname(save_path)
    os.makedirs(folder, exist_ok=True)

    with open(save_path, mode, encoding="utf-8") as f:
        f.write(json.dumps(sample) + "\n")


def load_prompt(data_name):
    if data_name in ['math']:
        data_name = "math_diversity"
    prompt_path = "./prompts/{}.md".format(data_name)
    if not os.path.exists(prompt_path):
        raise ValueError("Prompt file not found")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip() + "\n\n"



def extract_text(text, symbol="subquestion"):
    pattern = rf'<{symbol}>(.*?)</{symbol}>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip('\n')
    else:
        return None
    


def load_json(file: Union[str, Path]) -> Any:
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)


def diff_jsonl(fileA, fileB, save_file: str=None):
    
    unique_lines_in_a = set()
    with open(fileA, 'r') as file_a:
        for line in file_a:
            json_line = json.loads(line)
            unique_key = json.dumps(json_line, sort_keys=True)

            unique_key = json_line.get('instruction')
            # unique_key = "".join([str(value) for key, value in json_line.items()])
            # unique_key = "".join([str(value) for key, value in json_line.items() if key != "math_tag"])


            unique_lines_in_a.add(unique_key)

    unique_lines_in_b = []
    with open(fileB, 'r') as file_b:
        for line in file_b:
            json_line = json.loads(line)
            unique_key = json.dumps(json_line, sort_keys=True)

            unique_key = json_line.get('question').strip()
            unique_key = "<question>\n" + unique_key + "\n</question>\n"
            # unique_key = "".join([str(value) for key, value in json_line.items()])
            # unique_key = "".join([str(value) for key, value in json_line.items() if key != "math_tag"])

            json_line_sorted = unique_key
           
            if json_line_sorted not in unique_lines_in_a:
                unique_lines_in_b.append(line.rstrip('\n'))  


    with open(save_file, 'w') as file:
        for line in unique_lines_in_b:
            line = json.loads(line)
            line["tag"] = "not in trainset iteration 1"
            line = json.dumps(line)
            file.write(line + '\n')
    print("Saved to", save_file)


    print(f"Unique lines in B but not in A: {len(unique_lines_in_b)}")


import json

def deduplicate_jsonl(filepath, output_file='deduplicated_output.jsonl'):
    unique_entries = {}
    deduplicated_list = []

    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                print(f"Error decoding JSON: {line}")
                continue
        
            unique_key = "".join([str(value) for key, value in entry.items() if key != "source" and key != "idx"])
        
            if unique_key not in unique_entries:
                unique_entries[unique_key] = True
                deduplicated_list.append(entry)
    
    deduplicated_list = sorted(deduplicated_list, key=lambda x: (x['idx'], x.get('question', '')))

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for item in deduplicated_list:
            json.dump(item, outfile)
            outfile.write('\n')
    print(f"Deduplication complete. Total unique entries: {len(deduplicated_list)}")
