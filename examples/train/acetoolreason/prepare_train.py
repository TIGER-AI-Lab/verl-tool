# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import os
import datasets
import fire
from verl.utils.hdfs_io import copy, makedirs
import argparse
from pathlib import Path

from verl.utils.reward_score.math_dapo import remove_boxed, last_boxed_only_string
from verl.utils.reward_score import prime_math

def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))

default_system_prompt = """You are a helpful and harmless assistant."""

python_tool_system_prompt = """## python\n\nUse this tool to execute Python code in your chain of thought. The code will not be shown to the user. This tool should be used for internal reasoning, but not for code that is intended to be visible to the user (e.g. when creating plots, tables, or files).\n\nWhen you send a message containing Python code to python, it will be executed in a stateful Jupyter notebook environment. python will respond with the output of the execution or time out after 120.0 seconds. The drive at '/mnt/data' can be used to save and persist user files. Internet access for this session is UNKNOWN. Depends on the cluster."""

math_postfix = "Please reason step by step, and put your final answer within \\boxed{}. /think"

def apply_system_prompt(sys_prompt_style:str, question:str):
    """
    Apply the system prompt style to the question.
    Args:
        sys_prompt_style (str): The system prompt style to apply. Can be 'simple_rl' or 'torl'.
        question (str): The question to apply the system prompt to.
    Returns:
        list: A list of dictionaries representing the conversation with the system prompt applied.
    """
    if sys_prompt_style == 'no_tool':
        return [{'role': 'system', 'content': default_system_prompt}, {'role': 'user', 'content': question.rstrip("\n") + '\n' + math_postfix}]
    elif sys_prompt_style == 'tool':
        tool_system_prompt = "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n" + python_tool_system_prompt
        return [{'role': 'system', 'content': default_system_prompt + '\n\n' + tool_system_prompt}, {'role': 'user', 'content': question.rstrip("\n") + '\n' + math_postfix}]
    else:
        raise ValueError(f"Unknown system prompt style: {sys_prompt_style}")


def main(
    data_source='acemath',
    data_dir="data/math_rl",
    local_dir="data/math_rl",
    hdfs_dir=None,
):
    data_dir = Path(data_dir)
    json_files = []
    for file in data_dir.iterdir():
        if file.suffix == '.json' or file.suffix == '.jsonl':
            json_files.append(file)
    assert len(json_files) > 0, f"No json or jsonl files found in {data_dir}"
    train_datasets = {}
    for json_file in json_files:
        print(f"Loading data from {json_file}...", flush=True)
        dataset = datasets.load_dataset('json', data_files=str(json_file), split='train')
        data_source = json_file.stem
        dataset = dataset.add_column('data_source', [data_source]*len(dataset))
        train_datasets[data_source] = dataset

    def map_train(item, index):
        return {
            "problem_id": f"{item['data_source']}-{index}",
            "data_source": item['data_source'],
            "question": item['problem'],
            "answer": str(item['answer']),
        }
    for key in train_datasets:
        train_datasets[key] = train_datasets[key].map(map_train, with_indices=True)
        print(f"Train dataset {key}: {train_datasets[key]}")
    

    test_datasets = {}
    aime24_data_paths = ["MathArena/aime_2024_I", "MathArena/aime_2024_II"]
    aime25_data_path = "DongfuJiang/aime_2025"
    for dataset_path in aime24_data_paths:
        ds = datasets.load_dataset(dataset_path, split='train') # actually test set
        data_source = "aime24"
        def map_aime24(item):
            item['problem_id'] = f"aime24-{item['problem_idx']}"
            item['data_source'] = "aime24"
            return {
                "problem_id": item['problem_id'],
                "data_source": item['data_source'],
                "problem": item['problem'],
                "answer": str(item['answer']),
            }
        ds = ds.map(map_aime24)
        if data_source in test_datasets:
            test_datasets[data_source] = datasets.concatenate_datasets([test_datasets[data_source], ds])
        else:
            test_datasets[data_source] = ds

    aime25_dataset = datasets.load_dataset(aime25_data_path, split='train') # actually test set
    data_source = "aime25"
    def map_aime25(item):
        item['problem_id'] = f"aime25-{item['problem_idx']}"
        item['data_source'] = "aime25"
        return {
            "problem_id": item['problem_id'],
            "data_source": item['data_source'],
            "problem": item['problem'],
            "answer": str(item['answer']),
        }
    aime25_dataset = aime25_dataset.map(map_aime25)
    test_datasets[data_source] = aime25_dataset

    # math500_test_dataset = datasets.load_dataset('HuggingFaceH4/MATH-500', split='test')
    # def map_math500(item):
    #     item['problem_id'] = f"math500-{item['unique_id']}"
    #     item['data_source'] = "math500"
    #     return {
    #         "problem_id": item['problem_id'],
    #         "data_source": item['data_source'],
    #         "problem": item['problem'],
    #         "answer": str(item['solution']),
    #     }
    # math500_test_dataset = math500_test_dataset.map(map_math500)
    # test_datasets['math500'] = math500_test_dataset
    
    
    def make_map_fn(split, data_source, sys_prompt_style):

        def process_fn(example, idx):
            problem_id = example['problem_id']
            problem = example['problem']
            answer = str(example['answer'])

            data = {
                "data_source": f"{data_source}_{sys_prompt_style}",
                "prompt": apply_system_prompt(sys_prompt_style, problem),
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "use_tool": sys_prompt_style == "tool",
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'question': problem,
                    'problem_id': problem_id,
                }
            }
            return data

        return process_fn
    
    print(train_datasets)
    print(test_datasets)
    # for no tool processing
    all_train_datasets = []
    all_test_datasets = []
    for sys_prompt_style in ['no_tool', 'tool']:
        for key in train_datasets:
            print(f"Processing train dataset {key} with sys_prompt_style={sys_prompt_style}...", flush=True)
            dataset = train_datasets[key].map(function=make_map_fn('train', key, sys_prompt_style), with_indices=True, remove_columns=train_datasets[key].column_names)
            dataset.to_parquet(os.path.join(local_dir, f'train_{sys_prompt_style}_{key}.parquet'))
            print(f"Saved {len(dataset)} examples to train_{sys_prompt_style}_{key}.parquet")
            print(dataset)
            print(dataset[0])
            all_train_datasets.append(dataset)
        for key in test_datasets:
            print(f"Processing test dataset {key} with sys_prompt_style={sys_prompt_style}...", flush=True)
            dataset = test_datasets[key].map(function=make_map_fn('test', key, sys_prompt_style), with_indices=True, remove_columns=test_datasets[key].column_names)
            dataset.to_parquet(os.path.join(local_dir, f'test_{sys_prompt_style}_{key}.parquet'))
            print(f"Saved {len(dataset)} examples to test_{sys_prompt_style}_{key}.parquet")
            print(dataset)
            print(dataset[0])
            all_test_datasets.append(dataset)
    
    all_train_dataset = datasets.concatenate_datasets(all_train_datasets)
    all_train_dataset.to_parquet(os.path.join(local_dir, f'train_all.parquet'))
    print(f"Saved train_all.parquet")
    print(all_train_dataset)
    print(all_train_dataset[0])
    all_test_dataset = datasets.concatenate_datasets(all_test_datasets)
    all_test_dataset.to_parquet(os.path.join(local_dir, f'test_all.parquet'))
    print(f"Saved test_all.parquet")
    print(all_test_dataset)
    print(all_test_dataset[0])
    
    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
    

if __name__ == '__main__':
    fire.Fire(main)
    
"""
python examples/train/acetoolreason/prepare_train.py 
"""