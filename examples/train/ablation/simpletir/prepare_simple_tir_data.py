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

from verl.utils.reward_score.math_dapo import remove_boxed, last_boxed_only_string

def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


simple_rl_system_prompt = '''Please reason step by step, and put your final answer within \\boxed{}.'''

torl_system_prompt = '''A conversation between User and Assistant. The user asks a question, and the Assistant solves it. Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}.
'''

simple_tir_system_prompt = '''Solve the following problem step by step. You now have the ability to selectively write executable Python code to enhance your reasoning process. The Python code will be executed by an external sandbox, and the output (after "Code execution result: ") is returned to aid your reasoning and help you arrive at the final answer. The Python code should be complete scripts, including necessary imports.

Code Format:
Each code snippet is wrapped between ```. You need to use `print()` to output intermediate results.

Answer Format:
You can use the `final_answer()` function in the code to return your final answer. For example, to answer the User Question: What is the result of the 5 + 3 + 1294.678?, you can write:
```py
answer = 5 + 3 + 1294.678
final_answer(answer)
```

You can also use \\boxed to return your answer. The last part of your response should be:
\\boxed{'The final answer goes here.'}

User Question:
'''

def apply_system_prompt(sys_prompt_style:str, question:str):
    """
    Apply the system prompt style to the question.
    Args:
        sys_prompt_style (str): The system prompt style to apply. Can be 'simple_rl' or 'torl'.
        question (str): The question to apply the system prompt to.
    Returns:
        list: A list of dictionaries representing the conversation with the system prompt applied.
    """
    if sys_prompt_style == 'simple_rl':
        return [{'role': 'user', 'content': question + '\n' + simple_rl_system_prompt}]
    elif sys_prompt_style == 'torl':
        return [{'role': 'system', 'content': torl_system_prompt}, {'role': 'user', 'content': question}]
    elif sys_prompt_style == 'simple_tir':
        return [{'role': 'user', 'content': simple_tir_system_prompt + question}]
    else:
        raise ValueError(f"Unknown system prompt style: {sys_prompt_style}")

def main(
    math_35_train_parquet='examples/train/ablation/simpletir/math35_train.parquet',
    deepscaler_train_parquet='examples/train/ablation/simpletir/deepscaler_train.parquet',
    local_dir='data/simple_tir',
    hdfs_dir=None,
    sys_prompt_style= 'simple_tir',  # simple_rl, torl, simple_tir
):
    
    math35_dataset = datasets.load_dataset('parquet', data_files=math_35_train_parquet, split='train')
    deepscaler_dataset = datasets.load_dataset('parquet', data_files=deepscaler_train_parquet, split='train')
    
    math500_test_dataset = datasets.load_dataset('HuggingFaceH4/MATH-500', split='test')
    
    # add a row to each data item that represents a unique id
    def make_math_35_train_map_fn(split, data_source):

        def process_fn(example, idx):
            question = example['extra_info']['question']
            answer = example['extra_info']['answer']
            solution = answer
            
            data = {
                "data_source": data_source,
                "prompt": apply_system_prompt(sys_prompt_style, question),
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'question': question,
                }
            }
            return data

        return process_fn
    
    # add a row to each data item that represents a unique id
    def make_deepscaler_train_map_fn(split, data_source):

        def process_fn(example, idx):
            question = example['prompt'][0]['content']
            answer = example['reward_model']['ground_truth']
            solution = answer
            
            data = {
                "data_source": data_source,
                "prompt": apply_system_prompt(sys_prompt_style, question),
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'question': question,
                }
            }
            return data

        return process_fn
	
    def make_map_fn(split, data_source):

        def process_fn(example, idx):
            question = example.pop('problem')
            answer = example.pop('solution')
            solution = extract_solution(answer)
            
            data = {
                "data_source": data_source,
                "prompt": apply_system_prompt(sys_prompt_style, question),
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'question': question,
                }
            }
            return data

        return process_fn
    
    data_source = "math_dapo"
    # train_dataset = train_dataset.map(function=make_train_map_fn('train', data_source), with_indices=True, remove_columns=train_dataset.column_names)
    # test_dataset = test_dataset.map(function=make_train_map_fn('test', data_source), with_indices=True, remove_columns=test_dataset.column_names)
    math35_dataset = math35_dataset.map(function=make_math_35_train_map_fn('train', 'math_35'), with_indices=True, remove_columns=math35_dataset.column_names)
    deepscaler_dataset = deepscaler_dataset.map(function=make_deepscaler_train_map_fn('train', 'deepscaler'), with_indices=True, remove_columns=deepscaler_dataset.column_names)
    train_dataset = datasets.concatenate_datasets([math35_dataset, deepscaler_dataset])
    math500_test_dataset = math500_test_dataset.map(function=make_map_fn('test', 'HuggingFaceH4/MATH-500'), with_indices=True, remove_columns=math500_test_dataset.column_names)
    
    print(train_dataset)
    print(train_dataset[0])
    
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    math500_test_dataset.to_parquet(os.path.join(local_dir, 'math500_test.parquet'))
    
    # aime24
    aime24_dataset = datasets.load_dataset('Maxwell-Jia/AIME_2024', split='train') # actually test set
    def make_map_fn(split, data_source):

        def process_fn(example, idx):
            question = example.pop('Problem')
            answer = str(example.pop('Answer'))
            solution = example.pop('Solution')
            
            data = {
                "data_source": data_source,
                "prompt": apply_system_prompt(sys_prompt_style, question),
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'question': question,
                }
            }
            return data

        return process_fn
    
    aime24_dataset = aime24_dataset.map(function=make_map_fn('test', 'aime24'), with_indices=True, remove_columns=aime24_dataset.column_names)
    aime24_dataset.to_parquet(os.path.join(local_dir, 'aime24_test.parquet'))
    print(aime24_dataset)
    print(aime24_dataset[0])
    
    # aime25
    aime25_dataset = datasets.load_dataset('opencompass/AIME2025', 'AIME2025-I', split='test') # actually test set
    aime25_dataset2 = datasets.load_dataset('opencompass/AIME2025', 'AIME2025-II', split='test') # actually test set
    # concatenate the two datasets
    aime25_dataset = datasets.concatenate_datasets([aime25_dataset, aime25_dataset2])
    
    def make_map_fn(split, data_source):

        def process_fn(example, idx):
            question = example.pop('question')
            answer = str(example.pop('answer'))
            
            data = {
                "data_source": data_source,
                "prompt": apply_system_prompt(sys_prompt_style, question),
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'question': question,
                }
            }
            return data

        return process_fn

    aime25_dataset = aime25_dataset.map(function=make_map_fn('test', 'aime25'), with_indices=True)
    aime25_dataset.to_parquet(os.path.join(local_dir, 'aime25_test.parquet'))
    print(aime25_dataset)
    print(aime25_dataset[0])
    
    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
    

if __name__ == '__main__':
    fire.Fire(main)
    
"""
# simple rl system prompt (no tool)
python examples/data_preprocess/deepmath.py --data_source zwhe99/DeepMath-103K --local_dir data/deepmath_simple_rl --sys_prompt_style simple_rl
# torl system prompt (with code interpreter tool)
python examples/data_preprocess/deepmath.py --data_source zwhe99/DeepMath-103K --local_dir data/deepmath_torl --sys_prompt_style torl
"""