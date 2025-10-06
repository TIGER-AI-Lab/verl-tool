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
import regex as re
from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string
from verl.utils.hdfs_io import copy, makedirs

def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))

    
PROMPT="""Solve the following problem step by step. You now have the ability to selectively write executable Python code to enhance your reasoning process. The Python code will be executed by an external sandbox, and the output (after "Code execution result: ") is returned to aid your reasoning and help you arrive at the final answer. The Python code should be complete scripts, including necessary imports.

Code Format:
Each code snippet is wrapped between ```. You need to use `print()` to output intermediate results.

Answer Format:
You should use \\boxed to return your answer. The last part of your response should be:
\\boxed{'The final answer goes here.'}

User Question:
"""


def main(
    data_source='agentica-org/DeepScaleR-Preview-Dataset',
    local_dir='~/data/deepscaler',
    hdfs_dir=None,
):
    
    # deepscaler
    data_source = 'agentica-org/DeepScaleR-Preview-Dataset'
    dataset = datasets.load_dataset(data_source, trust_remote_code=True, split='train')
    # add a row to each data item that represents a unique id
    def make_map_fn(split, data_source):

        def process_fn(example, idx):
            question = example.pop('problem')
            answer = example.pop('answer')
            
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": PROMPT + question}],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": str(answer)
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'question': question,
                }
            }
            return data

        return process_fn
    
    deepscaler_train_dataset = dataset.map(function=make_map_fn('train', data_source), with_indices=True, remove_columns=dataset.column_names)
    deepscaler_train_dataset.to_parquet(os.path.join(local_dir, 'deepscaler_train.parquet'))
    print(deepscaler_train_dataset)
    
    # math
    
    data_source='DongfuJiang/math_35'
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    train_dataset = datasets.load_dataset(data_source, trust_remote_code=True, split='train')

    def make_map_fn(split, data_source):

        def process_fn(example, idx):
            
            question = example.pop('problem')
            answer = example.pop('answer')
            
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": PROMPT + question}],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": str(answer)
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'question': question,
                }
            }
            return data

        return process_fn
    
    math_train_dataset = train_dataset.map(function=make_map_fn('train', data_source), with_indices=True, remove_columns=train_dataset.column_names)
    math_train_dataset.to_parquet(os.path.join(local_dir, 'math35_train.parquet'))
    
    # math 500
    
    
    def make_map_fn(split, data_source):
        
        def process_fn(example, idx):
            question = example.pop('problem')
            answer = example.pop('solution')
            solution = extract_solution(answer)
            assert solution != "", f"Cannot find solution in answer: {answer}"
            
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": PROMPT + question}],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": str(solution)
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'question': question,
                }
            }
            return data

        return process_fn
    
    data_source='HuggingFaceH4/MATH-500'
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True, split='test')
    math500_test_dataset = dataset.map(function=make_map_fn('test', data_source), with_indices=True, remove_columns=dataset.column_names)
    math500_test_dataset.to_parquet(os.path.join(local_dir, 'math500_test.parquet'))
    
    # aime
    def make_map_fn(split, data_source):
        
        def process_fn(example, idx):
            question = example.pop('problem')
            answer = example.pop('answer')
            
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": PROMPT + question}],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": str(answer)
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'question': question,
                }
            }
            return data

        return process_fn
    
    aime25_data_source = 'MathArena/aime_2025'
    print(f"Loading the {aime25_data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(aime25_data_source, trust_remote_code=True, split='train')
    aime25_dataset = dataset.map(function=make_map_fn('test', aime25_data_source), with_indices=True, remove_columns=dataset.column_names)
    aime25_dataset.to_parquet(os.path.join(local_dir, 'aime25.parquet'))
    
    aime24_data_source = 'MathArena/aime_2024'
    dataset_1 = datasets.load_dataset(aime24_data_source+"_I", trust_remote_code=True, split='train')
    dataset_2 = datasets.load_dataset(aime24_data_source+"_II", trust_remote_code=True, split='train')
    dataset = datasets.concatenate_datasets([dataset_1, dataset_2])
    aime24_dataset = dataset.map(function=make_map_fn('test', aime24_data_source), with_indices=True, remove_columns=dataset.column_names)
    aime24_dataset.to_parquet(os.path.join(local_dir, 'aime24.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
    

if __name__ == '__main__':
    fire.Fire(main)
    
"""
python examples/train/simpletir/prepare_data.py --local_dir data/simpletir
"""