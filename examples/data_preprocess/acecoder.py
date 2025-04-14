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
import fire
import os
import datasets
from pathlib import Path

execution_prompt = """\
Answer the given coding question. You must conduct reasoning inside <think> and </think> first before you can finally output the final program. During the thinking, you can test your program by writing it inside ```python and ``` tags following with "```output". The code will be executed, and the terminal output (standard output and standard error) will be returned between <output> and </output>. Each program between ```python and ``` tags are independent program. You can test Python codes as many times as you want. If you find no further code execution needed, you can then give the final program in a markdown code block like this: ```python\nyour code here\n``` without appending anything,. The final program will be evaluated against the hidden test cases. If the final program passes all the test cases, you will get a reward. If the final program fails any of the test cases, you will get a penalty.
"""

naive_instruction = "Let's think step by step and generate the final program in a markdown code block like this: ```python\nyour code here\n```."

coder_instruction = """\
Let's think step by step and generate the correct program for this coding question. You should attempt multiple times before give the final program.
In each attempt, you should 
- test your program by reviewing the code syntax and logic, and fix any potential issues in the next attempt.
- imagine a set of test cases based on your understanding of the problem and the constraints. 
- You then need to test your program with these test cases. Since you are not able to run the program in a real environment, you need to use text to simulate the program running and think loudly to describe how each variable changes during the execution. Finally, see whether the program produces the expected output.
- if the program fails any of the test cases, you need to debug the program and fix the issues in the next attempt.
- if the program passes all the test cases, you can then give the final program in a markdown code block like this: ```python\nyour code here\n```.

You are also allowed to analyze the problem with any other domain-specific knowledge you have, like math, physics, etc to help you solve the problem.

Now start thinking and generate the final program in a markdown code block like this: ```python\nyour code here\n```.
"""

def main(
    dataset_path: str = 'CodeDPO/AceCoderV2-mini-processed',
    local_dir: str = 'data/acecoder',
    add_execution_prompt: bool = False,
    detaield_instruction: bool = False
):
    local_dir = Path(local_dir) / dataset_path.split('/')[-1]
    if add_execution_prompt:
        local_dir = local_dir.parent / (local_dir.name + '-with-execution-prompt')
    if detaield_instruction:
        local_dir = local_dir.parent / (local_dir.name + '-detailed')
    local_dir.mkdir(parents=True, exist_ok=True)

    dataset = datasets.load_dataset(dataset_path, split='train')

    # 500 examples for testing
    
    dataset = dataset.train_test_split(test_size=500, seed=42)
    train_dataset = dataset['train']
    test_dataset = dataset['test']

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question_raw = example.pop('question')

            # if not add_execution_prompt:
            #     if not detaield_instruction:
            #         question = question_raw + ' ' + naive_instruction
            #     else:
            #         question = question_raw + ' ' + coder_instruction
            # else:
            #     question = question_raw + ' ' + execution_prompt
            
            tests = example.pop('tests')
            data = {
                "data_source": "acecoder",
                "prompt": [
                    {
                        "role": "system",
                        "content": execution_prompt if add_execution_prompt else coder_instruction,
                    },
                    {
                        "role": "user",
                        "content": question_raw,
                    }
                ],
                "ability": "code",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": tests
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'id': example['id'],
                    "question": question_raw,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    
    print(f"Loaded {len(train_dataset)} training samples")
    print(f"Loaded {len(test_dataset)} testing samples")
    print(f"Example of a training sample:")
    print(train_dataset[0])

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    print(f"Saved to {len(train_dataset)} training samples to {local_dir}/train.parquet")
    print(f"Saved to {len(test_dataset)} testing samples to {local_dir}/test.parquet")

if __name__ == '__main__':
    fire.Fire(main)
    
"""
python examples/data_preprocess/acecoder.py --dataset_path CodeDPO/AceCoderV2-mini-processed --local_dir data/acecoder --add_execution_prompt
python examples/data_preprocess/acecoder.py --dataset_path CodeDPO/AceCoderV2-150K-processed --local_dir data/acecoder --add_execution_prompt
"""