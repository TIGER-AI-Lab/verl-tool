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

system_prompt1 = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. User: Please integrate natural language reasoning with programs to solve the coding problems below. If you want to test any python code, writing it inside <python> and </python> tags following with <output>. Please put your final answer in a markdown code block like this: python\nyour code here\n``` without appending anything.
"""

system_prompt2 = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. User: Please integrate natural language reasoning with programs to solve the coding problems below. If the you want to run any python code, execution result will be in the output markdown block like "```output\nexecution result here\n```" following the code block. Please put your final answer in a markdown code block like this: python\nyour code here\n``` without appending anything.
"""

system_prompt3 = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. User: Please solve the coding problems below and put your final answer in a markdown code block like this: python\nyour code here\n``` without appending anything.
"""
    

public_test_template = """\
### Public Test Cases
Here are some public test cases where you can use to test your program.
```python
{test_cases}
```
"""
def main(
    dataset_path: str = 'VerlTool/AceCoderV2-122K',
    local_dir: str = 'data/acecoder',
    system_prompt_idx: int = 1,
):
    local_dir = Path(local_dir)
    local_dir_post_fix = f"-system-prompt-{system_prompt_idx}"
    local_dir = local_dir / (dataset_path.split('/')[-1] + local_dir_post_fix)
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

            system_instruction = eval(f'system_prompt{system_prompt_idx}')
                    
            tests = example.pop('tests')
            data = {
                "data_source": dataset_path,
                "prompt": [
                    {
                        "role": "system",
                        "content": system_instruction.strip(' \n'),
                    },
                    {
                        "role": "user",
                        "content": question_raw,
                    }
                ],
                "ability": "code",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ""
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'id': str(example['id']),
                    "question": question_raw,
                    "public_tests": None,
                    "test_cases": tests,
                    "inputs_outputs": None,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True, remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True, remove_columns=test_dataset.column_names)
    
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
python examples/data_preprocess/acecoder_custom.py --dataset_path VerlTool/AceCoderV2-69K --local_dir data/acecoder_custom --system_prompt_idx 1
python examples/data_preprocess/acecoder_custom.py --dataset_path VerlTool/AceCoderV2-69K --local_dir data/acecoder_custom --system_prompt_idx 2
python examples/data_preprocess/acecoder_custom.py --dataset_path VerlTool/AceCoderV2-69K --local_dir data/acecoder_custom --system_prompt_idx 3
python examples/data_preprocess/acecoder_custom.py --dataset_path VerlTool/AceCoderV2-122K --local_dir data/acecoder_custom --system_prompt_idx 1
python examples/data_preprocess/acecoder_custom.py --dataset_path VerlTool/AceCoderV2-122K --local_dir data/acecoder_custom --system_prompt_idx 2
"""