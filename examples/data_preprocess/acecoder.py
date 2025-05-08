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
Answer the given coding question. You must conduct reasoning about the problem and then provide the final program as answer. 
During the thinking process, you can write test cases or test your current solutions using a testing tool. if you want to test any python code, writing it inside ```python and ``` tags following with "```output". 
The code between "```python" and "``````output" will then be executed, and the terminal output (standard output and standard error) will be provided to you. 
Each program between ```python and ``` tags are independent program. You can test Python codes as many times as you want. 
If you find no further code execution needed, you can then give your final solution in a markdown code block like this: ```python\nyour code here\n``` without appending anything. 
The final program will be evaluated against the hidden test cases. If the final program passes all the test cases, you will get a reward. If the final program fails any of the test cases, you will get a penalty.
"""

naive_instruction = "Let's think step by step and generate the final program in a markdown code block like this: ```python\nyour code here\n```."
naive_execution_prompt = """
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The Assistant can reason with the help of Python code. If the Assistant wants to run any Python code, it writes it inside ```python and ``` tags, and makes sure to follow it with "```output", meaning that it is requesting the code to be executed. Then the result of execution will be provided to the Assistant between "```output" and "```" for the python code block that it follows. The Assistant can test Python codes as many times as it wants. If the Assistant finds no further code execution needed, it can then give the final solution in a markdown code block like this: ```python\nyour code here\n``` without appending anything.
"""

r1_naive_execution_prompt = """\
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. If the Assistant wants to run any Python code when thinking, it writes it inside "```python" and "```" tags, and makes sure to have "```output" after the python code block, meaning that it is requesting the code to be executed. Then the result of execution will in the output markdown block.
"""

complex_execution_prompt = """
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The Assistant can reason with the help of Python code. If the Assistant wants to run any Python code, it writes it inside "```python" and "```" tags, and makes sure to have "```output" after the python code block, meaning that it is requesting the code to be executed. Then the result of execution will in the output markdown block.

Coding questions can ask various forms of program solutions:
- If the coding question has a starter code, you may use the starter code to write the solution to the problem.
- Elif the coding question has a function signature, you may use the function signature to write the solution to the problem.
- Else you may write a program that reads the input from standard input and writes the output to standard output. (do not directly test on the sample inputs)

The Assistant can test Python codes as many times as it wants. If the Assistant finds no further code execution needed, it can then give the final solution in a markdown code block like this: ```python\nyour code here\n``` without appending anything. 
"""
r1_complex_execution_prompt = """\
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. If the Assistant wants to run any Python code when thinking, it writes it inside "```python" and "```" tags, and makes sure to have "```output" after the python code block, meaning that it is requesting the code to be executed. Then the result of execution will in the output markdown block.

Coding questions can ask various forms of program solutions:
- If the coding question has a starter code, you may use the starter code to write the solution to the problem.
- Elif the coding question has a function signature, you may use the function signature to write the solution to the problem.
- Else you may write a program that reads the input from standard input and writes the output to standard output. (do not directly test on the sample inputs)

The Assistant can test Python codes as many times as it wants. If the Assistant finds no further code execution needed, it can then give the final solution in a markdown code block like this: ```python\nyour code here\n``` without appending anything. 
"""

# naive_execution_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. User: Please integrate natural language reasoning with programs to solve the coding problems below. If you want to test any python code, writing it inside <python> and </python> tags, results will be inside <output> and </output>. Please put your final answer in a markdown code block like this: python\nyour code here\n``` without appending anything."""

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
naive_coder_instruction = """\
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. 

Let's think step by step and generate the final program in a markdown code block like this: ```python\nyour code here\n```.
"""

r1_naive_coder_instruction = """\
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
"""

complex_coder_instruction = """\
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. 

Coding questions can ask various forms of program solutions:
- If the coding question has a starter code, you may use the starter code to write the solution to the problem.
- Elif the coding question has a function signature, you may use the function signature to write the solution to the problem.
- Else you may write a program that reads the input from standard input and writes the output to standard output. (do not directly test on the sample inputs)

Let's think step by step and generate the final program in a markdown code block like this: ```python\nyour code here\n```.
"""
r1_complex_coder_instruction = """\
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.

Coding questions can ask various forms of program solutions:
- If the coding question has a starter code, you may use the starter code to write the solution to the problem.
- Elif the coding question has a function signature, you may use the function signature to write the solution to the problem.
- Else you may write a program that reads the input from standard input and writes the output to standard output. (do not directly test on the sample inputs)
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
    add_execution_prompt: bool = False,
    propmt_type='complex',
    add_public_tests: bool = False,
    add_r1: bool = False,
):
    local_dir = Path(local_dir)
    local_dir_post_fix = ""
    if add_execution_prompt:
        local_dir_post_fix = "-with-execution-prompt"
    if add_public_tests:
        local_dir_post_fix += "-with-public-tests"
    if add_r1:
        local_dir_post_fix += "-r1"
    local_dir_post_fix += f"-{propmt_type}"
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

            if propmt_type == 'complex':
                if add_r1:
                    system_instruction = r1_complex_execution_prompt if add_execution_prompt else r1_complex_coder_instruction
                else:
                    system_instruction = complex_execution_prompt if add_execution_prompt else complex_coder_instruction
            elif propmt_type == 'naive':
                if add_r1:
                    system_instruction = r1_naive_execution_prompt if add_execution_prompt else r1_naive_coder_instruction
                else:
                    system_instruction = naive_execution_prompt if add_execution_prompt else naive_coder_instruction
            else:
                raise ValueError(f"Unknown propmt_type: {propmt_type}")
            
            if add_public_tests:
                # system_instruction = system_instruction + "\n" + "Note that there may or may not be public test cases for this question. If there are, you can use them to test your program and even write more test cases by your own based on the public test cases to test your program. If there are no public test cases, you can write your own test cases to test your program. Put your test cases in the same markdown code block as your program to be executed so you can see the output of your test cases."
                public_tests = example.pop('public_tests')
                if public_tests:
                    public_tests_str = "\n".join(public_tests)
                    public_tests_str = public_test_template.format(test_cases=public_tests_str)
                    # question_raw = f"{question_raw}\n\n{public_tests_str}"
                    
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
                    "public_tests": public_tests if add_public_tests else None,
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
python examples/data_preprocess/acecoder.py --dataset_path CodeDPO/AceCoderV2-mini-processed --local_dir data/acecoder --add_execution_prompt
python examples/data_preprocess/acecoder.py --dataset_path VerlTool/AceCoderV2-69K --local_dir data/acecoder --add_execution_prompt
python examples/data_preprocess/acecoder.py --dataset_path CodeDPO/AceCoderV2-150K-processed --local_dir data/acecoder --add_execution_prompt

python examples/data_preprocess/acecoder.py --dataset_path VerlTool/AceCoderV2-69K --local_dir data/acecoder_naive --add_execution_prompt

python examples/data_preprocess/acecoder.py --dataset_path VerlTool/AceCoderV2-69K --local_dir data/acecoder_long --add_execution_prompt --propmt_type complex

python examples/data_preprocess/acecoder.py --dataset_path VerlTool/AceCoderV2-69K --local_dir data/acecoder_long --add_execution_prompt --propmt_type complex --add_public_tests True --add_r1 False
python examples/data_preprocess/acecoder.py --dataset_path VerlTool/AceCoderV2-69K --local_dir data/acecoder_long --add_execution_prompt --propmt_type complex --add_public_tests True --add_r1 True
python examples/data_preprocess/acecoder.py --dataset_path VerlTool/AceCoderV2-69K --local_dir data/acecoder_long --add_execution_prompt --propmt_type naive --add_public_tests True --add_r1 True


python examples/data_preprocess/acecoder.py --dataset_path VerlTool/AceCoderV2-122K --local_dir data/acecoderv2 --add_execution_prompt True --propmt_type complex
python examples/data_preprocess/acecoder.py --dataset_path VerlTool/AceCoderV2-122K --local_dir data/acecoderv2 --add_execution_prompt True --propmt_type naive
python examples/data_preprocess/acecoder.py --dataset_path VerlTool/AceCoderV2-122K --local_dir data/acecoderv2 --add_execution_prompt False --propmt_type complex
python examples/data_preprocess/acecoder.py --dataset_path VerlTool/AceCoderV2-122K --local_dir data/acecoderv2 --add_execution_prompt False --propmt_type naive
"""