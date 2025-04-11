# Copyright 2024 PRIME team and/or its affiliates
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

import asyncio
import regex as re
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import torch

from verl import DataProto
from verl.utils.reward_score import _default_compute_score


import hashlib
import random
import os
import json
import subprocess
import time
from pathlib import Path

import wandb

def hash_string(s):
    return hashlib.sha256(s.encode()).hexdigest()


async def single_compute_score(evaluation_func, completion, reference, task, executor, timeout=300.):
    loop = asyncio.get_running_loop()
    try:
        # Ensure process_completion is called properly
        tasks = [
            asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    partial(evaluation_func, task, completion, reference)  # Ensure synchronous
                ),
                timeout=timeout)
        ]
        return await asyncio.gather(*tasks)
    except asyncio.TimeoutError:
        print(f"Timeout occurred for completion: {completion}")
        return None  # Default value for timed-out rows
    except Exception as e:
        print(f"Error processing completion: {completion[:10]}, Error: {e}")
        return None  # Default value for failed rows


async def parallel_compute_score_async(evaluation_func, completions, references, tasks, num_processes=64):
    scores = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Create tasks for all rows
        tasks_async = [
            single_compute_score(evaluation_func, completion, reference, task, executor, timeout=300.)
            for completion, reference, task in zip(completions, references, tasks)
        ]
        # to prevent very occasional starvation caused by some anomalous programs ( like infinite loop ), the exceptions in async programs will instantly halt the evaluation, and all summoned processes will be killed.
        try:
            results = await asyncio.gather(*tasks_async, return_exceptions=False)
        except:
            for pid, proc in executor._processes.items():
                try:
                    proc.kill()
                except Exception as kill_err:
                    print('shut down failed: ' + str(kill_err))
            raise

    # Process results
    for result, completion, reference, task in zip(results, completions, references, tasks):
        if isinstance(result, Exception) or result is None:
            # Handle failed or timed-out tasks
            scores.append(0.0)
        elif isinstance(result[0], (int, float, bool)):
            scores.append(float(result[0]))
        else:
            scores.append(float(result[0][0]))
    return scores


class AceCoderRewardManager:
    """
    The Reward Manager used in https://github.com/TIGER-AI-Lab/AceCoder
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, config=None) -> None:
        self.rm_config = config
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.step_idx = 0
        self.n_workers = 64
        self.binary = False
        self.run_id = os.getenv("VERL_RUN_ID", f"acecoder_{time.strftime('%Y-%m-%d-%H-%M-%S')}")
        self.record_dir = Path(__file__).parent.parent.parent.parent / "verl_step_records" / self.run_id
        self.record_dir.mkdir(parents=True, exist_ok=True)
        try:
            from acecoder import evaluate_test_cases
        except ImportError:
            raise ImportError("`from acecoder import evaluate_test_cases` failed, please install acecoder to use test_case rule")
        

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        # batched scoring
        prompt_ids = data.batch['prompts']
        prompt_length = prompt_ids.shape[-1]

        response_ids = data.batch['responses']
        valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(dim=-1)
        response_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        prompt_str = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
        ground_truth = [data_item.non_tensor_batch['reward_model']['ground_truth'] for data_item in data]
        data_sources = data.non_tensor_batch['data_source']
        
        extracted_answers = [re.sub(r"<think>(.|\n)*?</think>", "", response) for response in response_str]
        question_hashes = [hash_string(question) for question in prompt_str]
        assert len(response_str) == len(ground_truth) == len(data_sources)
        
        # TODO: process response strs. count the number of tool calls, if the number >= min and <= max then reward it for 1.
        tool_call_counts = [len(re.findall(r"<python>(.|\n)*?</python>", response)) for response in response_str]
        tool_use_min = self.rm_config.reward_model.tool_use_min_calls
        tool_use_max = self.rm_config.reward_model.tool_use_max_calls
        tool_call_rewards = [1.0 if tool_use_min <= count <= tool_use_max else 0.0 for count in tool_call_counts]
        
        # TODO: report average_toll_call_cnt to wandb
        avg_tool_call_cnt = sum(tool_call_counts) / len(tool_call_counts)
        print(f"[DEBUG] average tool call counts: {avg_tool_call_cnt}")
        wandb.log({"toll_use/avg_tool_call_cnt": avg_tool_call_cnt})
        
        samples = [
            {
                'task_id': question_hash,
                'prompt': question,
                'output': answer,
                'original_response': response,
                'tests': list(test_case),
                '_identifier': f"{question_hash}_{i}",
                "number_of_tool_calls": tool_call_counts[i],
            }
            for i, (question_hash, question, answer, test_case, response) in enumerate(zip(question_hashes, prompt_str, extracted_answers, ground_truth, response_str))
        ]
        ## save samples to a file
        temp_file = self.record_dir / f"step-{self.step_idx}_{hash_string(''.join(question_hashes))}.jsonl"
        self.step_idx += 1
        with open(temp_file, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
        output_file = Path(temp_file).with_suffix(f".eval_results{'_binary' if self.binary else ''}.jsonl").absolute()
        command = f"python -m acecoder.eval_test_cases --samples {temp_file} --n_workers {self.n_workers} \
            --extract_solution True --output_file {output_file} --test_details {not self.binary} \
            --i_just_wanna_run True"
        subprocess.run(command, shell=True)
        with open(output_file, "r") as f:
            all_samples_results = [json.loads(x) for x in f]
        pass_rates = [x['eval_results']['pass_rate'] for x in all_samples_results]
        
        # remove temp_file
        try:
            os.remove(temp_file)
        except:
            pass
        try:
            os.remove(output_file)
        except:
            pass
        # save random 100 samples into a file for debugging
        for i, sample_result in enumerate(all_samples_results):
            sample_result['original_response'] = samples[i]['original_response']
            sample_result['question'] = samples[i]['prompt']
            sample_result['id'] = data[i].non_tensor_batch['extra_info']['id']
        num_samples = min(100, len(all_samples_results))
        sampled_results = random.sample(all_samples_results, num_samples)
        sampled_output_file = Path(temp_file).with_suffix(f".{num_samples}_samples.json").absolute()
        with open(sampled_output_file, "w") as f:
            json.dump(sampled_results, f, indent=4)
        
        scores = pass_rates
        print(f"Step {self.step_idx}: {len(scores)} scores computed.")
        print(f"Step {self.step_idx}: {len([x for x in scores if x == 1.0])} perfect scores.")
        print(f"Step {self.step_idx}: {len([x for x in scores if x == 0.0])} zero scores.")
        print(f"Step {self.step_idx}: average score: {sum(scores) / len(scores)}")
        
        for i in range(len(data)):
            data_source = data_sources[i]
            
            # TODO: add tool_use rewards
            # reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]
            reward_tensor[i, valid_response_length[i].item() - 1] = scores[i] + tool_call_rewards[i]
            
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[response]", response_str[i])

        return reward_tensor