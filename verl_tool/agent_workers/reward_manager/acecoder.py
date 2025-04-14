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
from collections import defaultdict

import torch

from verl import DataProto
from .reward_score import _default_compute_score


import hashlib
import random
import os
import json
import subprocess
import time
from pathlib import Path

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

    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
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
        

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        # TODO: implement new reward computing & statistic mechanism
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        
        already_print_data_sources = {}
        
        # retrieve the list of prompt_token_ids and their length
        prompt_ids = data.batch['prompts']
        prompt_length = prompt_ids.shape[-1]

        # retrieve the list of response ids and their valid length
        response_ids = data.batch['responses']
        valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(dim=-1)
        
        # batch decode the list of responses and prompts
        response_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        prompt_str = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
        
        # retrieve the list of ground truths
        ground_truth = [data_item.non_tensor_batch['reward_model']['ground_truth'] for data_item in data]
        data_sources = data.non_tensor_batch['data_source']
        
        # extract the answer for the list of responses
        extracted_answers = [re.sub(r"<think>(.|\n)*?</think>", "", response) for response in response_str]
        question_hashes = [hash_string(question) for question in prompt_str]
        
        # ensure the length of lists are of the same, avoid Ray error
        assert len(response_str) == len(ground_truth) == len(data_sources)
        
        # before perform batched scoring: dump the statistics of the list of responses
        samples = [
            {
                'task_id': question_hash,
                'prompt': question,
                'output': answer,
                'original_response': response,
                'tests': list(test_case),
                '_identifier': f"{question_hash}_{i}"
            }
            for i, (question_hash, question, answer, test_case, response) in enumerate(zip(question_hashes, prompt_str, extracted_answers, ground_truth, response_str))
        ]
        
        # save the dumped samples to a file
        temp_file = self.record_dir / f"step-{self.step_idx}_{hash_string(''.join(question_hashes))}.jsonl"
        self.step_idx += 1
        with open(temp_file, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
        
        # perform batched scoring for coding score: call the acecoder evaluation script to retrieve the coder part scores
        output_file = Path(temp_file).with_suffix(f".eval_results{'_binary' if self.binary else ''}.jsonl").absolute()
        command = f"python -m acecoder.eval_test_cases --samples {temp_file} --n_workers {self.n_workers} \
            --extract_solution True --output_file {output_file} --test_details {not self.binary} \
            --i_just_wanna_run True"
        subprocess.run(command, shell=True)
        
        # the script will dump the results into the output_file, read it and parse it as a list
        with open(output_file, "r") as f:
            all_samples_results = [json.loads(x) for x in f]
        pass_rates = [x['eval_results']['pass_rate'] for x in all_samples_results]
        
        # remove the temp_file and output_file after finish code pass rate computation and result extraction
        try:
            os.remove(temp_file)
        except:
            pass
        try:
            os.remove(output_file)
        except:
            pass
        
        
        # debugging only: save random 100 samples into a sample file
        for i, sample_result in enumerate(all_samples_results):
            sample_result['original_response'] = samples[i]['original_response']
            sample_result['question'] = samples[i]['prompt']
            sample_result['id'] = data[i].non_tensor_batch['extra_info']['id']
        num_samples = min(100, len(all_samples_results))
        sampled_results = random.sample(all_samples_results, num_samples)
        sampled_output_file = Path(temp_file).with_suffix(f".{num_samples}_samples.json").absolute()
        with open(sampled_output_file, "w") as f:
            json.dump(sampled_results, f, indent=4)
        
        coding_scores = pass_rates
        print(f"Step {self.step_idx}: {len(coding_scores)} scores computed.")
        print(f"Step {self.step_idx}: {len([x for x in coding_scores if x == 1.0])} perfect scores.")
        print(f"Step {self.step_idx}: {len([x for x in coding_scores if x == 0.0])} zero scores.")
        print(f"Step {self.step_idx}: average score: {sum(coding_scores) / len(coding_scores)}")
        
        # TODO: compute time-out score tensors
        time_out_reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        tool_call_reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        
        
        keywords = ["ERROR:\nTraceback", "Execution timed out"]
        for i, response in enumerate(response_str):
            if any(keyword in response for keyword in keywords):
                # time_out_reward_tensor[i, valid_response_length[i].item() - 1] -= 0.5
                samples[i]['add_exec_penalty'] = True
            else:
                samples[i]['add_exec_penalty'] = False
        
        # 1. compute penalty of errored or timed-out execution for each code response sample
        for i, data_item in enumerate(data):
            if "turns_stats" in data_item.non_tensor_batch:
                num_turn = data_item.non_tensor_batch["turns_stats"]
                num_valid_action = data_item.non_tensor_batch["valid_action_stats"]
                is_active = data_item.non_tensor_batch["active_mask"]
                is_done = not is_active
                samples[i]['num_turn'] = num_turn
                samples[i]['num_valid_action'] = num_valid_action
                samples[i]['is_done'] = is_done

            MIN_TOOL_CALL_CNT = 0
            MAX_TOOL_CALL_CNT = 10
            
            if num_valid_action < MIN_TOOL_CALL_CNT or num_valid_action > MAX_TOOL_CALL_CNT:
                # tool_call_reward_tensor[i, valid_response_length[i].item() - 1] -= 0.5
                samples[i]['add_tool_use_penalty'] = True
            else:
                samples[i]['add_tool_use_penalty'] = False
        
        # 3. save the records for each code response sample, which will be reported to wandb
        for i in range(len(data)):
            data_source = data_sources[i]
            
            reward_tensor[i, valid_response_length[i].item() - 1] = coding_scores[i]
            samples[i]['pass_rate'] = coding_scores[i]

            # add execution penalty to the reward tensor
            if samples[i]['add_exec_penalty']:
                reward_tensor[i, valid_response_length[i].item() - 1] -= 0.5
            # add tool call penalty to the reward tensor
            if samples[i]['add_tool_use_penalty']:
                reward_tensor[i, valid_response_length[i].item() - 1] -= 0.5
            
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[response]", response_str[i])

        if return_dict: 
            return {
                "rewad_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor