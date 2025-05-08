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
import os
import time
import json
from pathlib import Path
from verl import DataProto
from .reward_score import _default_compute_score
from .reward_score.torl_math import compute_score as torl_compute_score
import torch
from collections import defaultdict


class ToRLRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key='data_source') -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score if compute_score else _default_compute_score
        self.torl_compute_score = torl_compute_score
        self.reward_fn_key = reward_fn_key
        self.step = 0

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""
        save_record = data.meta_info.get('save_record', True)

        if not hasattr(self, 'record_dir'):
            if hasattr(self, 'run_id'):
                self.record_dir = Path(__file__).parent.parent.parent.parent / "verl_step_records" / self.run_id
                self.record_dir.mkdir(parents=True, exist_ok=True)
            else:
                self.record_dir = Path(__file__).parent.parent.parent.parent / "verl_step_records" / f"torl-{time.strftime('%Y-%m-%d-%H-%M-%S')}"
                self.record_dir.mkdir(parents=True, exist_ok=True)
                
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch['rm_scores']}
            else:
                return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}
        to_save_records = []

        for i in range(len(data)):
            score = {}
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            torl_score = self.torl_compute_score(
                # data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                # extra_info=extra_info,
            ) # 1 or -1
            score['accuracy'] = 1 if torl_score > 0 else 0
            score['score'] = torl_score
        
            # # penalty to errored or timeout execution
            # keywords = ["ERROR:\nTraceback", "Execution timed out"]
            # if any(keyword in response_str for keyword in keywords):
            #     score['exec_error'] = 1
            #     score['score'] -= 0.5
            # else:
            #     score['exec_error'] = 0
                
            # execution penalty to do
            if "turns_stats" in data_item.non_tensor_batch:
                num_turn = data_item.non_tensor_batch["turns_stats"]
                num_valid_action = data_item.non_tensor_batch["valid_action_stats"]
                is_active = data_item.non_tensor_batch["active_mask"]
                is_done = not is_active
                # add penalty to wrong response but did not use tool
                if score['accuracy'] == 0 and num_valid_action < 1:
                    # score['score'] -= 0.5
                    score['tool_use_penalty'] = 1
                else:
                    score['tool_use_penalty'] = 0
            else:
                num_turn = 0
                num_valid_action = 0
                is_done = True
            
            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
                if self.num_examine == 1:
                    reward = score["accuracy"] # for validation
            else:
                if self.num_examine == 1:
                    reward = score if score > 0 else 0.0
                else:
                    reward = score
                    

            reward_tensor[i, valid_response_length - 1] = reward 

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print(f"[score]", score)
                    
            # Save the records
            to_save_records.append({
                'id': data_item.non_tensor_batch['extra_info']['id'] if 'id' in data_item.non_tensor_batch['extra_info'] else None,
                'prompt': prompt_str,
                'data_source': data_source,
                'response': response_str,
                'ground_truth': ground_truth,
                'score': score,
                'reward': reward,
                'num_turn': num_turn,
                'num_valid_action': num_valid_action,
                'is_done': is_done,
                'extra_info': data_item.non_tensor_batch.get('extra_info', None),
            })
        
        if save_record:
            # Save the records to a file
            if self.num_examine == 1:
                temp_file = self.record_dir / f"math-step-val-{self.step}.json"
            else:
                temp_file = self.record_dir / f"math-step-{self.step}.json"
            self.step += 1
            with open(temp_file, "w") as f:
                json.dump(to_save_records, f, indent=4)
        
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
