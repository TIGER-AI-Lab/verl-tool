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
import hashlib
import random
import os
import json
import subprocess
import time
import regex as re
from pathlib import Path
import uuid

import torch
from collections import defaultdict
from verl import DataProto
from verl.protocol import collate_fn
from .reward_score import _default_compute_score
from verl.workers.reward_manager import register

def hash_string(s):
    return hashlib.sha256(s.encode()).hexdigest()

from .torl import ToRLRewardManager
from .acecoder import AceCoderRewardManager

@register("mathcoder")
class MathCoderRewardManager:
    def __init__(
        self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score if compute_score else _default_compute_score
        self.reward_fn_key = reward_fn_key
        
        self.step = 0
        self.ToRLRewardManager = ToRLRewardManager(
            tokenizer, num_examine, compute_score, reward_fn_key
        )
        self.AceCoderRewardManager = AceCoderRewardManager(
            tokenizer, num_examine, compute_score
        )

    def __call__(self, data: DataProto, return_dict=False):
        save_record = data.meta_info.get('save_record', True)
        
        if not hasattr(self, 'record_dir'):
            if hasattr(self, 'run_id'):
                self.record_dir = Path(__file__).parent.parent.parent.parent / "verl_step_records" / self.run_id
                self.record_dir.mkdir(parents=True, exist_ok=True)
            else:
                self.record_dir = Path(__file__).parent.parent.parent.parent / "verl_step_records" / f"mathcoder-{time.strftime('%Y-%m-%d-%H-%M-%S')}"
                self.record_dir.mkdir(parents=True, exist_ok=True)
        
        # check the last step index
        if self.step is None:
            last_step_idx = 0
            for file in os.listdir(self.record_dir):
                if self.num_examine == 1:
                    if re.search(r"step-val-\d+\.json", file):
                        step_idx = int(file[:-len(".json")].split("-")[-1])
                        if step_idx > last_step_idx:
                            last_step_idx = step_idx
                else:
                    if re.search(r"step-\d+\.json", file):
                        step_idx = int(file[:-len(".json")].split("-")[-1])
                        if step_idx > last_step_idx:
                            last_step_idx = step_idx
            self.step = last_step_idx + 1
        if data.meta_info.get('global_step', None) is not None:
            self.step = data.meta_info['global_step']

        to_save_records = []
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        # reward extra info every key of it is a default len(data) list filled with None
        reward_extra_info = defaultdict(
            lambda: [None] * len(data)
        )
        prompt_ids = data.batch['prompts']
        prompt_length = prompt_ids.shape[-1]
        response_ids = data.batch['responses']
        valid_prompt_length = data.batch['attention_mask'][:, :prompt_length].sum(dim=-1)
        valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(dim=-1)
        
        code_data_idxs = [
            i for i in range(len(data)) if data[i].non_tensor_batch["ability"] == "code"
        ]
        math_data_idxs = [
            i for i in range(len(data)) if data[i].non_tensor_batch["ability"] == "math"
        ]
        code_data = data[code_data_idxs]
        math_data = data[math_data_idxs]
        code_data.meta_info['save_record'] = False
        math_data.meta_info['save_record'] = False
        code_reward = self.AceCoderRewardManager(code_data, return_dict=True)
        math_reward = self.ToRLRewardManager(math_data, return_dict=True)
        print("len code_reward", len(code_reward['reward_tensor']))
        print("len math_reward", len(math_reward['reward_tensor']))
        # put the code and math reward together in the original order
        reward_tensor[code_data_idxs] = code_reward['reward_tensor']
        reward_tensor[math_data_idxs] = math_reward['reward_tensor']
        
        for k, v in code_reward['reward_extra_info'].items():
            if k not in reward_extra_info:
                for i in range(len(v)):
                    reward_extra_info[f"code_{k}"][code_data_idxs[i]] = v[i]
        for k, v in math_reward['reward_extra_info'].items():
            if k not in reward_extra_info:
                for i in range(len(v)):
                    reward_extra_info[f"math_{k}"][math_data_idxs[i]] = v[i]
        reward_extra_keys = list(reward_extra_info.keys())
        scores = [{key: reward_extra_info[key][i] for key in reward_extra_keys} for i in range(len(data))]
        
        # Save the records
        ground_truths = [
            data_item.non_tensor_batch["reward_model"]["ground_truth"]
            for data_item in data
        ]
        if "turns_stats" in data.non_tensor_batch:
            num_turn = data.non_tensor_batch["turns_stats"]
            num_valid_action = data.non_tensor_batch["valid_action_stats"]
            is_active = data.non_tensor_batch["active_mask"]
            is_done = [not is_active[i] for i in range(len(is_active))]

        data_source = data.non_tensor_batch[self.reward_fn_key]
        
        if save_record:
            raw_score = [reward_extra_info['math_accuracy'][i] if data[i].non_tensor_batch['ability'] == 'math' else \
                reward_extra_info['code_binary_pass_rate'][i] for i in range(len(data))]
            to_save_records = [
                {
                    "id": data[i].non_tensor_batch['extra_info']['id'] if 'id' in data[i].non_tensor_batch['extra_info'] else None,
                    "data_source": data_source[i],
                    "prompt": self.tokenizer.decode(prompt_ids[i][-valid_prompt_length[i].item():], skip_special_tokens=False),
                    "response": self.tokenizer.decode(response_ids[i][:valid_response_length[i].item()], skip_special_tokens=False),
                    "ground_truth": ground_truths[i],
                    "score": scores[i],
                    'extra_info': data[i].non_tensor_batch.get('extra_info', None),
                }
                for i in range(len(data))
            ]
            if "turns_stats" in data.non_tensor_batch:
                for i, record in enumerate(to_save_records):
                    to_save_records[i]['num_turn'] = num_turn[i]
                    to_save_records[i]['num_valid_action'] = num_valid_action[i]
                    to_save_records[i]['is_done'] = is_done[i]
            
            # Save the records to a file
            if self.num_examine == 1:
                temp_file = self.record_dir / f"mathcoder-step-val-{self.step}.json"
            else:
                temp_file = self.record_dir / f"mathcoder-step-{self.step}.json"
            self.step += 1
            with open(temp_file, "w") as f:
                json.dump(to_save_records, f, indent=4)
            
        if self.num_examine == 1:
            # for validation, empty the reward_extra_info, becuase there are None items and cannot be mean
            reward_extra_info = defaultdict(list)
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor