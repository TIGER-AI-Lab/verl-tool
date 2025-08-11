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
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import Tuple

import torch
from collections import defaultdict
from verl import DataProto
from verl.protocol import collate_fn
from .reward_score import _default_compute_score
from verl.workers.reward_manager import register
from verl_tool.servers.tools.utils.sql_executor import score


THINK_START, THINK_END = "<think>", "</think>"
SQL_START, SQL_END = "<sql>", "</sql>"
SOLUTION_START, SOLUTION_END = "<solution>", "</solution>"
OBS_START, OBS_END = "<observation>", "</observation>"



def parse_action(action: str, tag_type: str = "sql") -> Tuple[str, bool]:
    """
    Parse the raw action string to extract SQL code from either <sql></sql> or <solution></solution> tags.
    
    Args:
        action: Raw action string containing SQL code
        tag_type: Type of tag to extract ("sql" or "solution")
        
    Returns:
        Tuple containing the extracted code and a validity flag
    """
    tag_start_map = {
        "sql": SQL_START,
        "solution": SOLUTION_START
    }
    tag_end_map = {
        "sql": SQL_END,
        "solution": SOLUTION_END
    }

    # Find the last occurrence of the start tag
    start_tag = tag_start_map[tag_type]
    end_tag = tag_end_map[tag_type]
    
    sql_code_start_idx = action.rfind(start_tag)
    if sql_code_start_idx == -1:
        return "", False
    
    # Find the corresponding end tag after the start tag
    sql_code_end_idx = action.find(end_tag, sql_code_start_idx + len(start_tag))
    if sql_code_end_idx == -1:
        return "", False
    
    # Extract the content between the tags
    sql_code = action[sql_code_start_idx + len(start_tag):sql_code_end_idx].strip()
    return sql_code, True

# Copied from SkyRL-SQL/skyrl_gym/envs/sql/utils.py
def verify_format_and_extract(output: str):
    if output.count(SOLUTION_START) != 1:
        return False, None, None, None
    pre_solution, tail = output.split(SOLUTION_START, 1)

    if tail.count(SOLUTION_END) != 1:
        return False, None, None, None

    solution_text, _ = tail.split(SOLUTION_END, 1)

    if re.search(r"</?(think|sql|observation)\b", solution_text, re.I):
        return False, None, None, None

    thoughts = re.findall(r"<think>(.*?)</think>", output, re.S)
    if not thoughts:
        return False, None, None, None

    for m in re.finditer(r"</observation>", pre_solution, re.I):
        rest = pre_solution[m.end() :].lstrip()
        if not rest.lower().startswith(THINK_START):
            return False, None, None, None

    return True, thoughts, solution_text.strip(), None


def hash_string(s):
    return hashlib.sha256(s.encode()).hexdigest()

class AsyncJSONLWriter:
    """Async JSONL writer that saves records without blocking the main process."""
    
    def __init__(self):
        self.write_queue = Queue()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jsonl_writer")
        self.running = True
        
    def _write_worker(self, file_path, records):
        """Worker function to write records to JSONL file."""
        try:
            with open(file_path, 'a', encoding='utf-8') as f:
                for record in records:
                    json_line = json.dumps(record, ensure_ascii=False)
                    f.write(json_line + '\n')
            print(f"===> Successfully saved {len(records)} records to {file_path}")
        except Exception as e:
            print(f"===> Error saving records to {file_path}: {str(e)}")
    
    def save_async(self, file_path, records):
        """Queue records for async saving."""
        if self.running:
            future = self.executor.submit(self._write_worker, file_path, records)
            return future
        return None
    
    def shutdown(self):
        """Shutdown the async writer gracefully."""
        self.running = False
        self.executor.shutdown(wait=True)

@register("sqlcoder")
class SQLCoderRewardManager:
    def __init__(
        self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score if compute_score else _default_compute_score
        self.reward_fn_key = reward_fn_key
        
        self.step = 0
        
        # Initialize async JSONL writer
        self.async_writer = AsyncJSONLWriter()
        

    def __call__(self, data: DataProto, return_dict=False):
        save_record = data.meta_info.get('save_record', True)
        
        if not hasattr(self, 'record_dir'):
            if hasattr(self, 'run_id'):
                self.record_dir = Path(__file__).parent.parent.parent.parent / "verl_step_records" / self.run_id
                self.record_dir.mkdir(parents=True, exist_ok=True)
            else:
                self.record_dir = Path(__file__).parent.parent.parent.parent / "verl_step_records" / f"sqlcoder-{time.strftime('%Y-%m-%d-%H-%M-%S')}"
                self.record_dir.mkdir(parents=True, exist_ok=True)
        
        # check the last step index - updated for JSONL files
        if self.step is None:
            last_step_idx = 0
            for file in os.listdir(self.record_dir):
                if self.num_examine == 1:
                    if re.search(r"step-val-\d+\.jsonl", file):
                        step_idx = int(file[:-len(".jsonl")].split("-")[-1])
                        if step_idx > last_step_idx:
                            last_step_idx = step_idx
                else:
                    if re.search(r"step-\d+\.jsonl", file):
                        step_idx = int(file[:-len(".jsonl")].split("-")[-1])
                        if step_idx > last_step_idx:
                            last_step_idx = step_idx
            self.step = last_step_idx + 1
        if data.meta_info.get('global_step', None) is not None:
            self.step = data.meta_info['global_step']

        to_save_records = []
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        
        # reward extra info every key of it is a default len(data) list filled with None
        prompt_ids = data.batch['prompts']
        prompt_length = prompt_ids.shape[-1]
        response_ids = data.batch['responses']
        valid_prompt_length = data.batch['attention_mask'][:, :prompt_length].sum(dim=-1)
        valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(dim=-1)
        non_tensor_batch = data.non_tensor_batch # dict
        
        final_rewards = []
        format_scores = []
        execution_scores = []
        positive_rewards = []
        
        for i in range(len(data)):
            # Get the entire response for format checking
            entire_block = data.batch['responses'][i]
            entire_block_decoded = self.tokenizer.decode(entire_block, skip_special_tokens=False)
            # Get database and ground truth information
            extra_info = data[i].non_tensor_batch.get('extra_info', {})
            meta = {
                "db_id": extra_info.get("db_id"),
                "gold_sql": extra_info.get("gt_sql"),
                "cmp_method": "bird",
                "db_path": extra_info.get("db_path")
            }
            
            is_valid, thoughts, parsed_solution, _ = verify_format_and_extract(entire_block_decoded)
            final_reward = -1.0
            
            if is_valid and parsed_solution:
                try:
                    execution_score = score(parsed_solution, meta)
                    final_reward = execution_score
                except Exception as e:
                    execution_score = 0.0
                    print(f"Execution error for trajectory {i}: {str(e)}")
            elif not is_valid:
                # Format validation failed - assign penalty (same as SkyRL)
                final_reward = -1.0
            format_score = 0.0 if is_valid else -1.0
            execution_score = final_reward if is_valid else 0.0
            
            
            if final_reward > 0:
                positive_rewards.append(1)
            
            if self.num_examine == 1:
                final_reward = max(final_reward, 0.0)
            
            
            final_rewards.append(final_reward)
            format_scores.append(format_score)
            execution_scores.append(execution_score)
        
            # Set reward at the last token position
            reward_tensor[i, valid_response_length[i].item() - 1] = final_reward

        # Check for additional trajectory statistics if available
        if "turns_stats" in data.non_tensor_batch:
            num_turn = data.non_tensor_batch["turns_stats"]
            num_valid_action = data.non_tensor_batch["valid_action_stats"]
            is_active = data.non_tensor_batch["active_mask"]
            is_done = [not is_active[i] for i in range(len(is_active))]

        data_source = data.non_tensor_batch[self.reward_fn_key]
        
        if save_record:
            to_save_records = [
                {
                    "id": data[i].non_tensor_batch['extra_info'].get('id') if 'extra_info' in data[i].non_tensor_batch and data[i].non_tensor_batch['extra_info'] else None,
                    "data_source": data_source[i],
                    "prompt": self.tokenizer.decode(prompt_ids[i][-valid_prompt_length[i].item():], skip_special_tokens=False),
                    "prompt_ntokens": valid_prompt_length[i].item(),
                    "response": self.tokenizer.decode(response_ids[i][:valid_response_length[i].item()], skip_special_tokens=False),
                    "response_ntokens": valid_response_length[i].item(),
                    "final_reward": final_rewards[i],
                    "format_score": format_scores[i],
                    "execution_score": execution_scores[i],
                    "tool_interact_info": data[i].non_tensor_batch.get('tool_interact_info', None),
                    'extra_info': data[i].non_tensor_batch.get('extra_info', None),
                    "step": self.step,  # Add step info for easier tracking
                    "timestamp": time.time(),  # Add timestamp for debugging
                }
                for i in range(len(data))
            ]
            if "turns_stats" in data.non_tensor_batch:
                for i, record in enumerate(to_save_records):
                    to_save_records[i]['num_turn'] = num_turn[i]
                    to_save_records[i]['num_valid_action'] = num_valid_action[i]
                    to_save_records[i]['is_done'] = is_done[i]
            
            # Async save to JSONL file
            if self.num_examine == 1:
                temp_file = self.record_dir / f"sqlcoder-step-val-{self.step}.jsonl"
            else:
                temp_file = self.record_dir / f"sqlcoder-step-{self.step}.jsonl"
            
            # Save asynchronously without blocking
            future = self.async_writer.save_async(temp_file, to_save_records)
            print(f"===> Queued {len(to_save_records)} records for async save to {temp_file}")
            
            self.step += 1
            
        if self.num_examine == 1:
            # for validation, empty the reward_extra_info, because there are None items and cannot be mean
            reward_extra_info = defaultdict(list)
        else:
            reward_extra_info = {
                "format_scores": format_scores,
                "execution_scores": execution_scores,
                "final_rewards": final_rewards,
                "positive_rewards": positive_rewards,
            }
            
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
    
    def shutdown(self):
        """Gracefully shutdown the reward manager and async writer."""
        if hasattr(self, 'async_writer'):
            self.async_writer.shutdown()
    
    def __del__(self):
        """Ensure cleanup when object is destroyed."""
        try:
            self.shutdown()
        except:
            pass  # Ignore cleanup errors during destruction