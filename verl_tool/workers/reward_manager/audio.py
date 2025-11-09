import os
import time
import json
import regex as re
from collections import defaultdict


import torch
from pathlib import Path
from verl import DataProto
import numpy as np

from .utils import replace_consecutive_tokens
from verl.workers.reward_manager import register


def normalize_solution(solution_str):
    # be careful with the escape characters!
    if r"\boxed{" in solution_str:
        # extract solution from \boxed{...}
        solution_str = solution_str.split(r"\boxed{")[-1].split("}")[0]
    return solution_str


def is_contained(target, source):
    if target in source:
        return True
    return False


def replace_tokens(text: str, tokens: list[str]):
    """replace consecutive tokens with <token>*n"""
    for token in tokens:
        text = replace_consecutive_tokens(text, token)
    return text


def audio_compute_score(solution_str, ground_truth):
    if isinstance(ground_truth, list):
        return max([audio_compute_score(solution_str, gt) for gt in ground_truth])

    solution_str = normalize_solution(solution_str)
    if not solution_str or not ground_truth:
        # either solution_str or ground_truth is invalid
        return 0.0

    verify_result = is_contained(
        solution_str.lower().strip(),
        ground_truth.lower().strip()
    )
    if verify_result:
        return 1.0
    else:
        return 0.0


@register("audio")
class AudioRewardManager:
    name = "audio" # reward manager identifier
    
    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key='data_source', **kwargs):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = audio_compute_score
        self.reward_fn_key = reward_fn_key
        self.step = None

        # ----------
        # curiosity penalty
        self.add_curiousity_penalty = True
        self.group_tool_call_rate_lower_bound = 0.5
        self.alpha = 0.5
        # ----------
        # execution penalty
        self.add_tool_exec_penalty = True
        # ----------

        if "record_dir" in kwargs:
            self.record_dir = Path(kwargs['record_dir'])
            self.record_dir.mkdir(parents=True, exist_ok=True)

    def add_additional_penalties(self, response: str, data_i, scores_i: dict, group_info:dict):
        """
            compute additional penalities:
            - curiosity_penalty: incentivize the model to explore tool-using more often
            - execution penalty: penalize the model for incorrect tool-using
        """

        if "turns_stats" in data_i.non_tensor_batch:
            num_turn = data_i.non_tensor_batch["turns_stats"]
            num_valid_action = data_i.non_tensor_batch["valid_action_stats"]

            if self.add_curiousity_penalty:
                # penalize the model if group_tool_call_rate is 
                # less than group_tool_call_rate_lower_bound
                penalty = max(0, self.group_tool_call_rate_lower_bound - group_info['group_tool_call_rate'])
                penalty *= self.alpha
                scores_i['score'] -= penalty
                scores_i['curiousity_penalty'] = penalty

            if self.add_tool_exec_penalty:
                keywords = ["error", "failed"]
                if any(keyword in response.lower() for keyword in keywords):
                    scores_i['score'] -= 0.1
                    scores_i['exec_error'] = 1
                else:
                    scores_i['exec_error'] = 0

        return scores_i 

    def get_group_info(self, data: DataProto):
        """
            get group information:
            - num_turns
            - num_valid_actions
            - group_tool_call_rate
            - tool_call_total
        """
        group_info = {}
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            num_turn = data_item.non_tensor_batch["turns_stats"]
            num_valid_action = data_item.non_tensor_batch["valid_action_stats"]
            if "turns_stats" in data_item.non_tensor_batch:
                uid = data_item.non_tensor_batch.get('uid', i)
                if uid not in group_info:
                    group_info[uid] = {}
                if 'num_turns' not in group_info[uid]:
                    group_info[uid]['num_turns'] = []
                if 'num_valid_actions' not in group_info[uid]:
                    group_info[uid]['num_valid_actions'] = []
                group_info[uid]['num_turns'].append(num_turn)
                group_info[uid]['num_valid_actions'].append(num_valid_action)
        for uid, info in group_info.items():
            info['num_turns'] = np.array(info['num_turns'])
            info['num_valid_actions'] = np.array(info['num_valid_actions'])
            info['group_tool_call_rate'] = np.mean([1 if num_valid_action > 0 else 0 for num_valid_action in info['num_valid_actions']])
            info['tool_call_total'] = info['num_valid_actions'].sum()
        return group_info

    def __call__(self, data: DataProto, return_dict=False):
        save_record = data.meta_info.get('save_record', True)

        if not hasattr(self, 'record_dir'):
            # intialize record_dir
            if hasattr(self, 'run_id'):
                self.record_dir = Path(__file__).parent.parent.parent.parent / "verl_step_records" / self.run_id
                self.record_dir.mkdir(parents=True, exist_ok=True)
            else:
                self.record_dir = Path(__file__).parent.parent.parent.parent / "verl_step_records" / f"torl-{time.strftime('%Y-%m-%d-%H-%M-%S')}"
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

        group_info = self.get_group_info(data)
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
            if "loss_mask" in data_item.batch:
                loss_mask = data_item.batch['loss_mask']
                valid_response_ids_with_loss_mask = torch.where(loss_mask[prompt_length:prompt_length + valid_response_length] == 1, valid_response_ids, self.tokenizer.pad_token_id)
            else:
                valid_response_ids_with_loss_mask = valid_response_ids

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            torl_score = self.compute_score(
                solution_str=response_str,
                ground_truth=ground_truth,
            ) # 1.0 or 0.0
            score['accuracy'] = 1 if torl_score > 0 else 0
            score['score'] = torl_score

            # add additional penalty
            score = self.add_additional_penalties(response_str, data_item, score, group_info.get(data_item.non_tensor_batch.get('uid', i), {}))      

            if score['accuracy'] > 0:
                reward_extra_info['correct_response_length'].append(valid_response_length)
            else:
                reward_extra_info['wrong_response_length'].append(valid_response_length)

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
            tool_interact_info_i = data_item.non_tensor_batch.get('tool_interact_info', None)
            if tool_interact_info_i is not None:
                # crop the image
                for tool_interact in tool_interact_info_i:
                    if "image" in tool_interact:
                        if isinstance(tool_interact['image'], list):
                            tool_interact['image'] = [x[:50] for x in tool_interact['image']]  # crop the image to first 50 characters
                        elif isinstance(tool_interact['image'], str):
                            tool_interact['image'] = tool_interact['image'][:50] # for debug
            
            to_save_prompt = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
            to_save_resposne = self.tokenizer.decode(response_ids[:valid_response_length], skip_special_tokens=False)
            to_save_prompt = replace_tokens(to_save_prompt, tokens=["<|AUDIO|>", "<audio_pad>"])
            to_save_response = replace_tokens(to_save_resposne, tokens=["<|AUDIO|>", "<audio_pad>"])
            if 'responses_with_loss_mask' in data_item.batch:
                to_save_response_with_loss_mask = self.tokenizer.decode(valid_response_ids_with_loss_mask, skip_special_tokens=False)
                to_save_response_with_loss_mask = replace_tokens(to_save_response_with_loss_mask, tokens=[self.tokenizer.pad_token])
            to_save_records.append({
                'id': data_item.non_tensor_batch.get('extra_info', {}).get('id'),
                'data_source': data_source,
                "prompt": to_save_prompt,
                "response": to_save_response,
                'response_with_loss_mask': to_save_response_with_loss_mask if 'responses_with_loss_mask' in data_item.batch else None,
                'ground_truth': ground_truth,
                'score': score,
                'reward': reward,
                'tool_interact_info': tool_interact_info_i,
                'extra_info': data_item.non_tensor_batch.get('extra_info', None),
            })
            if "turns_stats" in data_item.non_tensor_batch:
                to_save_records[i]['num_turn'] = data[i].non_tensor_batch["turns_stats"]
                to_save_records[i]['num_valid_action'] = data[i].non_tensor_batch["valid_action_stats"]
                to_save_records[i]['is_done'] = not data[i].non_tensor_batch["active_mask"]
        if save_record:
            # Save the records to a file
            if self.num_examine == 1:
                temp_file = self.record_dir / f"{self.name}-step-val-{self.step}.json"
            else:
                temp_file = self.record_dir / f"{self.name}-step-{self.step}.json"
            self.step += 1
            if temp_file.exists():
                with open(temp_file, "r") as f:
                    existing_records = json.load(f)
                to_save_records = existing_records + to_save_records
            with open(temp_file, "w") as f:
                json.dump(to_save_records, f, indent=4)
            print(f"Saved records to {temp_file}")
        
        correct_response_length_mean = np.mean(reward_extra_info['correct_response_length']) if reward_extra_info['correct_response_length'] else 0.0
        wrong_response_length_mean = np.mean(reward_extra_info['wrong_response_length']) if reward_extra_info['wrong_response_length'] else 0.0
        reward_extra_info['correct_response_length'] = [correct_response_length_mean] * len(reward_tensor)
        reward_extra_info['wrong_response_length'] = [wrong_response_length_mean] * len(reward_tensor)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
