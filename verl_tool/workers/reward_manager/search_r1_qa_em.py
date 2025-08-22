"""
Search-R1 style QA Exact Match Reward Manager
"""
import torch
import random
import regex as re
import json
import time
import os
import time
import os
import numpy as np
from typing import Dict, Any
from verl import DataProto
from verl.workers.reward_manager.registry import register
from .reward_score import _default_compute_score
from collections import defaultdict
from pathlib import Path

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    import string
    
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score
    

def extract_solution(solution_str: str) -> str:
    """Extract the final answer from <answer> tags in the solution string."""
    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are 0  matches, return None
    if len(matches) < 1:
        return None

    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def count_answer_tags(text):
    opening_tags = text.count("<answer>")
    closing_tags = text.count("</answer>")
    return opening_tags, closing_tags


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """
    The scoring function for Search-R1 style exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth dict with 'target' field
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    open_count, close_count = count_answer_tags(solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        # ground truth
        print(f"Golden answers: {ground_truth.get('target', ground_truth)}")
        
        # extracted answer from model
        if answer is not None:
            print(f"Extracted answer is not None: {answer}")
        else:
            print("Extracted answer: None!")

        # raw output of the model
        print(f"Solution string: {solution_str}")

    if answer is None:
        return 0
    else:
        # Handle both dict and list ground truth formats
        target_answers = ground_truth.get('target', ground_truth) if isinstance(ground_truth, dict) else ground_truth
        
        if em_check(answer, target_answers):
            if open_count > 10 or close_count > 10:  # prevent output a lot of </answer>
                score = score / 4
                return score
            return score
        else:
            return format_score




@register("search_r1_qa_em")
class SearchR1QAEMRewardManager:
    """
    Reward Manager for Search-R1 style QA tasks with Exact Match scoring.
    """
    name = "search_r1_qa_em"
    
    # fix the error: in reward.py force passing "reward_fn_key" param
    def __init__(self, tokenizer=None, num_examine=1, compute_score=None, format_score=0.0, score=1.0, run_id=None, **kwargs) -> None:
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or _default_compute_score
        self.format_score = format_score
        self.score = score
        self.step = None

    def __call__(self, data: DataProto, return_dict=False):
        """Compute rewards for Search-R1 style responses."""
        save_record = data.meta_info.get('save_record', True)

        if not hasattr(self, 'record_dir'):
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

        # If there is rm score, we directly return rm score
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        scores = [{} for _ in range(len(data))]
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        already_print_data_sources = {}
        reward_extra_info = defaultdict(list)
        to_save_records = []

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # Decode the full sequence
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            # Get ground truth
            if 'reward_model' in data_item.non_tensor_batch:
                ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            else:
                # Fallback to direct ground truth or golden_answers
                ground_truth = data_item.non_tensor_batch.get('ground_truth', 
                              data_item.non_tensor_batch.get('golden_answers', []))

            # Compute score
            score = compute_score(
                solution_str=sequences_str, 
                ground_truth=ground_truth, 
                format_score=self.format_score,
                score=self.score
            )
            if score > 0:
                reward_extra_info['correct_response_length'].append(valid_response_length)
            else:
                reward_extra_info['wrong_response_length'].append(valid_response_length)

            # TODO: check if logic is correct
            # update this score to the scores
            scores[i] = {"score": score}

            reward_tensor[i, valid_response_length - 1] = score

            # Print examples for debugging
            data_source = data_item.non_tensor_batch.get('data_source', 'unknown')
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(f"=== Search-R1 QA EM Reward Debug ===")
                print(f"Data source: {data_source}")
                print(f"Score: {score}")
                print(f"Sequence: {sequences_str}")
                print("=" * 50)

        # Save the records
            to_save_records.append({
                'id': data_item.non_tensor_batch['extra_info']['id'] if 'id' in data_item.non_tensor_batch['extra_info'] else None,
                'data_source': data_source,
                "prompt": self.tokenizer.decode(prompt_ids[-valid_prompt_length:], skip_special_tokens=False),
                "response": self.tokenizer.decode(response_ids[:valid_response_length], skip_special_tokens=False),
                'ground_truth': ground_truth,
                'score': score,
                'tool_interact_info': data[i].non_tensor_batch.get('tool_interact_info', None),
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
                existing_records.extend(to_save_records)
                with open(temp_file, "w") as f:
                    json.dump(existing_records, f, indent=4)
            else:
                with open(temp_file, "w") as f:
                    json.dump(to_save_records, f, indent=4)
            print(f"Saved records to {temp_file}")

        for i, score in enumerate(scores):
            if isinstance(score, dict):
                
                # convert the length to a Python int
                length_i = data[i].batch['attention_mask'][data[i].batch['prompts'].shape[-1]:].sum().item()
                # subtract 1 because you want the last *valid* token
                reward_tensor[i, length_i - 1] = score['score']

                # reward_tensor[i, valid_response_length[i].item() - 1] = score['score']
                for k, v in score.items():
                    reward_extra_info[k].append(v)
            else:
                length_i = data[i].batch['attention_mask'][data[i].batch['prompts'].shape[-1]:].sum().item()
                reward_tensor[i, length_i - 1] = score

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
