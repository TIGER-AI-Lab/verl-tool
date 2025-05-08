"""
Metrics related to the Agent PPO trainer. Change it to add more metrics.
"""

import verl.trainer.ppo.metric_utils
verl_computer_data_metrics = verl.trainer.ppo.metric_utils.compute_data_metrics

import torch
from typing import Any, Dict, List
import numpy as np
from verl import DataProto

def _compute_response_info(batch: DataProto) -> Dict[str, Any]:
    response_length = batch.batch['responses'].shape[-1]

    # use observation-masked attention masks
    if 'info_mask' in batch.batch.keys():
        prompt_mask = batch.batch['info_mask'][:, :-response_length]
        response_mask = batch.batch['info_mask'][:, -response_length:]
    else:
        prompt_mask = batch.batch['attention_mask'][:, :-response_length]
        response_mask = batch.batch['attention_mask'][:, -response_length:]
    
    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )
    
def agent_compute_data_metrics(batch: DataProto, use_critic: bool = True) -> Dict[str, Any]:
    metrics = verl_computer_data_metrics(batch, use_critic)
    
    max_response_length = batch.batch['responses'].shape[-1]

    response_info = _compute_response_info(batch)
    response_length = response_info['response_length']
    
     # metrics for actions
    if 'turns_stats' in batch.non_tensor_batch:
        metrics['env/number_of_actions/mean'] = float(np.array(batch.non_tensor_batch['turns_stats'], dtype=np.int16).mean())
        metrics['env/number_of_actions/max'] = float(np.array(batch.non_tensor_batch['turns_stats'], dtype=np.int16).max())
        metrics['env/number_of_actions/min'] = float(np.array(batch.non_tensor_batch['turns_stats'], dtype=np.int16).min())
    if 'active_mask' in batch.non_tensor_batch:
        metrics['env/finish_ratio'] = 1 - float(np.array(batch.non_tensor_batch['active_mask'], dtype=np.int16).mean())
    if 'valid_action_stats' in batch.non_tensor_batch:
        metrics['env/number_of_valid_action'] = float(np.array(batch.non_tensor_batch['valid_action_stats'], dtype=np.int16).mean())
        metrics['env/ratio_of_valid_action'] = float((np.array(batch.non_tensor_batch['valid_action_stats'], dtype=np.int16) / np.array(batch.non_tensor_batch['turns_stats'], dtype=np.int16)).mean())
    
    metrics.update({
        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
    })
    
    return metrics

def compute_timing_metrics(batch: DataProto, timing_raw: Dict[str, float]) -> Dict[str, Any]:
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }