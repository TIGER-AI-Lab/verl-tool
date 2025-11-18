import torch
import numpy as np
from typing import Any
import torch
from verl.protocol import DataProto
from verl.trainer.ppo.metric_utils import compute_data_metrics as verl_compute_data_metrics
def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> dict[str, Any]:
    """
    Computes various metrics from a batch of data for PPO training.

    This function calculates metrics related to scores, rewards, advantages, returns, values,
    and sequence lengths from a batch of data. It provides statistical information (mean, max, min)
    for each metric category.

    Args:
        batch: A DataProto object containing batch data with token-level scores, rewards, advantages, etc.
        use_critic: Whether to include critic-specific metrics. Defaults to True.

    Returns:
        A dictionary of metrics including:
            - critic/score/mean, max, min: Statistics about sequence scores
            - critic/rewards/mean, max, min: Statistics about sequence rewards
            - critic/advantages/mean, max, min: Statistics about advantages
            - critic/returns/mean, max, min: Statistics about returns
            - critic/values/mean, max, min: Statistics about critic values (if use_critic=True)
            - critic/vf_explained_var: Explained variance of the value function (if use_critic=True)
            - response_length/mean, max, min, clip_ratio: Statistics about response lengths
            - prompt_length/mean, max, min, clip_ratio: Statistics about prompt lengths
            - num_turns/mean, max, min: Statistics about the number of multi-turn conversations
    """
    result = verl_compute_data_metrics(batch, use_critic)
    
    verl_tool_metrics = batch.non_tensor_batch.get("verl_tool_metrics", [])
    all_keys = []
    for x in verl_tool_metrics:
        all_keys.extend(list(x.keys()))
    all_keys = set(all_keys)
    for key in all_keys:
        values = np.array([float(m[key]) for m in verl_tool_metrics if key in m])
        result[f"verl_tool/{key}/mean"] = values.mean()
        result[f"verl_tool/{key}/max"] = values.max()
        result[f"verl_tool/{key}/min"] = values.min()
    return result