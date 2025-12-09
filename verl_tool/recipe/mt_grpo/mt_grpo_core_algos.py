"""
MT-GRPO Core Algorithms

Turn-level credit assignment for multi-turn agent training.
Ported from Multi-Turn-RL-Agent project.
"""

import torch
import numpy as np
from collections import defaultdict
from typing import Optional, List, Dict, Any

import verl.utils.torch_functional as verl_F


def grpo_normalize(
    rewards: torch.Tensor,
    uid_index: np.ndarray,
    epsilon: float = 1e-4
) -> torch.Tensor:
    """
    GRPO group normalization: (reward - group_mean) / group_std
    
    Args:
        rewards: Trajectory-level rewards, shape (batch_size,)
        uid_index: UID for grouping (same prompt -> same UID)
        epsilon: Small value for numerical stability
        
    Returns:
        Normalized advantages, shape (batch_size,)
    """
    device = rewards.device
    advantages = torch.zeros_like(rewards)
    
    # Group by UID
    id2indices = defaultdict(list)
    for i, uid in enumerate(uid_index):
        id2indices[uid].append(i)
    
    for uid, indices in id2indices.items():
        if len(indices) == 0:
            continue
        indices_t = torch.tensor(indices, device=device, dtype=torch.long)
        group_rewards = rewards[indices_t]
        mean = group_rewards.mean()
        
        if len(indices) == 1:
            # Single element group: advantage = 0
            advantages[indices_t] = 0.0
        else:
            std = group_rewards.std()
            if std < epsilon:
                advantages[indices_t] = group_rewards - mean
            else:
                advantages[indices_t] = (group_rewards - mean) / (std + epsilon)
    
    return advantages


def compute_turn_reward_from_tool_info(tool_interact_info: List[Dict[str, Any]]) -> float:
    """
    Compute trajectory-level turn reward from tool interactions.
    
    Formula: 0.3 * (successful / total) + 0.05 * successful
    Same as Multi-Turn-RL-Agent's code_execution_reward_func.
    
    Args:
        tool_interact_info: List of tool interaction dicts with 'obs' and 'valid_action'
        
    Returns:
        Scalar turn reward for the trajectory
    """
    if not tool_interact_info:
        return 0.0
    
    total_valid = 0
    successful = 0
    
    for info in tool_interact_info:
        if info is None or not info.get('valid_action', False):
            continue
        total_valid += 1
        obs = info.get('obs', '')
        if isinstance(obs, str):
            is_error = 'Error:' in obs or 'Traceback' in obs or 'exception' in obs.lower()
            if not is_error and obs.strip():
                successful += 1
    
    if total_valid == 0:
        return 0.0
    return 0.3 * (successful / total_valid) + 0.05 * successful


def find_last_tool_response_position(response_mask: torch.Tensor) -> int:
    """
    Find position where last tool response ends (transition from 0 to 1).
    
    In verl-tool:
    - response_mask=1: model-generated tokens (actions)
    - response_mask=0: tool observations
    
    Args:
        response_mask: Binary mask, shape (seq_len,)
        
    Returns:
        Position of last obs->action transition, or 0 if none
    """
    mask_np = response_mask.cpu().numpy() if isinstance(response_mask, torch.Tensor) else np.array(response_mask)
    last_obs_end = 0
    in_obs = False
    
    for i in range(len(mask_np)):
        if mask_np[i] == 0:
            in_obs = True
        elif in_obs and mask_np[i] == 1:
            last_obs_end = i
            in_obs = False
    
    return last_obs_end


def compute_mt_grpo_advantage_return(
    data,
    response_mask: torch.Tensor,
    config,
    turn_advantage_coef: float = 1.0,
):
    """
    Compute MT-GRPO advantages with turn-level credit assignment.
    
    Core idea:
    - Tool calling phase: advantage = outcome_adv + turn_coef * turn_adv
    - Answer phase: advantage = outcome_adv
    
    Args:
        data: DataProto with batch and non_tensor_batch
        response_mask: Response mask tensor
        config: Training config
        turn_advantage_coef: Coefficient for turn advantage
        
    Returns:
        advantages: Token-level advantages, shape (batch_size, seq_len)
        returns: Same as advantages (for compatibility)
    """
    device = response_mask.device
    batch_size, seq_len = response_mask.shape
    
    # Get turn_advantage_coef from config if available
    turn_coef = turn_advantage_coef
    if config is not None:
        if hasattr(config, 'algorithm') and hasattr(config.algorithm, 'turn_advantage_coef'):
            turn_coef = config.algorithm.turn_advantage_coef
        elif hasattr(config, 'get'):
            turn_coef = config.get('turn_advantage_coef', turn_coef)
    
    # 1. Compute outcome rewards (sum of token-level rewards)
    token_level_rewards = data.batch["token_level_rewards"]
    outcome_rewards = token_level_rewards.sum(dim=-1)
    
    # 2. Get UID for grouping
    uid_index = data.non_tensor_batch.get("uid", np.arange(batch_size))
    
    # 3. Get or compute turn rewards
    turn_reward = data.non_tensor_batch.get("turn_reward", None)
    tool_interact_info = data.non_tensor_batch.get("tool_interact_info", None)
    
    if turn_reward is not None:
        if isinstance(turn_reward, np.ndarray):
            turn_rewards = torch.tensor(turn_reward, dtype=torch.float32, device=device)
        else:
            turn_rewards = torch.tensor(list(turn_reward), dtype=torch.float32, device=device)
    elif tool_interact_info is not None:
        turn_rewards_list = []
        for i in range(batch_size):
            info_i = tool_interact_info[i] if i < len(tool_interact_info) else None
            turn_r = compute_turn_reward_from_tool_info(info_i) if info_i else 0.0
            turn_rewards_list.append(turn_r)
        turn_rewards = torch.tensor(turn_rewards_list, dtype=torch.float32, device=device)
    else:
        # Fallback to standard GRPO (no turn-level credit assignment)
        combined_adv = grpo_normalize(outcome_rewards, uid_index)
        advantages = combined_adv.unsqueeze(-1).expand(-1, seq_len) * response_mask.float()
        advantages = verl_F.masked_whiten(advantages, response_mask)
        return advantages, advantages

    # 4. GRPO normalization for turn and outcome rewards
    turn_adv = grpo_normalize(turn_rewards, uid_index)
    outcome_adv = grpo_normalize(outcome_rewards, uid_index)
    combined_rewards = turn_rewards + outcome_rewards
    combined_adv = grpo_normalize(combined_rewards, uid_index)

    # 5. Find result segment for each sample
    result_segment_indices = []
    for i in range(batch_size):
        info_i = tool_interact_info[i] if tool_interact_info is not None and i < len(tool_interact_info) else None
        result_seg = find_result_segment(info_i)
        result_segment_indices.append(result_seg)

    # 6. Assign advantages with turn-level credit assignment
    advantages = torch.zeros(batch_size, seq_len, dtype=torch.float32, device=device)

    for i in range(batch_size):
        result_segment = result_segment_indices[i]
        mask_row = response_mask[i]

        outcome_adv_i = outcome_adv[i]
        turn_adv_i = turn_adv[i]
        combined_adv_i = combined_adv[i]

        if result_segment > 0:
            # Has <result> tag: apply turn-level credit assignment
            segment_boundaries = get_segment_boundaries(mask_row)

            if result_segment < len(segment_boundaries):
                split_point = segment_boundaries[result_segment]
                # before_result_mask: 1 for tokens before <result>, 0 after
                before_result_mask = (torch.arange(seq_len, device=device) < split_point).float() * mask_row.float()

                # Core formula:
                # Before <result>: advantage = outcome_adv + turn_coef * turn_adv
                # After <result>: advantage = outcome_adv only
                advantages[i] = outcome_adv_i + turn_coef * turn_adv_i * before_result_mask
            else:
                # Not enough segments, fallback to combined
                advantages[i] = combined_adv_i * mask_row.float()
        else:
            # No <result> tag: use combined advantage for all tokens
            advantages[i] = combined_adv_i * mask_row.float()

    # Apply mask and whiten
    advantages = advantages * response_mask.float()
    advantages = verl_F.masked_whiten(advantages, response_mask)

    return advantages, advantages


def get_segment_boundaries(mask_row: torch.Tensor) -> List[int]:
    """
    Get segment boundaries from mask transitions.

    A segment boundary occurs when mask value changes (0->1 or 1->0).
    This matches the original MT-GRPO implementation.

    Args:
        mask_row: 1D mask tensor

    Returns:
        List of boundary positions, including 0 and seq_len
    """
    seq_len = len(mask_row)
    boundaries = [0]

    mask_np = mask_row.cpu().numpy() if isinstance(mask_row, torch.Tensor) else mask_row

    for j in range(1, seq_len):
        if mask_np[j] != mask_np[j - 1]:
            boundaries.append(j)
    boundaries.append(seq_len)

    return boundaries


def find_result_segment(tool_interact_info: Optional[List[Dict[str, Any]]],
                        result_tag: str = '<result>') -> int:
    """
    Find the segment index containing the first <result> tag.

    Mapping: tool_interact_info[idx] corresponds to segment 2*idx+1
    (assistant messages are even segments, user/env messages are odd)

    Args:
        tool_interact_info: List of tool interaction info dicts
        result_tag: Tag to search for (default '<result>')

    Returns:
        Segment index, or -1 if not found
    """
    if tool_interact_info is None or len(tool_interact_info) == 0:
        return -1

    for idx, info in enumerate(tool_interact_info):
        if info is None:
            continue
        obs = info.get('obs', '')
        if isinstance(obs, str) and (result_tag in obs or '<output>' in obs):
            # segment index = 2 * idx + 1
            return 2 * idx + 1

    return -1

