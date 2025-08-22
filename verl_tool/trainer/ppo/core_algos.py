import torch
import numpy as np
import verl.trainer.ppo.core_algos
from verl.trainer.ppo.core_algos import (
    register_adv_est,
    register_policy_loss,
    AdvantageEstimator,
    AlgoConfig,
    agg_loss
)
import verl.utils.torch_functional as verl_F
from collections import defaultdict
from typing import Optional
from enum import Enum

class MyAdvantageEstimator(str, Enum):
    TDGRPO = "tdgrpo"
    GAPO = "gapo"

# Vectorized version (more efficient for larger batches)
def calculate_discounted_rewards_vectorized(mask, final_rewards, discount_factor):
    """
    Calculate discounted rewards for action sequences.
    Vectorized version for better performance on larger batches.
    
    Args:
        mask: Tensor of shape [batch_size, seq_length] with 1s for valid actions, 0s for padding
        final_rewards: Tensor of shape [batch_size] with final reward for each sequence
        discount_factor: Float, discount factor (lambda)
    
    Returns:
        Tensor of shape [batch_size, seq_length] with discounted rewards
    """
    batch_size, seq_length = mask.shape
    device = mask.device
    
    # Initialize output
    rewards = torch.zeros_like(mask, dtype=torch.float32, device=device)
    if isinstance(final_rewards, torch.Tensor) or isinstance(final_rewards, np.ndarray):
        final_rewards = final_rewards.tolist()
    
    # For each batch, process action groups
    for b in range(batch_size):
        seq_mask = mask[b]
        final_reward = final_rewards[b]
        
        # Find action group boundaries using the same logic as the first version
        # Add padding to handle edge cases
        padded_mask = torch.cat([torch.zeros(1, device=device), seq_mask, torch.zeros(1, device=device)])
        
        # Find start positions (0 -> 1 transitions)
        starts = torch.where((padded_mask[:-1] == 0) & (padded_mask[1:] == 1))[0]
        
        # Find end positions (1 -> 0 transitions) 
        ends = torch.where((padded_mask[:-1] == 1) & (padded_mask[1:] == 0))[0]
        
        # Calculate number of action groups
        num_groups = len(starts)
        
        if num_groups > 0:
            # Calculate discounted reward for each group (working backwards)
            current_reward = final_reward
            
            for i in range(num_groups - 1, -1, -1):  # Process groups in reverse order
                start_idx = starts[i]
                end_idx = ends[i]
                
                # Set reward for all positions in this action group
                rewards[b, start_idx:end_idx] = current_reward
                
                # Discount for next group (going backwards in time)
                current_reward *= discount_factor
    
    return rewards

# Vectorized version (more efficient for larger batches)
def get_num_actions(mask):
    """
    Calculate discounted rewards for action sequences.
    Vectorized version for better performance on larger batches.
    
    Args:
        mask: Tensor of shape [batch_size, seq_length] with 1s for valid actions, 0s for padding
        final_rewards: Tensor of shape [batch_size] with final reward for each sequence
        discount_factor: Float, discount factor (lambda)
    
    Returns:
        Tensor of shape [batch_size, seq_length] with discounted rewards
    """
    batch_size, seq_length = mask.shape
    device = mask.device
    total_num_actions = []
    # For each batch, process action groups
    for b in range(batch_size):
        seq_mask = mask[b]
        
        # Find action group boundaries using the same logic as the first version
        # Add padding to handle edge cases
        padded_mask = torch.cat([torch.zeros(1, device=device), seq_mask, torch.zeros(1, device=device)])
        
        # Find start positions (0 -> 1 transitions)
        starts = torch.where((padded_mask[:-1] == 0) & (padded_mask[1:] == 1))[0]
        
        # Find end positions (1 -> 0 transitions) 
        ends = torch.where((padded_mask[:-1] == 1) & (padded_mask[1:] == 0))[0]
        
        # Calculate number of action groups
        num_groups = len(starts)
        total_num_actions.append(num_groups)
    return torch.tensor(total_num_actions, device=device, dtype=torch.float32)

# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@register_adv_est(MyAdvantageEstimator.TDGRPO)
def compute_tdgrpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        index: `(np.ndarray)`
            index array for grouping
        epsilon: `(float)`
            small value to avoid division by zero
        norm_adv_by_std_in_grpo: `(bool)`
            whether to scale the GRPO advantage
        config: `(Optional[AlgoConfig])`
            algorithm configuration object

    Note:
        If norm_adv_by_std_in_grpo is True, the advantage is scaled by the std, as in the original GRPO.
        If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    assert hasattr(config, "tdgrpo_lambda") and config.tdgrpo_lambda is not None, \
        "tdgrpo_lambda must be set in the config for TDGRPO advantage estimation."

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]

        # response_mask is [bs, response_length]
        # each response is list [action_tokens, masked_observations, action_tokens, ..., padding]
        # in TD GRPO, we consider each turn as a action, since only the last action is associated with a reward, 
        # we propagate the reward to previous actions by temporal difference with factor lambda.

        scores = calculate_discounted_rewards_vectorized(
            response_mask,
            scores,
            config.tdgrpo_lambda
        )

    return scores, scores

# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@register_adv_est(MyAdvantageEstimator.GAPO)
def compute_gapo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        index: `(np.ndarray)`
            index array for grouping
        epsilon: `(float)`
            small value to avoid division by zero
        norm_adv_by_std_in_grpo: `(bool)`
            whether to scale the GRPO advantage
        config: `(Optional[AlgoConfig])`
            algorithm configuration object

    Note:
        If norm_adv_by_std_in_grpo is True, the advantage is scaled by the std, as in the original GRPO.
        If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    num_actions_per_sequence = get_num_actions(response_mask)
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            # id2score[index[i]].append(scores[i]* num_actions_per_sequence[i]) # treat each action as a separate seq
            id2score[index[i]].extend([scores[i]] * int(num_actions_per_sequence[i].item()))
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores

@register_policy_loss("gspo")
def compute_policy_loss_gspo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the clipped policy objective and related metrics for GSPO.

    See https://arxiv.org/pdf/2507.18071 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For GSPO, it is recommended to use "seq-mean-token-mean".
    """

    assert config is not None
    # assert isinstance(config, ActorConfig)
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio

    negative_approx_kl = log_prob - old_log_prob

    # compute sequence-level importance ratio:
    # si(θ) = (π_θ(yi|x)/π_θold(yi|x))^(1/|yi|) =
    # exp [(1/|y_i|) * Σ_t log(π_θ(y_i,t|x,y_i,<t)/π_θold(y_i,t|x,y_i,<t))]
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths

    # Combined ratio at token level:
    # s_i,t(θ) = sg[s_i(θ)] · π_θ(y_i,t|x, y_i,<t) / sg[π_θ(y_i,t|x, y_i,<t)]
    # In log space: log(s_i,t(θ)) = sg[log(s_i(θ))] + log_prob - sg[log_prob]
    log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
    log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)  # clamp for numerical stability

    # finaly exp() to remove log
    seq_importance_ratio = torch.exp(log_seq_importance_ratio)

    pg_losses1 = -advantages * seq_importance_ratio
    pg_losses2 = -advantages * torch.clamp(seq_importance_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # for GSPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean")

    # For compatibility, return zero for pg_clipfrac_lower (not used in standard GSPO)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower

verl.trainer.ppo.core_algos.POLICY_LOSS_REGISTRY["gspo"] = compute_policy_loss_gspo