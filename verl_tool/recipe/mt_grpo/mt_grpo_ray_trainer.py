"""
MT-GRPO Ray Trainer

Extends AgentRayPPOTrainer with turn-level credit assignment.
Following verl/recipe pattern (like PRIME).
"""

from verl import DataProto
from verl.trainer.ppo.ray_trainer import compute_response_mask as _compute_response_mask

# Import base trainer - we'll use AgentRayPPOTrainer but replace compute_advantage
from verl_tool.trainer.ppo.ray_trainer import AgentRayPPOTrainer
from . import mt_grpo_core_algos


def compute_response_mask(data: DataProto):
    """Compute response mask, delegating to verl's implementation."""
    return _compute_response_mask(data)


def compute_advantage(data: DataProto, adv_estimator, config=None, **kwargs):
    """
    Compute advantage using MT-GRPO algorithm or fallback to original.
    
    This function replaces verl's compute_advantage for MT-GRPO training.
    It checks if adv_estimator is "mt_grpo" (string or enum), and if so,
    uses MT-GRPO logic. Otherwise, falls back to verl's original compute_advantage.
    
    Args:
        data: DataProto with batch and non_tensor_batch
        adv_estimator: Advantage estimator name (string "mt_grpo" or AdvantageEstimator enum)
        config: Training config
        **kwargs: Additional arguments (gamma, lam, etc.)
        
    Returns:
        DataProto with computed advantages
    """
    # Check if adv_estimator is "mt_grpo" (handle both string and enum)
    is_mt_grpo = (
        adv_estimator == "mt_grpo" or 
        str(adv_estimator) == "mt_grpo" or
        (hasattr(adv_estimator, 'value') and adv_estimator.value == "mt_grpo")
    )
    
    if not is_mt_grpo:
        # Fallback to original compute_advantage
        from verl.trainer.ppo.ray_trainer import compute_advantage as _original_compute_advantage
        return _original_compute_advantage(data, adv_estimator, config=config, **kwargs)
    
    # MT-GRPO logic
    # Ensure response_mask exists
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    
    response_mask = data.batch["response_mask"]
    
    # Get turn_advantage_coef from config
    turn_coef = 1.0
    if config is not None:
        if hasattr(config, 'algorithm') and hasattr(config.algorithm, 'turn_advantage_coef'):
            turn_coef = config.algorithm.turn_advantage_coef
        elif hasattr(config, 'turn_advantage_coef'):
            turn_coef = config.turn_advantage_coef
        elif hasattr(config, 'get'):
            turn_coef = config.get('turn_advantage_coef', turn_coef)
            if turn_coef == 1.0 and hasattr(config, 'algorithm'):
                turn_coef = config.algorithm.get('turn_advantage_coef', turn_coef)
    
    # Compute MT-GRPO advantages
    advantages, returns = mt_grpo_core_algos.compute_mt_grpo_advantage_return(
        data=data,
        response_mask=response_mask,
        config=config,
        turn_advantage_coef=turn_coef,
    )
    
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    
    return data


# Export the trainer class (same as AgentRayPPOTrainer, compute_advantage is replaced at module level)
MTGRPORayTrainer = AgentRayPPOTrainer

