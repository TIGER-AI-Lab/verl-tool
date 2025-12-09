# MT-GRPO: Multi-Turn Group Relative Policy Optimization

This is the recipe that implementation of RLVR algorithm proposed in  
**Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Reward Design**  
([arXiv:2505.11821](https://arxiv.org/abs/2505.11821

MT-GRPO is an algorithm designed for multi-turn interactive reinforcement learning, enabling **turn-level credit assignment**.

## Core Principles

### GRPO vs MT-GRPO

**Standard GRPO**:
- Only provides a total reward at task completion
- Cannot distinguish which actions are more important

**MT-GRPO**:
- Computes rewards at each turn
- Can identify which specific actions (e.g., code execution) bring value
- Core formula: `advantage = outcome_adv + turn_coef × turn_adv × before_result_mask`

Where:
- `outcome_adv`: Advantage for final outcome (e.g., answer correctness)
- `turn_adv`: Advantage for intermediate steps (e.g., successful code execution)
- `before_result_mask`: Only reward tokens before the result

## Quick Start

### 1. Prepare Data

```bash
cd /path/to/verl-tool
python examples/data_preprocess/gsm8k_code.py
```

Data is saved to `data/gsm8k_code_updated/`

### 2. Evaluate Baseline

```bash
cd examples/train/mt-grpo

# Default: evaluate Qwen2.5-Coder-3B
./eval.sh

# Custom
MODEL="Qwen/Qwen2.5-Coder-0.5B-Instruct" GPUS="0,1" ./eval.sh
```

### 3. Training

```bash
# Default configuration (0.5B model, 2 GPUs)
./train.sh

# Custom configuration
MODEL="Qwen/Qwen2.5-Coder-0.5B-Instruct" \
GPUS="4,5" \
N_GPUS=2 \
BATCH_SIZE=16 \
EPOCHS=3 \
./train.sh
```

### 4. Monitoring

```bash
# Real-time logs
tail -f ../../logs/train_mt_grpo.log | grep "step:"

# Key metrics
# - train/turn_reward: should increase from 0
# - train/num_valid_actions: number of code executions
# - train/accuracy: accuracy
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| MODEL | Qwen/Qwen2.5-Coder-0.5B-Instruct | Model |
| GPUS | 4,5 | GPU indices |
| N_GPUS | 2 | Number of GPUs |
| BATCH_SIZE | 16 | Batch size |
| EPOCHS | 3 | Number of epochs |
| LR | 1e-6 | Learning rate |

### MT-GRPO Parameters

- `turn_advantage_coef=1.0`: Weight for turn advantage
- `max_turns=2`: Maximum interaction turns
- `reward_manager=gsm8k_code`: Turn reward calculator

## Output

- Checkpoints: `checkpoints/mt_grpo_train/`
- Logs: `logs/train_mt_grpo.log`

## Implementation

Core code:
- `verl_tool/recipe/mt_grpo/mt_grpo_core_algos.py`: Advantage computation
- `verl_tool/workers/reward_manager/gsm8k_code.py`: Turn reward computation
- `verl_tool/recipe/mt_grpo/main_mt_grpo.py`: Training main loop
