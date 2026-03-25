# test-rl-ppo-smoke (local, Mac) — tiny PPO smoke run

Goal: a **real PPO training smoke test** that runs locally on a Mac in <30 minutes:
- rollout/trajectory produced
- reward computed
- at least 1 update step executed
- checkpoint saved

## Critical note about Python versions
Your current environment is Python **3.14**, where `ray` is not installable today (no wheels).
The PPO trainer path requires Ray.

So for a *real* PPO smoke run locally, use Python **3.12** in a separate venv.

## 0) Create a Python 3.12 venv
From `VRTOOL-Framework/verl-tool/`:

```bash
/opt/homebrew/bin/python3.12 -m venv .venv312
source .venv312/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e .
python3 -m pip install torch tensordict ray hydra-core omegaconf transformers datasets pyarrow
```

## 1) Create a tiny dataset (10 math items)
```bash
python3 scripts/test-rl-ppo-smoke/make_tiny_math_parquet.py --out /tmp/tiny_math10.parquet
```

## 2) Run PPO with CPU-only Ray scheduling
This uses the repo entrypoint and Hydra config, but forces CPU scheduling:

```bash
export VERL_USE_GPU=0

python3 -m verl_tool.trainer.main_ppo \\
  trainer.device=cpu \\
  trainer.logger=[console] \\
  trainer.save_freq=1 \\
  trainer.test_freq=-1 \\
  trainer.total_training_steps=1 \\
  trainer.n_gpus_per_node=1 \\
  actor_rollout_ref.model.path=sshleifer/tiny-gpt2 \\
  actor_rollout_ref.model.lora_rank=8 \\
  data.train_files=/tmp/tiny_math10.parquet \\
  data.val_files=/tmp/tiny_math10.parquet \\
  data.train_max_samples=10 \\
  data.val_max_samples=10
```

Expected evidence:
- console logs for reward / loss / update
- checkpoint folder under `checkpoints/verl_examples/gsm8k/...` (default paths)

