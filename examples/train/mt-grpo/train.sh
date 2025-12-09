#!/bin/bash
# MT-GRPO Training Script
set -x

# Configuration
MODEL=${MODEL:-"Qwen/Qwen2.5-Coder-0.5B-Instruct"}
GPUS=${GPUS:-"4,5"}
N_GPUS=${N_GPUS:-2}
BATCH_SIZE=${BATCH_SIZE:-16}
EPOCHS=${EPOCHS:-3}
LR=${LR:-1e-6}

# Data paths
TRAIN_DATA=${TRAIN_DATA:-"./data/gsm8k_code_updated/train.parquet"}
VAL_DATA=${VAL_DATA:-"./data/gsm8k_code_updated/test.parquet"}

# Switch to project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.." || exit 1

# Activate virtual environment
source .venv/bin/activate
export PATH="$(pwd)/.venv/bin:$PATH"
export PYTHONPATH=$(pwd):$(pwd)/verl:$PYTHONPATH

# GPU settings
export CUDA_VISIBLE_DEVICES=$GPUS

# MT-GRPO configuration
rl_alg=mt_grpo
turn_advantage_coef=1.0

# Training configuration
n=4
ppo_mini_batch_size=$BATCH_SIZE
max_prompt_length=512
max_response_length=1024
max_obs_length=256
temperature=1.0
enable_agent=True
strategy="fsdp"
action_stop_tokens='</code>,</answer>'
max_turns=2
reward_manager=gsm8k_code
gpu_memory_utilization=0.6
do_offload=True
use_dynamic_bsz=True
additional_eos_token_ids=[151645]
mask_observations=True
save_freq=20
test_freq=10

run_name="mt_grpo_train_$(basename $MODEL | tr '/' '_')"
export VERL_RUN_ID=$run_name
export NCCL_DEBUG=WARN
export VLLM_USE_V1=1
rollout_mode='async'

# Action stop tokens temporary file
action_stop_tokens_file=$(mktemp)
echo -n "$action_stop_tokens" > $action_stop_tokens_file

# Start Tool Server
host=127.0.0.1
port=$(shuf -i 30000-31000 -n 1)
tool_server_url=http://$host:$port/get_observation

mkdir -p logs
python -m verl_tool.servers.serve --host $host --port $port --tool_type "python_code" --workers_per_tool 2 > logs/tool_server_train.log 2>&1 &
server_pid=$!
sleep 5
echo "Tool server started at $tool_server_url"

# Run MT-GRPO training
PYTHONUNBUFFERED=1 python3 -m verl_tool.recipe.mt_grpo.main_mt_grpo \
    algorithm.adv_estimator=$rl_alg \
    +algorithm.turn_advantage_coef=$turn_advantage_coef \
    algorithm.norm_adv_by_std_in_grpo=False \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=$BATCH_SIZE \
    data.val_batch_size=8 \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.truncation='right' \
    reward_model.reward_manager=$reward_manager \
    reward_model.launch_reward_fn_async=False \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.actor.strategy=$strategy \
    actor_rollout_ref.actor.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$do_offload \
    actor_rollout_ref.agent.enable_agent=$enable_agent \
    actor_rollout_ref.agent.tool_server_url=$tool_server_url \
    actor_rollout_ref.agent.max_prompt_length=$max_prompt_length \
    actor_rollout_ref.agent.max_response_length=$max_response_length \
    actor_rollout_ref.agent.max_start_length=$max_prompt_length \
    actor_rollout_ref.agent.max_obs_length=$max_obs_length \
    actor_rollout_ref.agent.max_turns=$max_turns \
    actor_rollout_ref.agent.additional_eos_token_ids=$additional_eos_token_ids \
    actor_rollout_ref.agent.mask_observations=$mask_observations \
    actor_rollout_ref.agent.action_stop_tokens=$action_stop_tokens_file \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.ref.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    trainer.logger=['console'] \
    trainer.project_name=mt_grpo_train \
    trainer.experiment_name=$run_name \
    trainer.val_before_train=True \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.total_epochs=$EPOCHS 2>&1 | tee logs/train_mt_grpo.log

# Cleanup
kill -9 $server_pid 2>/dev/null
rm -f $action_stop_tokens_file
echo "Training completed!"
