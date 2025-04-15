export CUDA_VISIBLE_DEVICES=6
export WANDB_API_KEY="31800be459915fc29628b8c03920c3b526d64109"
export HF_TOKEN="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"

set -x
rl_alg=grpo # gae(ppo) or grpo, if grpo, then better set n>1 otherwise the group norm can not be effective
train_data=data/math_hard/train.parquet
val_data=data/math_hard/test.parquet
model_name=/map-vepfs/models/Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2
batch_size=128
max_prompt_length=1024
max_response_length=3072
reward_manager=torl
lr=1e-6
ppo_mini_batch_size=$batch_size
strategy="fsdp_agent" # remove _agent for normal verl behavior
kl_loss_coef=0.0
kl_loss_type=low_var_kl
kl_coef=0
entropy_coeff=0

host=0.0.0.0
port=30268
tool_server_url=http://$host:$port/get_observation
# python -m verl_tool.servers.serve --host $host --port $port --tool_type "firejail_python_code" --workers_per_tool 32 &
server_pid=$!
echo "Server (pid=$server_pid) started at $tool_server_url"
max_obs_length=512
max_turns=1
n_gpus_per_node=1
n_nodes=1
action_stop_tokens="\`\`\`output"
temperature=1.0
top_p=1.0
n_samples_per_prompts=16

model_pretty_name=$(echo $model_name | tr '/' '_' | tr '[:upper:]' '[:lower:]')
run_name="${reward_manager}-${strategy}-${model_pretty_name}-${rl_alg}-n${n_samples_per_prompts}-b${batch_size}-t${temperature}-lr${lr}-$(date +%Y%m%d-%H%M%S)"
export VERL_RUN_ID=$run_name

# temp file for action tokens as verl cannot pass special strs as params
action_stop_tokens_file=$(mktemp)
echo "$action_stop_tokens" > $action_stop_tokens_file
echo "action_stop_tokens_file=$action_stop_tokens_file"

# export VLLM_USE_V1=1
PYTHONUNBUFFERED=1 python3 -m verl_tool.trainer.main_ppo \
    algorithm.adv_estimator=$rl_alg \
    data.train_files=$train_data \
    data.val_files=$val_data \
    data.train_batch_size=$batch_size \
    data.val_batch_size=2048 \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    reward_model.reward_manager=$reward_manager \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.strategy=$strategy \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    +actor_rollout_ref.agent.tool_server_url=$tool_server_url \
    +actor_rollout_ref.agent.max_prompt_length=$max_prompt_length \
    +actor_rollout_ref.agent.max_response_length=$max_response_length \
    +actor_rollout_ref.agent.max_start_length=$max_prompt_length \
    +actor_rollout_ref.agent.max_obs_length=$max_obs_length \
    +actor_rollout_ref.agent.max_turns=$max_turns \
    +actor_rollout_ref.agent.num_gpus=$n_gpus_per_node \
    +actor_rollout_ref.agent.action_stop_tokens=$action_stop_tokens_file \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=$top_p \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=$n_samples_per_prompts \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.max_num_seqs=1024 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.optim.lr=1e-5 \
    critic.strategy=$strategy \
    critic.model.path=$model_name \
    critic.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    trainer.logger=['console','wandb'] \
    trainer.project_name='torl' \
    trainer.experiment_name=$run_name \
    trainer.val_before_train=True \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$n_nodes \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    +trainer.remove_previous_ckpt_in_save=True \
    trainer.default_local_dir=verl_checkpoints/${run_name} \
    trainer.resume_mode=auto \
    trainer.resume_from_path=True \
    trainer.total_epochs=10

pkill -P -9 $server_pid
kill -9 $kill $server_pid