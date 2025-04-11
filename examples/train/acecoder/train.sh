set -x
dataset_name=AceCoderV2-mini-processed-with-execution-prompt
train_data=data/acecoder/$dataset_name/train.parquet
val_data=data/acecoder/$dataset_name/test.parquet
model_name=Qwen/Qwen2.5-7B-Instruct
rl_alg=grpo # gae(ppo) or grpo, if grpo, then better set n>1 otherwise the group norm can not be effective
n_gpus_per_node=4
n_nodes=1
n=8
batch_size=512
ppo_mini_batch_size=64
max_prompt_length=3072
max_response_length=2048
max_obs_length=512
temperature=0.9
strategy="fsdp_agent" # remove _agent for normal verl behavior
valid_actions="[python]" # "[answer,python]" are two valid actions, they are used to determine the stop token of each action, which are </answer> and </python> respectively

model_pretty_name=$(echo $model_name | tr '/' '_' | tr '[:upper:]' '[:lower:]')
run_name="${model_pretty_name}-${rl_alg}-n${n}-b${batch_size}-t${temperature}"
export VERL_RUN_ID=$run_name

host=0.0.0.0
port=$(shuf -i 30000-31000 -n 1)
tool_server_url=http://$host:$port/get_observation
python -m verl_tool.servers.serve --host $host --port $port --tool_type "python_code" &
server_pid=$!
echo "Server (pid=$server_pid) started at $tool_server_url"

# actor_rollout_ref.rollout.enforce_eager=False \
# actor_rollout_ref.rollout.free_cache_engine=False \

# export VLLM_USE_V1=1
PYTHONUNBUFFERED=1 python3 -m verl_tool.trainer.main_ppo \
    algorithm.adv_estimator=$rl_alg \
    data.train_files=$train_data \
    data.val_files=$val_data \
    data.train_batch_size=$batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.truncation='right' \
    reward_model.reward_manager=acecoder \
    +reward_model.tool_use_min_calls=1 \
    +reward_model.tool_use_max_calls=10 \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.strategy=$strategy \
    +actor_rollout_ref.agent.tool_server_url=$tool_server_url \
    +actor_rollout_ref.agent.max_prompt_length=$max_prompt_length \
    +actor_rollout_ref.agent.max_response_length=$max_response_length \
    +actor_rollout_ref.agent.max_start_length=$max_prompt_length \
    +actor_rollout_ref.agent.max_obs_length=$max_obs_length \
    +actor_rollout_ref.agent.max_turns=5 \
    +actor_rollout_ref.agent.num_gpus=$n_gpus_per_node \
    +actor_rollout_ref.agent.valid_actions=$valid_actions \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    critic.optim.lr=1e-5 \
    critic.strategy=$strategy \
    critic.model.path=$model_name \
    critic.ppo_micro_batch_size_per_gpu=1 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.remove_previous_ckpt_in_save=True \
    trainer.logger=['console','wandb'] \
    trainer.project_name='acecoder' \
    trainer.experiment_name=$run_name \
    trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$n_nodes \
    trainer.save_freq=40 \
    trainer.test_freq=10 \
    trainer.total_epochs=5


pkill -P -9 $server_pid
kill -9 $kill $server_pid
