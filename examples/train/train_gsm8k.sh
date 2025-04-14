aanswerset -x
train_data=~/data/gsm8k/train.parquet
val_data=~/data/gsm8k/test.parquet
model_name=Qwen/Qwen2.5-1.5B-Instruct
rl_alg=grpo # gae(ppo) or grpo, if grpo, then better set n>1 otherwise the group norm can not be effective
n_gpus_per_node=2
n_nodes=1
n=2
batch_size=16
ppo_mini_batch_size=8
max_prompt_length=2048
max_response_length=256
temperature=1.2
strategy="fsdp_agent" # remove _agent for normal verl behavior
valid_actions="[answer,python]" # "[answer,python]" are two valid actions, they are used to determine the stop token of each action, which are </answer> and </python> respectively

model_pretty_name=$(echo $model_name | tr '/' '_' | tr '[:upper:]' '[:lower:]')
run_name="${model_pretty_name}-${rl_alg}-n${n}-b${batch_size}-t${temperature}"

host=0.0.0.0
port=$(shuf -i 30000-31000 -n 1)
tool_server_url=http://$host:$port/get_observation
python -m verl_tool.servers.serve --host $host --port $port --tool_type "python_code" &
server_pid=$!
echo "Server (pid=$server_pid) started at $tool_server_url"

# export VLLM_USE_V1=1
PYTHONUNBUFFERED=1 python3 -m verl_tool.trainer.main_ppo \
    algorithm.adv_estimator=$rl_alg \
    data.train_files=$train_data \
    data.val_files=$val_data \
    data.train_batch_size=$batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.strategy=$strategy \
    +actor_rollout_ref.agent.tool_server_url=$tool_server_url \
    +actor_rollout_ref.agent.max_prompt_length=$max_prompt_length \
    +actor_rollout_ref.agent.max_response_length=$max_response_length \
    +actor_rollout_ref.agent.max_start_length=2048 \
    +actor_rollout_ref.agent.max_obs_length=512 \
    +actor_rollout_ref.agent.max_turns=10 \
    +actor_rollout_ref.agent.num_gpus=$n_gpus_per_node \
    +actor_rollout_ref.agent.valid_actions=$valid_actions \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    critic.optim.lr=1e-5 \
    critic.strategy=$strategy \
    critic.model.path=$model_name \
    critic.ppo_micro_batch_size_per_gpu=4 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='acecoder' \
    trainer.experiment_name=$run_name \
    trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$n_nodes \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=5 2>&1 | tee verl_demo.log


pkill -P -9 $server_pid
kill -9 $kill $server_pid
