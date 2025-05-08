# Verl-Tool

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/imgs/logo.png">
    <img alt="VerlTool" src="assets/imgs/logo.png" width=20%>
  </picture>
</p>

<h3 align="center">
VerlTool: An unified and easy-to-extend tool-agent training framework based on verl.
</h3>

---

<!-- <p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p> -->

## Installation
```bash
git submodule update --init --recursive
pip install uv # if not installed
uv sync
source .venv/bin/activate
uv pip install -e verl
uv pip install vllm==0.8.4
uv pip install flash-attn --no-build-isolation
uv pip install -e .[acecoder,torl]
uv pip install dill==0.4.0
uv pip install fsspec==2025.3.2
uv pip install protobuf==5.29.4
```

## Features
1. Fully separated the tool server and the training logic. By passing a list of `action_stop_tokens` to the training script, each action ending with any of the tokens will be passed to the tool server and further processed by identifying the tool type based on the custom pasing logic in each tool (`parse_action` function). 
2. Fast Tool Server: We use [ray serve](https://docs.ray.io/en/latest/serve/index.html) to serve the tool servers, which is fully asynchronous and compatible with the verl training.
3. Multiple tool types, see [./verl_tool/servers/tools](./verl_tool/servers/tools) for all the avaiable tools. Each python file is a tool type we support.
4. Natural multi-turn RL with tool calling support by specifying the `max_turns` in the script.
5. We make [verl](https://github.com/volcengine/verl) as a submodule of this repo, and only add additional logics by inheriting the `ActorRolloutRefWorker` and `RayPPOTrainer`, making it easy to extend and maintain.

## Training

We will take acecoder as an example. First prepare the data for training. You can use the provided script to preprocess the data.
```bash
python verl-tool/examples/data_preprocess/acecoder.py # preprocess the data and save
```

### Single Node Training

```bash
ray start --head --dashboard-host=0.0.0.0 # start ray head node
bash examples/train/acecoder/train.sh # train the model
```
Training tips:
1. For low VRAM GPUs, we recommend using set `do_offload=True`, `enforce_eager=True`, `tensor_parallel_size=1`, `use_dynamic_bsz=False`, and low `ppo_micro_batch_size_per_gpu`
2. If you encounter vllm generation stuck, try lower the `workers_per_tool` in the script, and use lower `gpu_memory_utilization` in the script.
3. For large VRAM GPUs, we recommend using set `do_offload=False`, `use_dynamic_bsz=True` for faster training.

### Multi Node Training
1. Head Node
```bash
ray start --head --dashboard-host=0.0.0.0 # start ray head node
bash examples/train/acecoder/train.sh # train the model
```
2. Worker Node
```bash
ray start --address='head_node_ip:6379' --block # start ray worker node
# no need to run the training script on worker node
```

## Evaluation
We do all the evaluation by serving the model in an openai compatible api way. After launching the eval server, you can evaluate the model just like calling the openai api. See [eval_service/README.md](./eval_service/README.md) for more details. 

## Test Tool Servers

```bash
# Start the ray server for the tool
python -m verl_tool.servers.serve --host 0.0.0.0 --port 5000 --tool_type "firejail_python_code" &
# Run the tests
python -m verl_tool.servers.tests.test_python_code_tool python --url=http://localhost:5000/get_observation
```

## ToDos
- [ ] Add VLM servers and example training scripts

## Contribution
### Contribution to tool libraries
Go to the [./verl_tool/servers/tools](./verl_tool/servers/tools) directory. Each tool has a name (e.g. `base`, `python_code`), the name of tool is exactly the name of the python file that you should create in the directory. See [./verl_tool/servers/tools/python_code.py](./verl_tool/servers/tools/python_code.py) for an example.

### New reward manager
Go to the [./verl_tool/agent_workers/reward_manager](./verl_tool/agent_workers/reward_manager) directory. Adding your new reward manager. import it in the `__init__.py` file and in `verl_tool/trainer/main_ppo.py` file add your reward manager importing logic.