# Verl-Tool
An unified and easy-to-extend tool-agent training framework based on verl.

## Installation
```bash
pip install uv # if not installed
uv sync
git submodule update --init --recursive
source .venv/bin/activate
uv pip install -e verl[vllm]
uv pip install vllm==0.8.1 # aligh with torl 
uv pip install flash-attn --no-build-isolation
```

## Training
```bash
python examples/data_preprocess/gsm8k.py # Preprocess the data
bash examples/train/train_gsm8k.sh
```

## Test Tool Servers

```bash
python -m verl_tool.servers.serve # Start the server
python -m verl_tool.servers.tests.test_base # Run the tests
```

## ToDos
- [ ] Add python servers and example training scripts
- [ ] Add browser servers and example training scripts
- [ ] Add VLM servers and example training scripts
- [ ] Custom truncation logic for large observations
- [ ] Add wandb logging statistics
- [ ] Add saving logic for every step's sampling results and observations for inspection

## Contribution
### Contribution to tool libraries
Go to the [./verl_tool/servers/tools](./verl_tool/servers/tools) directory. Each tool has a name (e.g. `base`, `python_code`), the name of tool is exactly the name of the python file that you should create in the directory. See [./verl_tool/servers/tools/python_code.py](./verl_tool/servers/tools/python_code.py) for an example.

### New reward manager
Go to the [./verl_tool/agent_workers/reward_manager](./verl_tool/agent_workers/reward_manager) directory. Adding your new reward manager. import it in the `__init__.py` file and in `verl_tool/trainer/main_ppo.py` file add your reward manager importing logic.