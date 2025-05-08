# LLM Tool Calling Service

## Overview
This package provides a service that enables LLMs to call tools, temporarily focusing on Python code execution capabilities.

The service can be accessed via OpenAI's `client.chat.completions.create` API.
Server is managed by `app.py`, while the main tool-calling logic is implemented in `model_service.py`.

## Setup and Installation

### 1. Activate Service

Start the Python code execution tool server and API Service:

```bash
bash eval_service/scripts/start_api_service.sh &
```

> You can set your own params in `start_api_service.sh`

### 2. Test the API Service

**Please Replace with your local server address**

```bash
python eval_service/test/test_api.py
```

## Misc

How to convert megatron/vllm model ckpt to huggingface ckpt:
- `backend` is the training backend, either `fsdp` or `megatron`
- `hf_model_path` is the backbone model path that training started from. 
- `hf_upload_path` is the model path that training started from
- `local_dir` is the model path that training started from
- `target_dir` is the local hugging face model that you want to save to

```bash
backend=fsdp
checkpoint_path=checkpoints/acecoder/acecoder-fsdp_agent-qwen_qwen2.5-coder-1.5b-grpo-n16-b128-t1.0-lr1e-6-with-deepcoder-data/global_step_400/actor
hf_upload_path=VerlTool/acecoder-fsdp_agent-qwen_qwen2.5-coder-1.5b-grpo-n16-b128-t1.0-lr1e-6-with-wdcoder-400-step
python3 verl/scripts/model_merger.py --backend $backend --hf_model_path $checkpoint_path/huggingface --hf_upload_path "$hf_upload_path" --local_dir $checkpoint_path --target_dir $checkpoint_path/huggingface

# optional: also upload the step records to the model
step_records_dir=verl_step_records/acecoder-fsdp_agent-qwen_qwen2.5-coder-1.5b-grpo-n16-b128-t1.0-lr1e-6-with-deepcoder-data
zip -r $step_records_dir/step_records.zip $step_records_dir 
huggingface-cli upload --repo-type model $hf_upload_path $step_records_dir/step_records.zip 
```