
# OLD_MODEL
/home/user/luyi-workspace/model_weights/acecoder-fsdp_agent-xiaomimimo_mimo-7b-base-grpo-n16-b128-t1.0-lr1e-6-69k-2turn-sys4-110-step

new: 
/home/user/luyi-workspace/model_weights/acecoder-fsdp_agent-xiaomimimo_mimo-7b-base-grpo-n16-b128-t1.0-lr1e-6-69k-2turn-sys4-110-step

# bcb
- modify system prompt in `bigcodebench/gen/util/openai_request.py`
- for hard split:
export BIGCODEBENCH_OVERRIDE_PATH="../bcb-v0.1.4-hard.jsonl"
export BIGCODEBENCH_TIMEOUT_PER_TASK=30 # originally 240
split=complete # instruct or complete
subset=hard # hard or full
bigcodebench.evaluate \
  --model /home/user/luyi-workspace/model_weights/acecoder-fsdp_agent-xiaomimimo_mimo-7b-base-grpo-n16-b128-t1.0-lr1e-6-69k-2turn-sys4-110-step \
  --execution local \
  --split $split \
  --subset $subset \
  --backend openai \
  --bs 2048 \
  --base_url http://0.0.0.0:5000 \
  --temperature 0.0

- for full split:
export BIGCODEBENCH_OVERRIDE_PATH=
export BIGCODEBENCH_TIMEOUT_PER_TASK=30 # originally 240
split=complete # instruct or complete
subset=full # hard or full
bigcodebench.evaluate \
  --model /home/user/luyi-workspace/model_weights/acecoder-fsdp_agent-xiaomimimo_mimo-7b-base-grpo-n16-b128-t1.0-lr1e-6-69k-2turn-sys4-110-step \
  --execution local \
  --split $split \
  --subset $subset \
  --backend openai \
  --bs 2048 \
  --base_url http://0.0.0.0:5000 \
  --temperature 0.0

# evalplus
- modify system prompt in `evalplus/gen/util/openai_request.py`


export OPENAI_API_KEY="{KEY}" # https://platform.deepseek.com/api_keys
evalplus.evaluate --model "/home/user/luyi-workspace/model_weights/acecoder-fsdp_agent-xiaomimimo_mimo-7b-base-grpo-n16-b128-t1.0-lr1e-6-69k-2turn-sys4-110-step"              \
                  --dataset humaneval           \
                  --base-url http://0.0.0.0:5000  \
                  --backend openai --greedy \
                  --temperature 0.0

export OPENAI_API_KEY="{KEY}" # https://platform.deepseek.com/api_keys
evalplus.evaluate --model "/home/user/luyi-workspace/model_weights/acecoder-fsdp_agent-xiaomimimo_mimo-7b-base-grpo-n16-b128-t1.0-lr1e-6-69k-2turn-sys4-110-step"              \
                  --dataset mbpp           \
                  --base-url http://0.0.0.0:5000  \
                  --backend openai --greedy \
                  --temperature 0.0

# lcb
- modify system prompt in `lcb_runner/runner/oai_runner.py`

## lcb v4
export OPENAI_API_KEY="111"
export OPENAI_BASE_URL="http://0.0.0.0:5000" 
python -m lcb_runner.runner.main \
    --model "/home/user/luyi-workspace/model_weights/acecoder-fsdp_agent-xiaomimimo_mimo-7b-base-grpo-n16-b128-t1.0-lr1e-6-69k-2turn-sys4-110-step" \
    --scenario codegeneration \
    --evaluate \
    --start_date 2023-05-01 \
    --end_date 2024-09-01 \
    --multiprocess 64 \
    --temperature 0.0 \
    --top_p 0.95 \
    --num_process_evaluate 32 \
    --n 1 
    
## lcb v5
export OPENAI_API_KEY="111"
export OPENAI_BASE_URL="http://0.0.0.0:5000" 
python -m lcb_runner.runner.main \
    --model "/home/user/luyi-workspace/model_weights/acecoder-fsdp_agent-xiaomimimo_mimo-7b-base-grpo-n16-b128-t1.0-lr1e-6-69k-2turn-sys4-110-step" \
    --scenario codegeneration \
    --evaluate \
    --start_date 2024-08-01 \
    --end_date 2025-02-01 \
    --multiprocess 64 \
    --temperature 0.0 \
    --top_p 0.95 \
    --num_process_evaluate 32 \
    --n 1 