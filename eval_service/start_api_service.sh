#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

# Shared host IP for both tool server and API service
export HOST=${HOST:-"0.0.0.0"}

# Save current directory
CUR_DIR=$(pwd)

# Change to project root to launch the tool server
cd "$(dirname "$0")/../"  # Assuming script is in verl_tool/eval_service/

# Start the tool server
port=$(shuf -i 30000-31000 -n 1)
tool_server_url=http://$HOST:$port/get_observation
python -m verl_tool.servers.serve --host $HOST --port $port --tool_type "python_code" &
server_pid=$!
echo "Tool server (pid=$server_pid) started at $tool_server_url"

# Return to original directory
cd "$CUR_DIR"

# Set default environment variables for the API service
export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-1.5B-Instruct"}
export PORT=${PORT:-"8000"}
export TOOL_SERVER_URL=${TOOL_SERVER_URL:-$tool_server_url}
export MAX_TURNS=${MAX_TURNS:-"5"}
export VALID_ACTIONS=${VALID_ACTIONS:-'["python"]'}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-"0.9"}
export TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-"1"}

# Print service configuration
echo "Starting LLM Tool API Service"
echo "Model path: $MODEL_PATH"
echo "Tool server URL: $TOOL_SERVER_URL"

# Launch the API service
python app.py
