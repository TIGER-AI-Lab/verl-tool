# local-llm (Ollama on macOS) — OpenAI-compatible backend

This folder explains how to run `scripts/test-agent/` **fully locally** on an Apple Silicon Mac (M3) using **Ollama** as the LLM server.

Ollama exposes an **OpenAI-compatible** endpoint:
- `http://localhost:11434/v1/chat/completions`

So we can reuse the existing `test-agent` code by setting `OPENAI_BASE_URL` to Ollama.

## 1) Install Ollama
Install Ollama for macOS (one-time). Follow Ollama’s official install instructions.

## 2) Start Ollama
Ollama usually runs a background service automatically. If needed:

```bash
ollama serve
```

## 3) Pull a small model (recommended for laptop)
Examples (pick one):

```bash
ollama pull llama3.2:3b
# or
ollama pull qwen2.5:3b
```

## 4) Start the tool server
From `VRTOOL-Framework/verl-tool/`:

```bash
python3 -m verl_tool.servers.tool_server --host 127.0.0.1 --port 5500 --tool_type python_code --workers_per_tool 2 --http h11
```

## 5) Run the agent loop against Ollama (no OpenAI credits needed)
In the same terminal:

```bash
export OPENAI_BASE_URL="http://localhost:11434"
export OPENAI_API_KEY="ollama"        # required by client, ignored by Ollama
export OPENAI_MODEL="llama3.2:3b"     # must match `ollama list`

python3 scripts/test-agent/run_agent_loop_openai_compatible.py \
  --tool-server-url http://127.0.0.1:5500/get_observation \
  --dump-trajectory-json trajectory_ollama.json
```

If it works, you’ll get:
- console logs for each turn
- `trajectory_ollama.json` with a structured trajectory (actions, observations, latencies, stop_reason)

## Notes (performance / energy)
- Local inference is slower than hosted models, but it’s ideal for **energy/latency instrumentation** because everything runs on your machine.
- Next step for energy: sample system power while the loop runs (macOS `powermetrics` typically requires `sudo`).

