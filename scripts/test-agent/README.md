# test-agent: minimal LLM + tool-server agent loop

This folder builds on `scripts/test-simple/` and adds a **real LLM** (OpenAI-compatible API) to generate tool-call actions.

What this validates:
- LLM produces an **action string**
- Tool server executes the action (`python_code`) and returns an **observation**
- We feed the observation back to the LLM (multi-turn loop)

What this does **not** validate:
- PPO training / weight updates (that is the next step, and is heavier)

## Prereqs
1) Tool-server deps installed (you already have this from `test-simple`).
2) An OpenAI-compatible chat endpoint:
   - **OpenAI**: `OPENAI_BASE_URL=https://api.openai.com`
   - **Local** (vLLM/SGLang): set `OPENAI_BASE_URL` to your local server (e.g. `http://127.0.0.1:8000`)

Environment variables:
- `OPENAI_API_KEY`: required for OpenAI; some local servers accept any value.
- `OPENAI_BASE_URL`: optional; defaults to `https://api.openai.com`
- `OPENAI_MODEL`: the model name (e.g. `gpt-4.1-mini`, or your local model id)

## Run
From `VRTOOL-Framework/verl-tool/` (with `.venv` activated):

```bash
python3 -m pip install -e .
python3 scripts/test-agent/run_agent_loop_openai_compatible.py --tool-server-url http://127.0.0.1:5500/get_observation
```

Tip: If you don’t want to manage the server yourself, start it in another terminal:

```bash
python3 -m verl_tool.servers.tool_server --host 127.0.0.1 --port 5500 --tool_type python_code --workers_per_tool 2 --http h11
```

