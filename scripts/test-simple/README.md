# test-simple: minimal functional validation (tool server)

This folder contains a **minimal functional end-to-end** test for `verl-tool` that you can run on a laptop:

- starts the tool server (`verl_tool.servers.tool_server`) with the `python_code` tool
- sends real HTTP requests to `/get_observation`
- verifies:
  - the tool parses an action containing `<python>...</python>`
  - returns an observation with execution output
  - the finish path works

## Mental model (agent / action space / observation / reward)
- **agent (in this demo)**: the “agent” is just the test script acting as a client. It sends actions to the tool server.
  - In full training, the agent would be an LLM policy that produces tool-call strings.
- **action space**: **strings** (the raw LLM output) that the tool parses.
  - For `python_code`, a valid action contains code in `<python>...</python>` or ```python ...```.
  - See parsing logic in `verl_tool/servers/tools/python_code.py` (`PythonCodeTool.parse_action`).
- **observation / state**: the server returns `observations` (plus `valids` and `dones`) for each `trajectory_id`.
  - The HTTP contract is implemented in `verl_tool/servers/tool_server.py` (endpoint `/get_observation`).
- **reward**: this demo does **not** compute reward. It validates the tool environment interface only.
  - In PPO training, rewards are computed in `verl_tool/trainer/ppo/reward.py` and reward managers under `verl_tool/workers/reward_manager/`.

## Why this is the right “first proof”
This validates the **tool-as-environment** contract the agent loop and RL training depend on:
`trajectory_id` + `action` → `observation`, `done`, `valid`.

It does **not** run PPO training or a real LLM backend (those require GPUs / vLLM / SGLang).

## Run
From `VRTOOL-Framework/verl-tool/` (with your venv activated):

```bash
python3 -m pip install -e .
python3 scripts/test-simple/run_tool_server_and_test.py
```

If port `5500` is busy, use:

```bash
python3 scripts/test-simple/run_tool_server_and_test.py --port 5501
```

## Notes about warnings
On macOS + newer Python versions you may see warnings on shutdown (e.g. `resource_tracker` about leaked semaphores).
If you see **`[PASS] Minimal functional test succeeded.`**, the functional validation still succeeded.

