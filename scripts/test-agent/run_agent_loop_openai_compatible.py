#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any

import httpx
import requests


def _now_ms() -> int:
    return int(time.time() * 1000)


def _looks_like_python_tool_action(text: str) -> bool:
    t = text or ""
    return ("<python>" in t and "</python>" in t) or ("```python" in t)


def _short(s: Any, n: int = 240) -> str:
    x = s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)
    x = x.replace("\n", "\\n")
    return x[:n] + ("…" if len(x) > n else "")


@dataclass
class OpenAICompat:
    base_url: str
    api_key: str
    model: str
    timeout_s: float = 60.0
    max_retries: int = 5

    def chat(self, messages: list[dict[str, Any]]) -> str:
        base = self.base_url.rstrip("/")
        # Accept either:
        # - base_url="https://api.openai.com" (no /v1 suffix)
        # - base_url="http://localhost:11434/v1" (Ollama-style)
        if base.endswith("/v1"):
            url = base + "/chat/completions"
        else:
            url = base + "/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
        }

        last_err: Exception | None = None
        with httpx.Client(timeout=self.timeout_s) as client:
            for attempt in range(1, self.max_retries + 1):
                try:
                    r = client.post(url, headers=headers, json=payload)
                    r.raise_for_status()
                    data = r.json()
                    break
                except httpx.HTTPStatusError as e:
                    last_err = e
                    status = e.response.status_code
                    if status in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                        retry_after = e.response.headers.get("retry-after")
                        sleep_s = float(retry_after) if retry_after and retry_after.isdigit() else min(2.0**attempt, 20.0)
                        print(f"[WARN] LLM HTTP {status}. Retrying in {sleep_s:.1f}s (attempt {attempt}/{self.max_retries})...", file=sys.stderr)
                        time.sleep(sleep_s)
                        continue
                    # Surface response body (truncated) for debugging/quota messages.
                    body = (e.response.text or "")[:2000]
                    raise RuntimeError(f"LLM request failed with HTTP {status}. Body (first 2000 chars): {body}") from e
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    if attempt < self.max_retries:
                        sleep_s = min(2.0**attempt, 20.0)
                        print(f"[WARN] LLM request error {type(e).__name__}: {e}. Retrying in {sleep_s:.1f}s...", file=sys.stderr)
                        time.sleep(sleep_s)
                        continue
                    raise
            else:
                raise RuntimeError(f"LLM request failed after retries. Last error: {last_err!r}")

        try:
            return data["choices"][0]["message"]["content"] or ""
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Unexpected OpenAI-compatible response shape: {json.dumps(data)[:2000]}") from e


def send_tool_action(tool_server_url: str, trajectory_id: str, action: str, *, finish: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "trajectory_ids": [trajectory_id],
        "actions": [action],
        "extra_fields": [{}],
    }
    if finish:
        payload["finish"] = [True]

    r = requests.post(tool_server_url, json=payload, timeout=30.0)
    r.raise_for_status()
    return r.json()


SYSTEM_PROMPT = """You are an agent that can use a Python tool via a tool server.

When you want to run Python, output ONLY one tool action using one of these formats:

<python>
print("hello")
</python>

OR

```python
print("hello")
```

After the tool runs, you will receive an observation containing the output.

Goal: solve the user task correctly. Prefer using the Python tool for calculations."""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tool-server-url",
        default="http://127.0.0.1:5500/get_observation",
        help="Tool server endpoint (POST /get_observation).",
    )
    ap.add_argument("--max-turns", type=int, default=3)
    ap.add_argument(
        "--task",
        default="Compute (12345 * 6789) and return the exact integer.",
        help="A simple task that benefits from python_code tool use.",
    )
    ap.add_argument(
        "--dump-trajectory-json",
        default="",
        help="If set, write a structured trajectory JSON to this path.",
    )
    args = ap.parse_args()

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com")
    api_key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("OPENAI_MODEL", "")
    if not model:
        print("[FAIL] Missing env var OPENAI_MODEL (e.g. gpt-4.1-mini, or your local model id).", file=sys.stderr)
        return 2
    if base_url.startswith("https://api.openai.com") and not api_key:
        print("[FAIL] Missing OPENAI_API_KEY for OpenAI.", file=sys.stderr)
        return 2

    llm = OpenAICompat(base_url=base_url, api_key=api_key, model=model)
    trajectory_id = f"test-agent-{uuid.uuid4().hex[:10]}"

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": args.task},
    ]

    rollout: dict[str, Any] = {
        "trajectory_id": trajectory_id,
        "task": args.task,
        "tool_server_url": args.tool_server_url,
        "openai_compat": {"base_url": base_url, "model": model},
        "started_at_ms": _now_ms(),
        "turns": [],
        "stop_reason": None,
        "final_answer": None,
    }

    print("[INFO] trajectory_id=", trajectory_id)
    print("[INFO] tool_server_url=", args.tool_server_url)
    print("[INFO] openai_compat_base_url=", base_url)
    print("[INFO] model=", model)
    print()

    for turn in range(1, args.max_turns + 1):
        print(f"[INFO] Turn {turn}: asking LLM for next action/answer...")
        llm_t0 = time.perf_counter()
        assistant = llm.chat(messages)
        llm_latency_ms = (time.perf_counter() - llm_t0) * 1000.0
        print("--- LLM output ---")
        print(assistant.strip())
        print("------------------\n")

        # If the LLM produced a tool action, execute it; otherwise stop.
        looks_like_tool = _looks_like_python_tool_action(assistant)
        if not looks_like_tool:
            print("[OK] LLM produced a final answer (no tool call detected).")
            print("[FINAL ANSWER]")
            print(assistant.strip())
            rollout["stop_reason"] = "final_answer_no_tool_call"
            rollout["final_answer"] = assistant.strip()
            send_tool_action(args.tool_server_url, trajectory_id, "", finish=True)
            rollout["finished_at_ms"] = _now_ms()
            if args.dump_trajectory_json:
                with open(args.dump_trajectory_json, "w", encoding="utf-8") as f:
                    json.dump(rollout, f, ensure_ascii=False, indent=2)
            return 0

        tool_t0 = time.perf_counter()
        resp = send_tool_action(args.tool_server_url, trajectory_id, assistant, finish=False)
        tool_latency_ms = (time.perf_counter() - tool_t0) * 1000.0
        obs = (resp.get("observations") or [""])[0]
        valid = (resp.get("valids") or [None])[0]
        done = (resp.get("dones") or [None])[0]

        print("[INFO] tool_response valid=", valid, "done=", done)
        print("--- observation (from tool server) ---")
        # observation may be dict or string depending on tool/server path
        if isinstance(obs, dict):
            print(json.dumps(obs, indent=2)[:4000])
        else:
            print(str(obs)[:4000])
        print("-------------------------------------\n")

        rollout["turns"].append(
            {
                "turn": turn,
                "llm_output": assistant,
                "llm_latency_ms": llm_latency_ms,
                "tool_called": True,
                "tool_type_assumed": "python_code",
                "tool_latency_ms": tool_latency_ms,
                "tool_response_valid": valid,
                "tool_response_done": done,
                "tool_observation": obs,
                "tool_raw_response": resp,
            }
        )

        # Feed back into LLM as a new user message (simple, but works).
        messages.append({"role": "assistant", "content": assistant})
        messages.append({"role": "user", "content": f"Tool observation:\n{obs}\nNow answer the original task."})

        # Small pause to be friendly to rate limits.
        time.sleep(0.2)

    print("[FAIL] Max turns reached without a final answer.", file=sys.stderr)
    rollout["stop_reason"] = "max_turns_reached"
    send_tool_action(args.tool_server_url, trajectory_id, "", finish=True)
    rollout["finished_at_ms"] = _now_ms()
    if args.dump_trajectory_json:
        with open(args.dump_trajectory_json, "w", encoding="utf-8") as f:
            json.dump(rollout, f, ensure_ascii=False, indent=2)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

