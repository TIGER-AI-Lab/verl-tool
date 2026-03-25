#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx


def port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        return s.connect_ex((host, port)) != 0


def wait_ollama_ready(base_url: str, timeout_s: float = 10.0) -> None:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    t0 = time.time()
    last_err: Exception | None = None
    while time.time() - t0 < timeout_s:
        try:
            with httpx.Client(timeout=2.0) as client:
                # Minimal request; may return 400 if model missing, but proves server is up.
                r = client.post(url, json={"model": "missing-model", "messages": [{"role": "user", "content": "ping"}]})
                if r.status_code in (200, 400, 404, 422):
                    return
        except Exception as e:  # noqa: BLE001
            last_err = e
        time.sleep(0.2)
    raise RuntimeError(f"Ollama not responding at {url}. Last error: {last_err!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ollama-base-url", default="http://localhost:11434/v1")
    ap.add_argument("--ollama-model", default="llama3.2:3b")
    ap.add_argument("--tool-port", type=int, default=5500)
    ap.add_argument("--dump", default="trajectory_ollama.json")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]  # <verl-tool>/
    assert (repo_root / "verl_tool").exists(), f"Expected verl-tool root at {repo_root}"

    # Ensure Ollama is up
    wait_ollama_ready(args.ollama_base_url)

    # Start tool server if not running
    host = "127.0.0.1"
    port = args.tool_port
    if not port_is_free(host, port):
        print(f"[INFO] Tool server port {port} is in use; assuming server already running.")
        tool_proc = None
    else:
        cmd = [
            sys.executable,
            "-m",
            "verl_tool.servers.tool_server",
            "--host",
            host,
            "--port",
            str(port),
            "--tool_type",
            "python_code",
            "--workers_per_tool",
            "2",
            "--http",
            "h11",
        ]
        print("[INFO] Starting tool server:")
        print("       " + " ".join(cmd))
        tool_proc = subprocess.Popen(cmd, cwd=str(repo_root), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1.0)

    env = os.environ.copy()
    env["OPENAI_BASE_URL"] = args.ollama_base_url
    env["OPENAI_API_KEY"] = env.get("OPENAI_API_KEY", "ollama")
    env["OPENAI_MODEL"] = args.ollama_model

    agent_script = repo_root / "scripts" / "test-agent" / "run_agent_loop_openai_compatible.py"
    cmd2 = [
        sys.executable,
        str(agent_script),
        "--tool-server-url",
        f"http://{host}:{port}/get_observation",
        "--dump-trajectory-json",
        str(Path(args.dump).resolve()),
    ]
    print("[INFO] Running agent loop against Ollama:")
    print("       " + " ".join(cmd2))
    r = subprocess.run(cmd2, cwd=str(repo_root), text=True, env=env)

    if tool_proc is not None and tool_proc.poll() is None:
        tool_proc.terminate()
        try:
            tool_proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            tool_proc.kill()

    return int(r.returncode)


if __name__ == "__main__":
    raise SystemExit(main())

