#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from typing import Any

import requests


def _port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        return s.connect_ex((host, port)) != 0


def _pick_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def _wait_http_ready(url: str, timeout_s: float) -> None:
    t0 = time.time()
    last_err: Exception | None = None
    while time.time() - t0 < timeout_s:
        try:
            r = requests.post(
                url,
                json={"trajectory_ids": ["health"], "actions": [""], "extra_fields": [{}]},
                timeout=1.0,
            )
            # We only care that the server responds; tool may mark invalid action.
            if r.status_code in (200, 422, 400):
                return
        except Exception as e:  # noqa: BLE001
            last_err = e
        time.sleep(0.2)
    raise RuntimeError(f"Server not ready at {url}. Last error: {last_err!r}")


def _send(url: str, trajectory_id: str, action: str, *, finish: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "trajectory_ids": [trajectory_id],
        "actions": [action],
        "extra_fields": [{}],
    }
    if finish:
        payload["finish"] = [True]

    r = requests.post(url, json=payload, timeout=20.0)
    r.raise_for_status()
    return r.json()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5500)
    ap.add_argument("--workers-per-tool", type=int, default=2)
    ap.add_argument("--startup-timeout-s", type=float, default=15.0)
    ap.add_argument(
        "--verbose-server-logs",
        action="store_true",
        help="Print server log tail even on success.",
    )
    args = ap.parse_args()

    host = args.host
    port = args.port
    if port == 0:
        port = _pick_free_port(host)
    elif not _port_is_free(host, port):
        new_port = _pick_free_port(host)
        print(
            f"[WARN] Port {port} is in use on {host}; switching to free port {new_port}.",
            file=sys.stderr,
        )
        port = new_port

    url = f"http://{host}:{port}/get_observation"

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

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
        str(args.workers_per_tool),
        "--http",
        "h11",
    ]

    print("[INFO] Starting tool server:")
    print("       " + " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    try:
        _wait_http_ready(url, timeout_s=args.startup_timeout_s)
        print(f"[OK] Server is responding at {url}")

        traj = f"test-simple-{int(time.time())}"

        print("[INFO] Test 1: python_code tool executes code")
        resp1 = _send(url, traj, "<python>print('hello from verl-tool')</python>")
        obs1 = (resp1.get("observations") or [""])[0]
        valid1 = (resp1.get("valids") or [None])[0]
        done1 = (resp1.get("dones") or [None])[0]
        if valid1 is not True:
            raise RuntimeError(f"Expected valids[0]==True, got {valid1}. Full resp: {resp1}")
        if done1 not in (False, None):
            raise RuntimeError(f"Expected dones[0] False/None, got {done1}. Full resp: {resp1}")
        if "hello from verl-tool" not in obs1:
            raise RuntimeError(f"Expected output in observation. Got: {obs1!r}")
        print("[OK] Tool executed and returned expected output.")

        print("[INFO] Test 2: finish trajectory")
        resp2 = _send(url, traj, "", finish=True)
        done2 = (resp2.get("dones") or [None])[0]
        if done2 not in (True, None):
            # Some implementations may omit dones; accept None but prefer True.
            raise RuntimeError(f"Expected dones[0] True/None on finish, got {done2}. Full resp: {resp2}")
        print("[OK] Finish path returned successfully.")

        print("\n[PASS] Minimal functional test succeeded.")
        print("       What you validated: tool server + python tool + finish over HTTP.")
        return 0

    finally:
        success = (proc.poll() is not None and proc.returncode == 0)
        if proc.poll() is None:
            print("[INFO] Stopping server...")
            try:
                proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                proc.kill()
                proc.wait(timeout=5)

        if proc.stdout:
            out = proc.stdout.read() or ""
            tail = out[-4000:]
            if tail.strip() and (args.verbose_server_logs or proc.returncode not in (0, None)):
                print("\n--- server log tail (last 4000 chars) ---")
                print(tail)


if __name__ == "__main__":
    raise SystemExit(main())

