"""
Pure-HTTP JSON-RPC client for counter-cube — zero cube imports.

Usage:
    uv run python client_raw.py <benchmark-url>

Example:
    uv run python client_raw.py http://127.0.0.1:8765

Identical behaviour to client.py but implemented with raw httpx calls and
inline JSON-RPC helpers.  Demonstrates that any HTTP client in any language
can drive a CUBE benchmark using only the wire protocol.
"""

import argparse
import json
import sys
import time
from typing import Any

import httpx

_req_id = 0


def rpc(url: str, method: str, params: dict | None = None) -> Any:
    global _req_id
    _req_id += 1
    body: dict = {"jsonrpc": "2.0", "method": method, "id": _req_id}
    if params is not None:
        body["params"] = params
    resp = httpx.post(url, json=body, timeout=10.0)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"JSON-RPC error {data['error']['code']}: {data['error']['message']}")
    return data["result"]


def wait_for_server(url: str, retries: int = 30, delay: float = 0.5) -> None:
    for _ in range(retries):
        try:
            if httpx.post(url, json={}, timeout=5.0).status_code == 200:
                return
        except httpx.ConnectError as e:
            print(f"Waiting for server at {url} to be ready... (error: {e})")
        time.sleep(delay)
    raise RuntimeError(f"Server at {url} did not become ready after {retries} attempts")


def main() -> None:
    parser = argparse.ArgumentParser(description="counter-cube raw HTTP client")
    parser.add_argument("url", help="Benchmark server URL (e.g. http://127.0.0.1:8765)")
    args = parser.parse_args()
    bench_url = args.url

    wait_for_server(bench_url)

    # ── 1. Benchmark info ─────────────────────────────────────────────────────
    print("=== cube/info ===")
    print(json.dumps(rpc(bench_url, "cube/info"), indent=2))

    # ── 2. List tasks ─────────────────────────────────────────────────────────
    print("\n=== cube/tasks ===")
    tasks = rpc(bench_url, "cube/tasks")
    print(json.dumps(tasks, indent=2))

    # ── 3. Get a ready-to-use task config for the first task ─────────────────
    task_id = tasks[0]["id"]
    print(f"\n=== cube/task_configs (task_id={task_id!r}) ===")
    task_configs = rpc(bench_url, "cube/task_configs", {"task_id": task_id})
    print(json.dumps(task_configs, indent=2))

    # ── 4. Spawn first task ───────────────────────────────────────────────────
    print("\n=== cube/spawn ===")
    task_url = rpc(bench_url, "cube/spawn", {"task_config": task_configs[0]})
    print(f"Task server spawning at {task_url} ...")
    wait_for_server(task_url)
    print("ready.")

    # ── 5. Available tools ────────────────────────────────────────────────────
    print("\n=== tools/list ===")
    print(json.dumps(rpc(task_url, "tools/list"), indent=2))

    # ── 6. Reset ──────────────────────────────────────────────────────────────
    print("\n=== cube/reset ===")
    print(json.dumps(rpc(task_url, "cube/reset"), indent=2))

    # ── 7. Episode loop ───────────────────────────────────────────────────────
    step = 0
    done = False
    while not done:
        step += 1
        print(f"\n=== cube/step {step} (increment) ===")
        result = rpc(task_url, "cube/step", {"name": "increment"})
        print(json.dumps(result, indent=2))
        done = result["done"]

    # ── 8. Final evaluation ───────────────────────────────────────────────────
    print("\n=== cube/evaluate ===")
    print(json.dumps(rpc(task_url, "cube/evaluate"), indent=2))

    # ── 9. Clean up ───────────────────────────────────────────────────────────
    rpc(task_url, "cube/close")
    rpc(bench_url, "cube/shutdown")
    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
