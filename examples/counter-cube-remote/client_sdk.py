"""
Remote client for a counter-cube benchmark server.

Usage:
    uv run python client.py <benchmark-url>

Example:
    uv run python client.py http://127.0.0.1:8765

Connects to the benchmark server, picks the first available task, spawns a
task server, runs a full episode, and prints every response as pretty JSON.
Uses only cube.client — no benchmark-specific imports needed.
"""

import argparse
import json
import sys

from cube.client import BenchmarkClient


def main() -> None:
    parser = argparse.ArgumentParser(description="counter-cube remote client")
    parser.add_argument("url", help="Benchmark server URL (e.g. http://127.0.0.1:8765)")
    args = parser.parse_args()

    bench = BenchmarkClient(args.url)

    # ── 1. Benchmark info ─────────────────────────────────────────────────────
    print("=== cube/info ===")
    print(json.dumps(bench.info(), indent=2))

    # ── 2. List tasks ─────────────────────────────────────────────────────────
    print("\n=== cube/tasks ===")
    tasks = bench.tasks()
    print(json.dumps(tasks, indent=2))

    # ── 3. Get a ready-to-use task config for the first task ─────────────────
    task_id = tasks[0]["id"]
    print(f"\n=== cube/task_configs (task_id={task_id!r}) ===")
    configs = bench.task_configs(task_id=task_id)
    print(json.dumps(configs, indent=2))

    # ── 4. Spawn — benchmark server starts a subprocess, we poll until ready ─
    print("\n=== cube/spawn ===")
    task = bench.spawn(configs[0])
    print(f"Task server ready at {task.url}")

    # ── 5. Available tools ────────────────────────────────────────────────────
    print("\n=== tools/list ===")
    print(json.dumps(task.tools_list(), indent=2))

    # ── 6. Reset ──────────────────────────────────────────────────────────────
    print("\n=== cube/reset ===")
    print(json.dumps(task.reset(), indent=2))

    # ── 7. Episode loop ───────────────────────────────────────────────────────
    step = 0
    done = False
    while not done:
        step += 1
        print(f"\n=== cube/step {step} (increment) ===")
        result = task.step("increment")
        print(json.dumps(result, indent=2))
        done = result["done"]

    # ── 8. Final evaluation ───────────────────────────────────────────────────
    print("\n=== cube/evaluate ===")
    print(json.dumps(task.evaluate(), indent=2))

    # ── 9. Clean up ───────────────────────────────────────────────────────────
    task.close()
    bench.shutdown()
    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
