# Debug & Testing Utilities

**Module:** `cube.testing` | **Related design:** `design/stress_test_specs.md`

## Purpose

Framework-level harness for debug episodes and compliance checks. Any CUBE package
that follows the debug-module protocol can be exercised by these utilities — no
benchmark-specific imports required.

## Module protocol

Benchmark packages expose `<package>.debug` with two callables:

```python
def get_debug_benchmark() -> Benchmark:
    """Return a Benchmark instance, optionally pre-filtered to debug tasks
    via subset_from_list. Base class calls install(), setup(), and close() on it."""

def make_debug_agent(task_id: str) -> Callable[[Observation, list[ActionSchema]], Action]:
    """Return a deterministic agent that solves the named task."""
```

## Public API

### `run_debug_episode(task, agent, *, max_steps=20) -> dict`
Runs one complete episode (reset → step* → close) and returns a JSON-serializable
report. Catches all exceptions — errors appear in `report["error"]`, not as raised
exceptions.

Report schema:
```python
{
    "task_id": str,
    "done": bool,
    "reward": float,
    "steps": int,
    "episode_time_s": float,
    "step_times_s": list[float],
    "error": str | None,
    "tools_list_ok": bool,
    "tools_list_error": str,
    "reset_time_s": float,
    "close_idempotent_ok": bool,       # calls close() twice; must not raise
    "profiling": list[dict],            # per-step profiling dicts from env.info
}
```

### `run_debug_suite(benchmark_name, module, *, max_steps=20, workers=0, on_episode_start=None, on_episode_done=None) -> list[dict]`
Discovers tasks via `module.get_debug_benchmark().get_task_configs()`, runs each
with `module.make_debug_agent(task_id)`, and returns a list of episode reports.

`workers=0` (default) runs one thread per task automatically. `workers=1` is sequential.

**Parallel runs (`workers=0` or `workers > 1`):** tasks share the benchmark's
`_runtime_context` by reference. After `setup()` returns, concurrent episodes must
treat that object as read-only. Writes from multiple workers are not safe.

### `assert_debug_tasks_reward_one(module, *, max_steps=20) -> None`
Runs `run_debug_suite` and asserts every task reaches `reward == 1.0`.
Raises `AssertionError` otherwise. Tasks run in parallel by default — treat the
benchmark's `_runtime_context` as read-only after `setup()`. Drop-in for pytest:

```python
def test_debug_tasks():
    from cube.testing import assert_debug_tasks_reward_one
    import my_cube.debug as mod
    assert_debug_tasks_reward_one(mod)
```

### Compliance checks (used by `cube test`)
- `check_benchmark_metadata(module)` → `(ok, err)` — verifies `BenchmarkMetadata` required fields
- `check_reset_reproducibility(module)` → `(ok, err)` — same config × 2 `make()` + `reset()` → identical first obs
- `aggregate_profiling(reports)` — roll-up of per-step profiling dicts
- `build_stress_test_report(...)` — assembles the full compliance report

### `_validate_action_set(action_set)` — internal
Checks action_set is a non-empty list of `ActionSchema` instances. Pydantic already
enforces non-empty name/description on each.

## Invariants

1. `run_debug_episode` catches all exceptions. Never raises.
2. `task.close()` is called twice during the episode to verify idempotency —
   implementers must tolerate double-close.
3. `check_reset_reproducibility` uses only the first task config — quick check, not exhaustive.

## Contracts for benchmark authors

- Expose `debug.py` with `get_debug_benchmark()` and `make_debug_agent(task_id)`.
- Every debug task must be solvable with `reward == 1.0` by the returned agent.
- `get_debug_benchmark()` should return a subset small enough to run quickly
  (typical: 1–5 tasks).
- The agent must be deterministic — seed any randomness internally.
