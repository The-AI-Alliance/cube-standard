# CUBE Stress Test Specification

> **CUBE Layer:** Developer Tooling / Quality Assurance
> **Related:** [main_specs.md](main_specs.md) | [user_experience.md](user_experience.md)

---

## Part 1: CUBE Standard Additions

### 1.1 Benchmark-Level: `cube/debug_tasks`

```python
# Python
debug_task_list = benchmark.get_debug_task_configs()
```

```http
GET /cube/debug_tasks
```

**Requirements:**

- At least one debug task must be exposed
- Debug tasks must be solvable in 5-10 steps (enough to measure transition time)
- Debug tasks must not depend on external services beyond what `benchmark.setup()` provisions
- Debug tasks must be deterministic given a fixed seed
- Debug tasks appear only in `cube/debug_tasks`, not in `cube/tasks`

### 1.2 Package-Level: `make_debug_agent`

```python
import my_cube

debug_agent = my_cube.make_debug_agent(task_id)
action = debug_agent.get_action(obs)
```

**Requirements:**

- Must be implemented for every task_id in `cube/debug_tasks`
- Hardcoded action sequence (no LLM, no external dependencies)
- Must complete the debug task with `reward > 0`
- Standard interface:

```python
class DebugAgent:
    def get_action(self, obs: Observation) -> Action: ...
```

### 1.3 Profiling Info (Optional)

Tasks may return profiling data in `info["profiling"]` from `step()`:

```python
info = {
    "profiling": {
        "container_exec": (1708175234.123, 1708175234.567),  # (start, stop) timestamps
        "tool_render": (1708175234.580, 1708175234.620),
        # ... other instrumented operations
    }
}
```

The CUBE framework will report these if present.

---

## Part 2: CUBE-Developer Responsibilities

### 2.1 Debug task implementation

The debug task should be cheap to run but exercise the full stack. It does **not** need special logic for testing — the CUBE framework harness handles `setup()`, `step()`, `evaluate()`, `close()`. The debug task just needs to be solvable by the debug agent in 5-10 steps.

Good examples:
- SWE-bench: edit a trivial file that passes a pre-written test
- WebArena: navigate to a URL, verify an element exists
- OSWorld: click a button, verify state change

### 2.2 Debug agent implementation

The agent must know the exact action sequence. It can be stateless (same actions every time) or use simple state tracking (step counter). No dependencies beyond the benchmark package.

### 2.3 Optional: Profiling instrumentation

If the benchmark has expensive internal operations (container exec, VM API calls, rendering), add timestamps to `info["profiling"]` so the stress test can report them. Format:

```python
info["profiling"] = {
    "operation_name": (start_timestamp, stop_timestamp),
    ...
}
```

### 2.4 Checklist before publishing

- [ ] `get_debug_task_configs()` returns ≥ 1 task
- [ ] `make_debug_agent(task_id)` implemented for each debug task
- [ ] Debug agent completes task with `reward > 0` in 5-10 steps
- [ ] `cube.testing.run_stress_test(my_cube)` passes all compliance checks
- [ ] Baseline results committed as `cube_stress_test_baseline.json`

---

## Part 3: CUBE Framework Implementation

### 3.1 MVP

#### Mini harness

The stress test provides a minimal evaluation harness so CUBE-Developers don't need to write custom test code:

```python
# In cube.testing
def run_debug_episode(benchmark, task_config, debug_agent):
    """Run one full episode of a debug task."""
    task = task_config.make(metadata=..., runtime_context=benchmark.runtime_context)

    obs, info = task.setup()
    steps = 0
    profiling = []

    while steps < 20:  # Safety limit
        action = debug_agent.get_action(obs)
        result = task.step(action)
        obs = result.obs

        if "profiling" in result.info:
            profiling.append(result.info["profiling"])

        if result.done:
            break
        steps += 1

    reward, eval_info = task.evaluate(obs)
    task.close()

    return {
        "done": result.done,
        "reward": reward,
        "steps": steps,
        "profiling": profiling
    }
```

#### Compliance checks (MVP)

| Test | What is checked |
|---|---|
| `test_debug_tasks_exist` | `get_debug_task_configs()` returns ≥ 1 task |
| `test_debug_agent_exists` | `make_debug_agent(task_id)` succeeds |
| `test_full_episode` | Debug agent reaches `done=True` with `reward > 0` |
| `test_reset_reproducibility` | Same seed → identical first observation |
| `test_tools_list` | `task.action_set` is non-empty after `setup()` |
| `test_close_idempotent` | `task.close()` twice does not raise |
| `test_benchmark_metadata` | `benchmark.metadata` has non-empty `name`, `version` |

#### Performance metrics (MVP)

All timing metrics reported in seconds.

| Metric | How measured |
|---|---|
| **Benchmark setup time** | Wall time for `benchmark.setup()` |
| **Task setup time** | Wall time for `task.setup()` (averaged over 3 runs) |
| **Step latency** | p50, p95, p99 over 20 consecutive steps |
| **Teardown time** | Wall time for `benchmark.close()` |
| **Episode time** | Total wall time for one full episode |
| **Memory delta** | Process RSS after episode - RSS before |
| **Profiling operations** | If `info["profiling"]` present, report mean duration for each operation |

#### MVP usage

```python
import my_cube
from cube.testing import run_stress_test

report = run_stress_test(my_cube)
report.print_summary()
report.save("cube_stress_test_baseline.json")
```

#### MVP output

```json
{
  "cube_version": "0.1.0",
  "benchmark": "my_cube",
  "timestamp": "2026-02-17T10:00:00Z",
  "hardware": {
    "cpu_count": 4,
    "ram_gb": 16,
    "storage_type": "ssd",
    "python_version": "3.11.4"
  },
  "compliance": {
    "passed": ["test_debug_tasks_exist", "test_full_episode", "..."],
    "failed": []
  },
  "performance": {
    "benchmark_setup_time_s": 8.4,
    "task_setup_time_s": 1.2,
    "step_latency_p50_s": 0.038,
    "step_latency_p95_s": 0.072,
    "step_latency_p99_s": 0.140,
    "teardown_time_s": 1.1,
    "episode_time_s": 6.8,
    "memory_delta_mb": 45,
    "profiling": {
      "container_exec_s": 0.0224,
      "tool_render_s": 0.0038
    }
  }
}
```

---

### 3.2 Final Solution

#### Additional compliance checks

| Test | What is checked |
|---|---|
| `test_graceful_shutdown` | `benchmark.close()` while task is mid-run terminates cleanly |
| `test_rpc_mode` | Full episode completes via RPC |
| `test_privileged_info` | `task.get_privileged_info()` returns without error |

#### Scalability metrics

Run with N = 1, 2, 4 Ray workers.

| Metric | How measured |
|---|---|
| **Throughput** | Tasks/minute at N = 1, 2, 4 |
| **Parallel efficiency** | `(throughput_N / N) / throughput_1` at N = 4. Target ≥ 0.7 |
| **Memory per worker** | RSS per Ray worker |
| **Cross-worker isolation** | Two workers on same task_id both get `reward > 0` |
| **Crash recovery** | Kill one worker mid-task, verify others complete + relaunched task succeeds |

#### CLI

```bash
cube stress-test my_cube --output results.json
cube stress-test my_cube --compliance-only
cube stress-test my_cube --workers 1,2,4
```

#### GitHub Actions integration

Template workflow:

```yaml
name: CUBE Stress Test
on: [push, pull_request]

jobs:
  stress-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install cube my_cube
      - run: |
          python -c "
          import my_cube
          from cube.testing import run_stress_test
          report = run_stress_test(my_cube)
          report.save('results.json')
          "
      - uses: actions/upload-artifact@v4
        with:
          name: stress-test-results
          path: results.json
```

**Cost estimate (10 min run):**
- Public repo: free
- Private repo: ~$0.08/run
- With cloud VM: ~$0.10/run total

#### Registry CI integration

On benchmark submission:
1. Install from PyPI
2. Run compliance-only check
3. Block if any compliance check fails
4. Store baseline in registry metadata

---


## Appendix: Terminal Dashboard (MVP+)

The stress test includes a live terminal dashboard using the `rich` library for real-time visualization during test execution.

### Usage

```python
import my_cube
from cube.testing import run_stress_test

# With live dashboard
report = run_stress_test(my_cube, live_dashboard=True)
```

```bash
# CLI (final phase)
cube stress-test my_cube --live
```

### Visual Layout

```
┌─ CUBE Stress Test ────────────────────────────────────────────┐
│ Benchmark: swe-bench-lite                    Status: Running   │
│ Workers: 4/4 active                         Progress: 67/100   │
├───────────────────────────────────────────────────────────────┤
│ COMPLIANCE                                                     │
│ ✓ debug_tasks_exist        ✓ full_episode                     │
│ ✓ debug_agent_exists       ✓ reset_reproducibility            │
│ ✓ tools_list               ✓ close_idempotent                 │
│ ✓ benchmark_metadata                                           │
├───────────────────────────────────────────────────────────────┤
│ LATENCY (seconds)                                              │
│ p50 │████████░░░░░░░░░░░░░░░░░░░░░░│ 0.042s                  │
│ p95 │████████████████░░░░░░░░░░░░░░│ 0.089s                  │
│ p99 │████████████████████░░░░░░░░░░│ 0.134s                  │
├───────────────────────────────────────────────────────────────┤
│ THROUGHPUT (tasks/min)                                         │
│ Workers │ Measured │ Illustrative │ Linear │ Efficiency         │
│    1    │  12.4  │  12.4  │ █████████████████████████ 100%    │
│    2    │  23.1  │  24.8  │ ███████████████████████░░  93%    │
│    4    │  41.8  │  49.6  │ █████████████████████░░░░  84%    │
├───────────────────────────────────────────────────────────────┤
│ PROFILING BREAKDOWN                                            │
│ container_exec  ██████████████████░░░░░░░░  0.0224s (59%)    │
│ tool_render     ████░░░░░░░░░░░░░░░░░░░░░░  0.0038s (10%)    │
│ other           ████████░░░░░░░░░░░░░░░░░░  0.0118s (31%)    │
└───────────────────────────────────────────────────────────────┘
```

### Features

- **Live updates:** Refreshes as tests complete (compliance → performance → scalability)
- **Progress tracking:** Shows current test phase and completion percentage
- **Color coding:**
  - Green: Tests passed
  - Red: Tests failed
  - Yellow: In progress
- **Bars scale dynamically** based on actual values
- **Works in terminal recordings** (asciinema) for documentation

### Implementation Notes

Uses `rich.live.Live` for updating display and `rich.progress.Progress` for bars. Falls back to JSON output if terminal doesn't support rich rendering (CI environments).

### Demo Value

- **Live presentations:** Run side-by-side for fast vs heavy benchmarks
- **GitHub README:** Embed asciinema recording showing test execution
- **Documentation:** Screenshot for quick-start guides
