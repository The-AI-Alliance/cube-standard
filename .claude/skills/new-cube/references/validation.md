# The 3 validation levels

CUBE cubes validate in three layers. Run them in order. Green at each level before moving to the next.

## Level 1 — pytest

**What:** Unit tests of tool / task / benchmark / resource code in isolation.

**When:** Phase 5, after all three implementation layers (tool, task, benchmark) are filled.

**How:** `pytest tests/` in the cube directory. The template ships smoke tests that check:
- `benchmark_metadata` and `task_metadata` non-empty, no `TODO` placeholders left
- Task IDs match (`task_metadata` dict keys == `TaskMetadata.id` values)
- Debug tasks reach `reward == 1.0` (duplicates Level 2 but catches obvious breakage early)

**Failure pattern:** fix the cube code until green before running `cube test`.

## Level 2 — debug module (`cube test`)

**What:** A mini harness runs a hardcoded "debug agent" on each debug task. Tests the actual task dynamics: `reset → step → evaluate → close`.

**When:** Phase 6, once pytest is green.

**How:** `cube test <benchmark-id>` from the cube directory or any parent. The compliance suite runs:
- `test_debug_tasks_exist` — at least one debug task
- `test_debug_agent_exists` — `make_debug_agent` returns a callable
- `test_full_episode` — every task reaches `reward == 1.0`
- `test_reset_reproducibility` — two resets ⇒ identical first obs
- `test_tools_list` — `action_set` non-empty list of `ActionSchema`
- `test_close_idempotent` — `task.close()` called twice safely
- `test_benchmark_metadata` — no TODO placeholders, non-empty name/version/description

Also reports latency (p50/p95/p99), throughput (1/2/4 workers), and profiling breakdown.

**Failure patterns:**
- `reward < 1.0` → wrong action sequence in `_TASK_ACTIONS`, or `evaluate()` logic is off.
- Reset not reproducible → `reset()` isn't deterministic, or `self.tool.reset()` missing.
- Close not idempotent → `close()` holds mutable state; guard with an `_already_closed` flag or make operations safe to repeat.

## Level 3 — real agent via recipe (optional)

**What:** Run a prompted LLM agent end-to-end via a harness recipe. Goal: reproduce published numbers if the benchmark is from literature.

**When:** Phase 8. Only after levels 1 and 2 are green. Optional — user can decline.

**How:** Draft a recipe modeled on `cube-harness/recipes/hello_miniwob.py`. The user runs it themselves (requires LLM credentials, compute, and a harness install).

**What to inspect in the trajectory:**
- Does the agent read the opening obs and choose appropriate actions?
- Does the reward distribution match published numbers (modulo model / prompt differences)?
- Any systemic failure mode across tasks (e.g. a missing action the agent wants to call)?

## Rule of thumb

- Don't offer Level 3 until Levels 1 and 2 are green.
- Don't promote a cube to the registry (phase 7) until Level 2 is green.
