"""
CUBE testing utilities — framework-level harness for debug episodes.

Public API
----------
run_debug_episode(task, agent, *, max_steps)  →  dict
run_debug_suite(benchmark_name, module, *, max_steps, workers=1)  →  list[dict]
assert_debug_tasks_reward_one(module, *, max_steps)  →  None

Module protocol (for assert_debug_tasks_reward_one and run_debug_suite)
----------------------------------------------------------------------
The ``module`` argument must expose two callables:

    get_debug_benchmark() -> Benchmark
        Called once before any debug episodes run. Returns a Benchmark instance
        (optionally pre-filtered to the debug subset via ``subset_from_list``).
        The harness calls ``install()``, ``setup()``, and ``close()`` on it and
        iterates ``get_task_configs()`` to discover which tasks to run.

    make_debug_agent(task_id: str) -> Callable[[Observation, list[ActionSchema]], Action]
        Return a deterministic agent for the given task_id.

    Parallel runs (``run_debug_suite(..., workers>1)``): tasks share the benchmark's
    ``_runtime_context`` by reference. After ``setup()`` returns, concurrent episodes
    must treat that object as read-only; writing to it during execution is not safe
    with multiple workers.

Example usage in a test file::

    def test_debug_tasks():
        from cube.testing import assert_debug_tasks_reward_one
        import osworld_cube.debug as _mod
        assert_debug_tasks_reward_one(_mod)
"""

from __future__ import annotations

import json
import logging
import platform
import time
import types
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from cube import __version__  # report.cube_version and .save()
from cube.core import Action, ActionSchema, Observation
from cube.task import Task

# Returned when two ``reset()`` observations differ; CLI uses this to decide contextual wording.
RESET_REPRO_OBS_MISMATCH_MSG = "first observation differed between two resets"

logger = logging.getLogger(__name__)


def _validate_action_set(action_set: list) -> tuple[bool, str]:
    """
    Validate action_set per stress_test_specs.md tools_list check (Option A).

    Each item must be a valid ActionSchema (Pydantic enforces non-empty name/description).
    Returns (True, "") if non-empty and all items are ActionSchema instances.
    """
    if not action_set or not isinstance(action_set, list):
        return False, "action_set is empty or not a list"
    for i, item in enumerate(action_set):
        if not isinstance(item, ActionSchema):
            return False, f"action_set[{i}] is not an ActionSchema"
    return True, ""


def run_debug_episode(
    task: Task,
    agent: Callable[[Observation, list[ActionSchema]], Action],
    *,
    max_steps: int = 20,
) -> dict:
    """
    Run one complete debug episode and return a minimal JSON-serialisable report.

    This is the generic harness from stress_test_specs.md §3.1. It works with
    any CUBE Task subclass and any agent callable — no benchmark-specific imports.

    Report schema (subset of the stress-test MVP output)::

        {
            "task_id": "simple-create-file",
            "done": true,
            "reward": 1.0,
            "steps": 6,
            "episode_time_s": 74.3,
            "step_times_s": [0.21, 63.1, 0.05, 0.04, 0.04, 0.03],
            "error": null
        }

    Args:
        task:       A fully-constructed Task instance (not yet reset).
        agent:      Callable with signature ``(obs, action_set) → Action``.
                    Compatible with ``DebugAgent.__call__`` and ``DebugAgent.get_action``.
        max_steps:  Safety cap on the step loop (default 20).

    Returns:
        dict with keys: task_id, done, reward, steps, episode_time_s,
        step_times_s, error.
    """
    task_id = task.metadata.id
    logger.info(f"[run_debug_episode] Starting episode for task={task_id!r}")

    report: dict = {
        "task_id": task_id,
        "done": False,
        "reward": 0.0,
        "steps": 0,
        "episode_time_s": 0.0,
        "step_times_s": [],
        "error": None,
        "tools_list_ok": False,
        "tools_list_error": "",
        "reset_time_s": 0.0,
        "close_idempotent_ok": False,
        "profiling": [],
    }

    episode_start = time.perf_counter()
    try:
        logger.info(f"[run_debug_episode] task={task_id!r}  calling reset() …")
        t0 = time.perf_counter()
        obs, info = task.reset()
        reset_time = time.perf_counter() - t0
        report["reset_time_s"] = round(reset_time, 4)
        logger.info(
            f"[run_debug_episode] task={task_id!r}  reset done in {reset_time:.1f}s  info={info}  obs={obs.to_markdown()}"
        )

        # tools_list compliance: non-empty action_set with name, description, parameters per schema
        tools_ok, tools_msg = _validate_action_set(getattr(task, "action_set", None) or [])
        report["tools_list_ok"] = tools_ok
        report["tools_list_error"] = tools_msg
        if not tools_ok:
            logger.warning(f"[run_debug_episode] task={task_id!r}  tools_list check failed: {tools_msg}")

        env_out = None
        while report["steps"] < max_steps:
            action = agent(obs, task.action_set)

            t_step = time.perf_counter()
            env_out = task.step(action)
            step_time = time.perf_counter() - t_step

            report["step_times_s"].append(round(step_time, 3))
            report["steps"] += 1

            obs = env_out.obs
            if isinstance(env_out.info, dict) and "profiling" in env_out.info:
                report["profiling"].append(env_out.info["profiling"])
            extra = f"error={env_out.error!r}  " if env_out.error else ""
            obs_md = env_out.obs.to_markdown()
            if len(obs_md) > 500:
                obs_md = obs_md[:250] + " ... [truncated] ... " + obs_md[-250:]
            logger.info(
                f"[run_debug_episode] task={task_id!r}  step={report['steps']}  action={action.name}  "
                f"reward={env_out.reward:.3f}  done={env_out.done}  step_time={step_time:.3f}s  {extra}obs={obs_md}"
            )

            if env_out.done:
                break

        if env_out is not None:
            report["done"] = env_out.done
            report["reward"] = env_out.reward

    except Exception as exc:
        logger.exception(f"[run_debug_episode] task={task_id!r}  episode failed: {exc}")
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        task.close()
        try:
            task.close()
            report["close_idempotent_ok"] = True
        except Exception:
            report["close_idempotent_ok"] = False
        report["episode_time_s"] = round(time.perf_counter() - episode_start, 2)
        logger.info(
            f"[run_debug_episode] task={task_id!r}  DONE  reward={report['reward']:.3f}  steps={report['steps']}  "
            f"episode_time={report['episode_time_s']:.1f}s  error={report['error']}"
        )

    return report


def _truncate_leaf_value(val: object, max_len: int) -> str:
    """Short single-line preview for mismatch reporting."""
    if val is None:
        return "null"
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, int | float):
        return repr(val)
    if isinstance(val, str):
        if len(val) <= max_len:
            return val
        return val[:max_len] + f"… (+{len(val) - max_len} chars)"
    if isinstance(val, (dict, list, tuple)):
        s = json.dumps(val, sort_keys=True, default=str)
    else:
        s = repr(val)
    if len(s) <= max_len:
        return s
    return s[:max_len] + f"… (+{len(s) - max_len} chars)"


def _structural_mismatch_lines(
    path: str,
    a: object,
    b: object,
    lines: list[str],
    max_len: int,
) -> None:
    if a == b:
        return
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            p = f"{path}.{k}" if path else str(k)
            if k not in a:
                lines.append(f"{p}\n  first:  <missing>\n  second: {_truncate_leaf_value(b[k], max_len)}")
            elif k not in b:
                lines.append(f"{p}\n  first:  {_truncate_leaf_value(a[k], max_len)}\n  second: <missing>")
            else:
                _structural_mismatch_lines(p, a[k], b[k], lines, max_len)
        return
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        la, lb = list(a), list(b)
        if len(la) != len(lb):
            lp = f"{path}.__len__" if path else "__len__"
            lines.append(f"{lp}\n  first:  len={len(la)}\n  second: len={len(lb)}")
        for i in range(min(len(la), len(lb))):
            p = f"{path}[{i}]" if path else f"[{i}]"
            _structural_mismatch_lines(p, la[i], lb[i], lines, max_len)
        return
    p = path or "<observation>"
    lines.append(f"{p}\n  first:  {_truncate_leaf_value(a, max_len)}\n  second: {_truncate_leaf_value(b, max_len)}")


def _observation_key_path_diff_report(
    dump_a: object,
    dump_b: object,
    *,
    max_value_len: int = 120,
) -> str:
    """Human-readable mismatches: dotted key paths and [i] indices; leaf values truncated."""
    out: list[str] = []
    _structural_mismatch_lines("", dump_a, dump_b, out, max_value_len)
    if not out:
        return ""
    header = "Observation differences\n\n"
    return header + "\n\n".join(out)


def format_observation_diff(obs_a: object, obs_b: object) -> str:
    """Key-path observation diff (same text as reset-repro when first observations differ)."""
    da = obs_a.model_dump() if hasattr(obs_a, "model_dump") else obs_a
    db = obs_b.model_dump() if hasattr(obs_b, "model_dump") else obs_b
    return _observation_key_path_diff_report(da, db)


def check_reset_reproducibility(module: types.ModuleType) -> tuple[bool, str, str]:
    """
    Same seed → identical first observation (stress_test_specs.md).
    Uses first task config only: make() twice, reset() each, compare first obs.
    Calls benchmark.install() and .setup() before get_task_configs(), and
    .close() when done, so the benchmark is in a consistent state.

    Returns:
        (ok, message, diff): ``diff`` lists mismatched key paths with truncated
        leaf values when the two first observations differ; otherwise ``""``.
        Equality uses the full raw payloads. ``message`` is empty when ``ok``.
    """
    bench_fn = getattr(module, "get_debug_benchmark", None)
    if not callable(bench_fn):
        return False, "no get_debug_benchmark", ""
    benchmark = bench_fn()
    try:
        benchmark.install()
        benchmark.setup()
        configs = list(benchmark.get_task_configs())
        if not configs:
            return False, "no debug task configs", ""
        tc = configs[0]
        # Tasks are created and closed sequentially so that tools that manage
        # their own event loops (e.g. Playwright sync API) don't collide.
        t1 = None
        try:
            t1 = tc.make(
                runtime_context=getattr(benchmark, "_runtime_context", None),
                container_backend=getattr(benchmark, "container_backend", None),
            )
            obs1, _ = t1.reset()
            dump1 = obs1.model_dump() if hasattr(obs1, "model_dump") else str(obs1)
        except Exception as e:
            return False, str(e), ""
        finally:
            if t1 is not None:
                try:
                    t1.close()
                except Exception:
                    pass

        t2 = None
        try:
            t2 = tc.make(
                runtime_context=getattr(benchmark, "_runtime_context", None),
                container_backend=getattr(benchmark, "container_backend", None),
            )
            obs2, _ = t2.reset()
            dump2 = obs2.model_dump() if hasattr(obs2, "model_dump") else str(obs2)
        except Exception as e:
            return False, str(e), ""
        finally:
            if t2 is not None:
                try:
                    t2.close()
                except Exception:
                    pass

        ok = dump1 == dump2
        diff_str = "" if ok else format_observation_diff(dump1, dump2)
        if ok:
            return True, "", ""
        return False, RESET_REPRO_OBS_MISMATCH_MSG, diff_str
    finally:
        try:
            benchmark.close()
        except Exception:
            pass


def check_benchmark_metadata(module: types.ModuleType) -> tuple[bool, str]:
    """
    Benchmark has non-empty name and version (stress_test_specs.md).
    Get metadata from the benchmark instance returned by get_debug_benchmark().
    """
    bench_fn = getattr(module, "get_debug_benchmark", None)
    if not callable(bench_fn):
        return False, "no get_debug_benchmark"
    benchmark = bench_fn()
    meta = benchmark.benchmark_metadata
    if meta is None:
        return False, "no benchmark_metadata"
    return True, ""


def aggregate_profiling(episode_reports: list[dict]) -> dict[str, float]:
    """
    Aggregate info["profiling"] from episode reports into mean duration per operation (seconds).

    Accepts two value formats per operation:
    - float: duration in seconds (emitted by Task.step() for "evaluate" and "obs_postprocess")
    - dict:  sub-fields, e.g. "tool_execute" → {"total": ..., "avg_per_action": ..., "n_actions": ...}
             Each sub-field becomes a separate "op_name/sub_key" entry.
    - (start_ts, end_ts) tuple: legacy format from benchmark authors

    Returns a flat dict of mean values across all steps/episodes. Example:
        {
            "step/tool_execute/total":          0.123,   # mean total tool time per step
            "step/tool_execute/avg_per_action": 0.041,   # mean per-action tool time
            "step/tool_execute/n_actions":      3.0,     # mean actions per step
            "step/evaluate":                    0.045,   # mean evaluate() duration
            "step/obs_postprocess":             0.001,   # mean obs_postprocess() duration
        }
    """
    buckets: dict[str, list[float]] = {}

    def _record(key: str, value: float) -> None:
        buckets.setdefault(key, []).append(value)

    for r in episode_reports:
        for step_prof in r.get("profiling") or []:
            if not isinstance(step_prof, dict):
                continue
            for op_name, val in step_prof.items():
                if isinstance(val, dict):
                    for sub_key, sub_val in val.items():
                        try:
                            _record(f"step/{op_name}/{sub_key}", float(sub_val))
                        except (TypeError, ValueError):
                            pass
                elif isinstance(val, float):
                    _record(f"step/{op_name}", val)
                elif isinstance(val, (list, tuple)) and len(val) == 2:
                    # Legacy (start_ts, end_ts) format
                    try:
                        _record(f"step/{op_name}", float(val[1]) - float(val[0]))
                    except (TypeError, ValueError):
                        pass

    return {key: sum(v) / len(v) for key, v in buckets.items() if v}


def run_debug_suite(
    benchmark_name: str,
    module: types.ModuleType,
    *,
    max_steps: int = 20,
    print_json: bool = True,
    workers: int = 1,
    on_episode_start: Callable[[str], None] | None = None,
    on_episode_done: Callable[[dict], None] | None = None,
) -> list[dict]:
    """
    Run all debug tasks for a benchmark and optionally print a JSON report.

    Args:
        benchmark_name:   Label used in the JSON output (e.g. ``"osworld-cube"``).
        module:           A module exposing ``get_debug_benchmark()`` and
                          ``make_debug_agent(task_id)``.
        max_steps:        Safety cap passed to ``run_debug_episode`` (default 20).
        print_json:       If True, print the JSON report to stdout (default True).
        workers:          Number of threads for episode execution (default 1). Values
                          greater than 1 require tasks not to mutate
                          ``benchmark._runtime_context`` after ``setup()``; see module
                          docstring.
        on_episode_start: Optional callback called with ``task_id`` just before each
                          episode starts (sequential mode only, ignored for workers>1).
        on_episode_done:  Optional callback called with the episode report dict just
                          after each episode finishes (sequential mode only).

    Returns:
        List of per-episode report dicts (same schema as ``run_debug_episode``),
        in ``get_task_configs()`` order.

    Raises:
        ValueError: If ``workers < 1``.
    """
    if workers < 1:
        raise ValueError("workers must be >= 1")

    benchmark = None
    results = []
    try:
        # Step 1: create and install the benchmark.
        logger.info(f"[run_debug_suite] benchmark={benchmark_name!r}  calling get_debug_benchmark()")
        benchmark = module.get_debug_benchmark()
        benchmark.install()
        benchmark.setup()

        # Step 2: iterate task configs from the benchmark and run episodes.
        task_configs = list(benchmark.get_task_configs())
        logger.info(
            f"[run_debug_suite] benchmark={benchmark_name!r}  running {len(task_configs)} task(s) "
            f"workers={workers}: {[tc.task_id for tc in task_configs]}"
        )

        def _episode_for_config(tc):
            try:
                task = tc.make(
                    runtime_context=benchmark._runtime_context, container_backend=benchmark.container_backend
                )
            except ImportError as exc:
                raise ImportError(
                    f"{exc}\n\n"
                    f"Hint: '{benchmark_name}' may require an optional tool package that is not installed.\n"
                    f"Check the benchmark's optional extras in its pyproject.toml"
                ) from exc
            return run_debug_episode(task, module.make_debug_agent(tc.task_id), max_steps=max_steps)

        if workers == 1:
            for tc in task_configs:
                if on_episode_start is not None:
                    on_episode_start(tc.task_id)
                results.append(_episode_for_config(tc))
                if on_episode_done is not None:
                    on_episode_done(results[-1])
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_episode_for_config, tc) for tc in task_configs]
                # Call .result() on every future so exceptions in later tasks are not lost
                # when an earlier future raises (list comprehension would stop early).
                for tc, fut in zip(task_configs, futures, strict=True):
                    try:
                        results.append(fut.result())
                    except Exception as exc:
                        logger.exception(
                            "[run_debug_suite] benchmark=%r parallel episode failed task_id=%r",
                            benchmark_name,
                            tc.task_id,
                        )
                        results.append(
                            {
                                "task_id": tc.task_id,
                                "done": False,
                                "reward": 0.0,
                                "steps": 0,
                                "episode_time_s": 0.0,
                                "step_times_s": [],
                                "error": f"{type(exc).__name__}: {exc}",
                                "tools_list_ok": False,
                                "tools_list_error": "",
                                "reset_time_s": 0.0,
                                "close_idempotent_ok": False,
                                "profiling": [],
                            }
                        )
    finally:
        # Step 3: close the benchmark to free resources.
        if benchmark is not None:
            logger.info(f"[run_debug_suite] benchmark={benchmark_name!r}  calling close()")
            benchmark.close()

    if print_json:
        output = {"benchmark": benchmark_name, "debug_episodes": results}
        print(json.dumps(output, indent=2))
    return results


def build_stress_test_report(
    benchmark_name: str,
    results: list[dict],
    compliance_passed: list[str],
    compliance_failed: list[str],
) -> "StressTestReport":
    """Build a stress-test report from suite results and compliance check names."""
    all_step_times: list[float] = []
    reset_times: list[float] = []
    for r in results:
        all_step_times.extend(r.get("step_times_s") or [])
        if "reset_time_s" in r and r["reset_time_s"]:
            reset_times.append(r["reset_time_s"])
    if all_step_times:
        sorted_times = sorted(all_step_times)
        n = len(sorted_times)
        p50 = sorted_times[int(0.50 * (n - 1))] if n else 0.0
        p95 = sorted_times[int(0.95 * (n - 1))] if n else 0.0
        p99 = sorted_times[int(0.99 * (n - 1))] if n else 0.0
    else:
        p50 = p95 = p99 = 0.0
    task_setup_time_s = sum(reset_times) / len(reset_times) if reset_times else None
    episode_times = [r.get("episode_time_s", 0) for r in results if "episode_time_s" in r]
    episode_time_s = sum(episode_times) / len(episode_times) if episode_times else 0.0
    profiling = aggregate_profiling(results)
    return StressTestReport(
        cube_version=__version__,
        benchmark=benchmark_name,
        timestamp=datetime.now(timezone.utc).isoformat(),
        hardware={
            "cpu_count": getattr(platform, "cpu_count", lambda: None)(),
            "python_version": platform.python_version(),
        },
        compliance={"passed": compliance_passed, "failed": compliance_failed},
        performance={
            "task_setup_time_s": task_setup_time_s,
            "step_latency_p50_s": p50,
            "step_latency_p95_s": p95,
            "step_latency_p99_s": p99,
            "episode_time_s": episode_time_s,
            "profiling": profiling if profiling else None,
        },
    )


class StressTestReport:
    """
    Stress-test report (stress_test_specs.md §3.1 MVP output).
    """

    def __init__(
        self,
        cube_version: str,
        benchmark: str,
        timestamp: str,
        hardware: dict,
        compliance: dict,
        performance: dict,
    ):
        self.cube_version = cube_version
        self.benchmark = benchmark
        self.timestamp = timestamp
        self.hardware = hardware
        self.compliance = compliance
        self.performance = performance

    def save(self, path: str) -> None:
        """Write report as JSON (e.g. cube_stress_test_baseline.json)."""
        data = {
            "cube_version": self.cube_version,
            "benchmark": self.benchmark,
            "timestamp": self.timestamp,
            "hardware": self.hardware,
            "compliance": self.compliance,
            "performance": {k: v for k, v in self.performance.items() if v is not None},
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def print_summary(self) -> None:
        """Print a short summary to stdout (optional)."""
        print(f"Compliance: {len(self.compliance['passed'])} passed, {len(self.compliance['failed'])} failed")
        print(f"Performance: task_setup_time_s={self.performance.get('task_setup_time_s')}, ...")


def assert_debug_tasks_reward_one(
    module: types.ModuleType,
    *,
    max_steps: int = 20,
) -> None:
    """
    Assert that every debug task in ``module`` completes with reward == 1.0.

    Delegates to ``run_debug_suite`` (using ``module.__name__`` as the benchmark
    label), then asserts reward == 1.0 for every episode.

    Intended for use in a single catch-all test function::

        def test_debug_tasks():
            import osworld_cube.debug as mod
            from cube.testing import assert_debug_tasks_reward_one
            assert_debug_tasks_reward_one(mod)

    Args:
        module:    A module exposing ``get_debug_benchmark()`` and
                   ``make_debug_agent(task_id)``.
        max_steps: Safety cap passed to ``run_debug_episode`` (default 20).

    Raises:
        AssertionError: If any episode does not complete, errors, or gets
                        reward < 1.0.
    """
    for report in run_debug_suite(module.__name__, module, max_steps=max_steps):
        task_id = report["task_id"]
        assert not report["error"], f"[{task_id}] Episode error: {report['error']}"
        assert report["done"], f"[{task_id}] Episode did not complete: {report}"
        assert report["reward"] == 1.0, f"[{task_id}] Expected reward=1.0, got {report['reward']}: {report}"
