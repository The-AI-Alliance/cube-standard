"""
CUBE testing utilities — framework-level harness for debug episodes.

Public API
----------
run_debug_episode(task, agent, *, max_steps)  →  dict
run_debug_suite(benchmark_name, module, *, max_steps, workers=0, infra=None)  →  list[dict]
assert_debug_tasks_reward_one(module, *, infra=None, max_steps)  →  None

Module protocol (for ``assert_debug_tasks_reward_one`` and ``run_debug_suite``)
-------------------------------------------------------------------------------
The ``module`` argument must expose two callables:

    get_debug_benchmark() -> BenchmarkConfig
        Return a BenchmarkConfig (optionally pre-filtered to the debug subset
        via ``subset_from_list``). The harness calls ``config.install()`` then
        ``config.make(infra)`` to obtain a live ``Benchmark`` ready to spawn
        tasks, and ``benchmark.close()`` at the end to free resources.

    make_debug_agent(task_id: str) -> Callable[[Observation, list[ActionSchema]], Action]
        Return a deterministic agent for the given task_id.

    Parallel runs (``workers > 1``, or ``workers=0`` for auto): tasks share the benchmark's
    ``_runtime_context`` by reference. After ``bench_config.make()`` returns, concurrent episodes
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from cube import __version__  # report.cube_version and .save()
from cube.core import Action, ActionSchema, Observation
from cube.resource import InfraConfig
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


def check_reset_reproducibility(module: types.ModuleType, *, infra: InfraConfig | None = None) -> tuple[bool, str, str]:
    """
    Same seed → identical first observation (stress_test_specs.md).
    Uses first task config only: make() twice, reset() each, compare first obs.
    Calls ``config.install()`` and ``config.make(infra)`` before iterating
    ``get_task_configs``, and ``benchmark.close()`` when done.

    The two Task instances are created in separate threads so tools with their own
    event loops (e.g. Playwright sync API) don't collide — each thread owns its
    own event loop.

    Returns:
        (ok, message, diff): ``diff`` lists mismatched key paths with truncated
        leaf values when the two first observations differ; otherwise ``""``.
        ``message`` is empty when ``ok``.
    """
    bench_fn = getattr(module, "get_debug_benchmark", None)
    if not callable(bench_fn):
        return False, "no get_debug_benchmark", ""
    config = bench_fn()
    config.install()  # classmethod on BenchmarkConfig; accessing via instance is idiomatic
    benchmark = config.make(infra)
    try:
        configs = list(config.get_task_configs())
        if not configs:
            return False, "no debug task configs", ""
        tc = configs[0]
        runtime_context = getattr(benchmark, "_runtime_context", None)

        def _reset_once() -> object:
            t = tc.make(runtime_context=runtime_context)
            try:
                obs, _ = t.reset()
                return obs.model_dump() if hasattr(obs, "model_dump") else str(obs)
            finally:
                try:
                    t.close()
                except Exception:
                    pass

        with ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(_reset_once)
            f2 = pool.submit(_reset_once)
            try:
                dump1 = f1.result()
            except Exception as e:
                return False, str(e), ""
            try:
                dump2 = f2.result()
            except Exception as e:
                return False, str(e), ""

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
    Gets metadata from the BenchmarkConfig returned by ``get_debug_benchmark()``.
    """
    bench_fn = getattr(module, "get_debug_benchmark", None)
    if not callable(bench_fn):
        return False, "no get_debug_benchmark"
    config = bench_fn()
    meta = config.benchmark_metadata
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
    workers: int = 0,
    infra: InfraConfig | None = None,
    on_episode_start: Callable[[str], None] | None = None,
    on_episode_done: Callable[[dict], None] | None = None,
) -> list[dict]:
    """
    Run all debug tasks for a benchmark and optionally print a JSON report.

    Args:
        benchmark_name:   Label used in the JSON output (e.g. ``"osworld-cube"``).
        module:           A module exposing ``get_debug_benchmark() -> BenchmarkConfig``
                          and ``make_debug_agent(task_id)``.
        max_steps:        Safety cap passed to ``run_debug_episode`` (default 20).
        print_json:       If True, print the JSON report to stdout (default True).
        workers:          Number of threads for episode execution. ``0`` (default)
                          means auto: one thread per task (all tasks in parallel).
                          ``1`` runs sequentially. Values > 1 require tasks not to
                          mutate ``benchmark._runtime_context`` after ``setup()``;
                          see module docstring.
        infra:            Optional ``InfraConfig`` passed to ``config.make(infra)`` for
                          resource provisioning.
        on_episode_start: Optional callback called with ``task_id`` just before each
                          episode starts. In parallel mode all start callbacks fire
                          upfront before any episode finishes.
        on_episode_done:  Optional callback called with the episode report dict just
                          after each episode finishes (fires in completion order for
                          parallel runs).

    Returns:
        List of per-episode report dicts (same schema as ``run_debug_episode``),
        in ``config.get_task_configs()`` order.

    Raises:
        ValueError: If ``workers < 0``.
    """
    if workers < 0:
        raise ValueError("workers must be >= 0")

    benchmark = None
    results = []
    try:
        # Step 1: resolve the BenchmarkConfig and bring a live Benchmark into existence.
        logger.info(f"[run_debug_suite] benchmark={benchmark_name!r}  calling get_debug_benchmark()")
        bench_fn = module.get_debug_benchmark
        config = bench_fn()
        config.install()  # classmethod on BenchmarkConfig; accessing via instance is idiomatic
        benchmark = config.make(infra)

        # Step 2: iterate task configs from the benchmark config and run episodes.
        task_configs = list(config.get_task_configs())
        effective_workers = len(task_configs) if workers == 0 else workers

        logger.info(
            f"[run_debug_suite] benchmark={benchmark_name!r}  running {len(task_configs)} task(s) "
            f"workers={effective_workers}: {[tc.task_id for tc in task_configs]}"
        )

        def _episode_for_config(tc):
            try:
                task = tc.make(runtime_context=benchmark._runtime_context)
            except ImportError as exc:
                raise ImportError(
                    f"{exc}\n\n"
                    f"Hint: '{benchmark_name}' may require an optional tool package that is not installed.\n"
                    f"Check the benchmark's optional extras in its pyproject.toml"
                ) from exc
            return run_debug_episode(task, module.make_debug_agent(tc.task_id), max_steps=max_steps)

        def _make_error_report(tc, exc: Exception) -> dict:
            return {
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

        if effective_workers <= 1:
            for tc in task_configs:
                if on_episode_start is not None:
                    on_episode_start(tc.task_id)
                try:
                    report = _episode_for_config(tc)
                except Exception as exc:
                    logger.exception(
                        "[run_debug_suite] benchmark=%r serial episode failed task_id=%r",
                        benchmark_name,
                        tc.task_id,
                    )
                    report = _make_error_report(tc, exc)
                results.append(report)
                if on_episode_done is not None:
                    on_episode_done(report)
        else:
            # Snapshot _runtime_context key→value-identity before parallel run.
            # Any mutation (new key, deleted key, or reassigned value) during
            # episode execution is a thread-safety bug — warn immediately.
            _ctx_before = {k: id(v) for k, v in benchmark._runtime_context.items()}

            # Fire start callbacks upfront — all tasks launch simultaneously.
            if on_episode_start is not None:
                for tc in task_configs:
                    on_episode_start(tc.task_id)
            results = [None] * len(task_configs)
            tc_index = {id(tc): i for i, tc in enumerate(task_configs)}
            with ThreadPoolExecutor(max_workers=effective_workers) as pool:
                future_to_tc = {pool.submit(_episode_for_config, tc): tc for tc in task_configs}
                for fut in as_completed(future_to_tc):
                    tc = future_to_tc[fut]
                    idx = tc_index[id(tc)]
                    try:
                        report = fut.result()
                    except Exception as exc:
                        logger.exception(
                            "[run_debug_suite] benchmark=%r parallel episode failed task_id=%r",
                            benchmark_name,
                            tc.task_id,
                        )
                        report = _make_error_report(tc, exc)
                    results[idx] = report
                    if on_episode_done is not None:
                        on_episode_done(report)

            _ctx_after = {k: id(v) for k, v in benchmark._runtime_context.items()}
            if _ctx_before != _ctx_after:
                added = set(_ctx_after) - set(_ctx_before)
                removed = set(_ctx_before) - set(_ctx_after)
                mutated = {k for k in _ctx_before if k in _ctx_after and _ctx_before[k] != _ctx_after[k]}
                logger.warning(
                    "[run_debug_suite] benchmark=%r _runtime_context mutated during parallel episodes "
                    "(added=%r removed=%r reassigned=%r) — likely thread-safety bug",
                    benchmark_name,
                    added,
                    removed,
                    mutated,
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
    infra: InfraConfig | None = None,
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
        module:    A module exposing ``get_debug_benchmark() -> BenchmarkConfig``
                   and ``make_debug_agent(task_id)``.
        infra:     Optional ``InfraConfig`` passed to ``config.make(infra)``.
        max_steps: Safety cap passed to ``run_debug_episode`` (default 20).

    Raises:
        AssertionError: If any episode does not complete, errors, or gets
                        reward < 1.0.

    Note:
        Tasks run in parallel by default (``workers=0``). The benchmark's
        ``_runtime_context`` is shared across threads — treat it as read-only
        after ``setup()`` returns.
    """
    for report in run_debug_suite(module.__name__, module, max_steps=max_steps, infra=infra):
        task_id = report["task_id"]
        assert not report["error"], f"[{task_id}] Episode error: {report['error']}"
        assert report["done"], f"[{task_id}] Episode did not complete: {report}"
        assert report["reward"] == 1.0, f"[{task_id}] Expected reward=1.0, got {report['reward']}: {report}"
