"""
CUBE testing utilities — framework-level harness for debug episodes.

Public API
----------
run_stress_test(module, *, benchmark_name, max_steps, print_json)  →  StressTestOutcome
run_debug_episode(task, agent, *, max_steps)  →  dict
run_debug_suite(benchmark_name, module, *, max_steps)  →  list[dict]
assert_debug_tasks_reward_one(module, *, max_steps)  →  None
collect_stress_compliance(results, module)  →  (passed_names, failed_names)

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

Example usage in a test file::

    def test_debug_tasks():
        from cube.testing import assert_debug_tasks_reward_one
        import osworld_cube.debug as _mod
        assert_debug_tasks_reward_one(_mod)

Stress test report (stress_test_specs.md)::

    import counter_cube.debug as mod
    from cube.testing import run_stress_test

    outcome = run_stress_test(mod)
    outcome.print_summary()
    outcome.save("cube_stress_test_baseline.json")
"""

from __future__ import annotations

import json
import logging
import platform
import time
import types
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher

from PIL import Image, ImageChops

from cube import __version__  # report.cube_version and .save()
from cube.core import (
    Action,
    ActionSchema,
    AudioContent,
    Content,
    ImageContent,
    Observation,
    StructuredContent,
    TextContent,
    VideoContent,
)
from cube.task import Task

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResetReproducibilityConfig:
    """
    Tolerance for ``check_reset_reproducibility`` when byte-identical observations
    are unrealistic (timestamps, AXTree ids, screenshot clocks).

    Attach to a benchmark instance as ``benchmark.reset_reproducibility_config`` to override defaults.
    """

    text_similarity_min: float = 0.97
    """Minimum ``difflib.SequenceMatcher`` ratio for text and JSON-serialised structured data."""

    image_max_mae: float = 0.12
    """Mean absolute error between resized grayscale pixels, normalised to ``[0, 1]`` (divide by 255)."""

    image_compare_size: tuple[int, int] = (64, 64)
    """Both images are resized to this (width, height) before MAE."""


def _text_similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _structured_to_json(data: object) -> str:
    return json.dumps(data, sort_keys=True, default=str, separators=(",", ":"))


def _image_mae_normalized(a: Image.Image, b: Image.Image, size: tuple[int, int]) -> float:
    ga = a.convert("L").resize(size, Image.Resampling.BILINEAR)
    gb = b.convert("L").resize(size, Image.Resampling.BILINEAR)
    diff = ImageChops.difference(ga, gb)
    hist = diff.histogram()
    n = sum(hist)
    if n == 0:
        return 0.0
    mean_abs = sum(i * c for i, c in enumerate(hist)) / n
    return mean_abs / 255.0


def _contents_equivalent_for_reset(
    c1: Content,
    c2: Content,
    cfg: ResetReproducibilityConfig,
) -> tuple[bool, str]:
    if type(c1) is not type(c2):
        return False, f"type mismatch: {type(c1).__name__} vs {type(c2).__name__}"
    if c1.tool_call_id != c2.tool_call_id or c1.name != c2.name:
        return False, "content metadata (tool_call_id / name) differs"

    if isinstance(c1, TextContent) and isinstance(c2, TextContent):
        r = _text_similarity(c1.data, c2.data)
        if r >= cfg.text_similarity_min:
            return True, ""
        return False, f"text similarity {r:.4f} < {cfg.text_similarity_min}"

    if isinstance(c1, StructuredContent) and isinstance(c2, StructuredContent):
        j1, j2 = _structured_to_json(c1.data), _structured_to_json(c2.data)
        r = _text_similarity(j1, j2)
        if r >= cfg.text_similarity_min:
            return True, ""
        return False, f"structured JSON similarity {r:.4f} < {cfg.text_similarity_min}"

    if isinstance(c1, ImageContent) and isinstance(c2, ImageContent):
        mae = _image_mae_normalized(c1.data, c2.data, cfg.image_compare_size)
        if mae <= cfg.image_max_mae:
            return True, ""
        return False, f"image MAE {mae:.4f} > {cfg.image_max_mae}"

    if isinstance(c1, AudioContent) and isinstance(c2, AudioContent):
        if c1.data == c2.data and c1.duration_seconds == c2.duration_seconds:
            return True, ""
        return False, "audio content differs"

    if isinstance(c1, VideoContent) and isinstance(c2, VideoContent):
        if c1.data == c2.data and c1.duration_seconds == c2.duration_seconds:
            return True, ""
        return False, "video content differs"

    d1, d2 = c1.model_dump(), c2.model_dump()
    if d1 == d2:
        return True, ""
    return False, f"content type {type(c1).__name__} not similar (strict dump mismatch)"


def _observations_equivalent_for_reset(
    obs1: Observation,
    obs2: Observation,
    cfg: ResetReproducibilityConfig,
) -> tuple[bool, str]:
    a, b = obs1.contents, obs2.contents
    if len(a) != len(b):
        return False, f"content count differs: {len(a)} vs {len(b)}"
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        ok, msg = _contents_equivalent_for_reset(x, y, cfg)
        if not ok:
            return False, f"contents[{i}]: {msg}"
    return True, ""


def _resolve_reset_reproducibility_config(benchmark: object) -> ResetReproducibilityConfig:
    raw = getattr(benchmark, "reset_reproducibility_config", None)
    if raw is None:
        return ResetReproducibilityConfig()
    if isinstance(raw, ResetReproducibilityConfig):
        return raw
    raise TypeError(
        f"reset_reproducibility_config must be None or ResetReproducibilityConfig, got {type(raw).__name__}"
    )


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
        logger.info(f"[run_debug_episode] task={task_id!r}  reset done in {reset_time:.1f}s  info={info}")

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
            logger.info(
                f"[run_debug_episode] task={task_id!r}  step={report['steps']}  action={action.name}  "
                f"reward={env_out.reward:.3f}  done={env_out.done}  step_time={step_time:.3f}s"
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


def check_reset_reproducibility(module: types.ModuleType) -> tuple[bool, str]:
    """
    Same seed → equivalent first observation after reset (stress_test_specs.md).

    Uses first task config only: ``make()`` twice, ``reset()`` each, then compare
    first observations with modality-aware tolerance (text / structured JSON fuzzy
    match, downscaled image MAE). Byte-identical ``model_dump()`` is not required.

    Optional: set ``benchmark.reset_reproducibility_config`` to a
    :class:`ResetReproducibilityConfig` instance to tune thresholds per benchmark.

    Calls ``benchmark.install()`` and ``.setup()`` before ``get_task_configs()``,
    and ``.close()`` when done.
    """
    bench_fn = getattr(module, "get_debug_benchmark", None)
    if not callable(bench_fn):
        return False, "no get_debug_benchmark"
    benchmark = bench_fn()
    try:
        try:
            cfg = _resolve_reset_reproducibility_config(benchmark)
        except TypeError as e:
            return False, str(e)
        benchmark.install()
        benchmark.setup()
        configs = list(benchmark.get_task_configs())
        if not configs:
            return False, "no debug task configs"
        tc = configs[0]
        t1 = t2 = None
        try:
            t1 = tc.make(
                runtime_context=getattr(benchmark, "_runtime_context", None),
                container_backend=getattr(benchmark, "container_backend", None),
            )
            t2 = tc.make(
                runtime_context=getattr(benchmark, "_runtime_context", None),
                container_backend=getattr(benchmark, "container_backend", None),
            )
            obs1, _ = t1.reset()
            obs2, _ = t2.reset()
            if isinstance(obs1, Observation) and isinstance(obs2, Observation):
                ok, detail = _observations_equivalent_for_reset(obs1, obs2, cfg)
            else:

                def _fallback_str(o: object) -> str:
                    if isinstance(o, str):
                        return o
                    if hasattr(o, "model_dump"):
                        return json.dumps(o.model_dump(), sort_keys=True, default=str)
                    return str(o)

                fs1, fs2 = _fallback_str(obs1), _fallback_str(obs2)
                if fs1 == fs2:
                    ok, detail = True, ""
                else:
                    r = _text_similarity(fs1, fs2)
                    ok = r >= cfg.text_similarity_min
                    detail = "" if ok else f"fallback string similarity {r:.4f} < {cfg.text_similarity_min}"
            msg = "" if ok else (detail or "first observation differed between two resets")
        except Exception as e:
            return False, str(e)
        finally:
            for t in (t1, t2):
                if t is not None:
                    try:
                        t.close()
                    except Exception:
                        pass
        return ok, msg
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


def collect_stress_compliance(
    results: list[dict],
    module: types.ModuleType,
) -> tuple[list[str], list[str]]:
    """
    Derive stress_test_specs.md compliance pass/fail lists from episode reports and module checks.
    """
    failures = [r for r in results if r.get("error") or not r.get("done") or r.get("reward") != 1.0]
    reset_ok, _ = check_reset_reproducibility(module)
    meta_ok, _ = check_benchmark_metadata(module)
    close_idempotent_ok = all(r.get("close_idempotent_ok", False) for r in results)
    tools_list_ok = all(r.get("tools_list_ok", False) for r in results)
    compliance_passed: list[str] = []
    compliance_failed: list[str] = []
    if results:
        compliance_passed.append("test_debug_tasks_exist")
        compliance_passed.append("test_debug_agent_exists")
    if not failures:
        compliance_passed.append("test_full_episode")
    else:
        compliance_failed.append("test_full_episode")
    if reset_ok:
        compliance_passed.append("test_reset_reproducibility")
    else:
        compliance_failed.append("test_reset_reproducibility")
    if tools_list_ok:
        compliance_passed.append("test_tools_list")
    else:
        compliance_failed.append("test_tools_list")
    if close_idempotent_ok:
        compliance_passed.append("test_close_idempotent")
    else:
        compliance_failed.append("test_close_idempotent")
    if meta_ok:
        compliance_passed.append("test_benchmark_metadata")
    else:
        compliance_failed.append("test_benchmark_metadata")
    return compliance_passed, compliance_failed


def aggregate_profiling(episode_reports: list[dict]) -> dict[str, float]:
    """
    Aggregate info["profiling"] from episode reports into mean duration per operation (seconds).
    Each step's profiling is { "op_name": (start_ts, end_ts), ... }.
    """
    durations: dict[str, list[float]] = {}
    for r in episode_reports:
        for step_prof in r.get("profiling") or []:
            if not isinstance(step_prof, dict):
                continue
            for op_name, val in step_prof.items():
                if isinstance(val, (list, tuple)) and len(val) >= 2:
                    try:
                        dur = float(val[1]) - float(val[0])
                        durations.setdefault(op_name, []).append(dur)
                    except (TypeError, ValueError):
                        pass
    return {op: sum(v) / len(v) for op, v in durations.items() if v}


def run_debug_suite(
    benchmark_name: str,
    module: types.ModuleType,
    *,
    max_steps: int = 20,
    print_json: bool = True,
) -> list[dict]:
    """
    Run all debug tasks for a benchmark and optionally print a JSON report.

    Args:
        benchmark_name: Label used in the JSON output (e.g. ``"osworld-cube"``).
        module:         A module exposing ``get_debug_benchmark()`` and
                        ``make_debug_agent(task_id)``.
        max_steps:      Safety cap passed to ``run_debug_episode`` (default 20).
        print_json:     If True, print the JSON report to stdout (default True).

    Returns:
        List of per-episode report dicts (same schema as ``run_debug_episode``).
        The caller is responsible for exit-code handling.
    """
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
            f"[run_debug_suite] benchmark={benchmark_name!r}  running {len(task_configs)} task(s): "
            f"{[tc.task_id for tc in task_configs]}"
        )
        for tc in task_configs:
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
            results.append(run_debug_episode(task, module.make_debug_agent(tc.task_id), max_steps=max_steps))
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


@dataclass
class StressTestOutcome:
    """
    Result of run_stress_test: StressTestReport plus episode rows for inspection.
    save() / print_summary() forward to .report so stress_test_specs.md examples work on the outcome object.
    """

    report: StressTestReport
    episodes: list[dict]
    failed_episodes: list[dict]

    def print_summary(self) -> None:
        self.report.print_summary()

    def save(self, path: str) -> None:
        self.report.save(path)


def _ensure_debug_module_protocol(module: types.ModuleType) -> None:
    # Required surface for any code path that runs the debug suite + compliance.
    for attr in ("get_debug_benchmark", "make_debug_agent"):
        if not callable(getattr(module, attr, None)):
            raise TypeError(f"debug module must expose callable {attr}(); see cube_package/debug.py template.")


def run_stress_test(
    module: types.ModuleType,
    *,
    benchmark_name: str | None = None,
    max_steps: int = 20,
    print_json: bool = False,
) -> StressTestOutcome:
    """
    Single entry point for the CUBE stress test (stress_test_specs.md §3.1).

    Runs all debug episodes, applies compliance checks, and builds ``StressTestReport``.

    Args:
        module: Imported debug module with ``get_debug_benchmark()`` and ``make_debug_agent(task_id)``.
        benchmark_name: Label stored in the report (default: ``module.__name__``).
        max_steps: Safety cap per episode.
        print_json: If True, print debug suite JSON to stdout (same as ``run_debug_suite``).

    Returns:
        StressTestOutcome: call ``.save(path)`` / ``.print_summary()`` or read ``.report`` / ``.episodes``.

    Raises:
        TypeError: If the module does not expose the required callables.
        RuntimeError: If no debug tasks ran.
    """
    _ensure_debug_module_protocol(module)
    label = benchmark_name if benchmark_name is not None else module.__name__
    results = run_debug_suite(
        label,
        module,
        max_steps=max_steps,
        print_json=print_json,
    )
    if not results:
        raise RuntimeError(
            "No debug episodes ran; ensure get_debug_benchmark().get_task_configs() yields at least one task."
        )
    compliance_passed, compliance_failed = collect_stress_compliance(results, module)
    report = build_stress_test_report(label, results, compliance_passed, compliance_failed)
    failures = [r for r in results if r.get("error") or not r.get("done") or r.get("reward") != 1.0]
    return StressTestOutcome(report=report, episodes=list(results), failed_episodes=failures)


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
