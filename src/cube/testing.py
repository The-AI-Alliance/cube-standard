"""
CUBE testing utilities — framework-level harness for debug episodes.

Public API
----------
run_debug_episode(task, agent, *, max_steps)  →  dict
run_debug_suite(benchmark_name, module, *, max_steps)  →  list[dict]
assert_debug_tasks_reward_one(module, *, max_steps)  →  None
run_debug_agent(benchmark, infra)  →  dict

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
"""

from __future__ import annotations

import json
import logging
import platform
import time
import types
from collections.abc import Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cube.benchmark import Benchmark
    from cube.resource import InfraConfig, ResourceConfig

from cube import __version__  # report.cube_version and .save()
from cube.core import Action, ActionSchema, Observation
from cube.task import Task

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
    Same seed → identical first observation (stress_test_specs.md).
    Uses first task config only: make() twice, reset() each, compare first obs.
    Calls benchmark.install() and .setup() before get_task_configs(), and
    .close() when done, so the benchmark is in a consistent state.
    """
    bench_fn = getattr(module, "get_debug_benchmark", None)
    if not callable(bench_fn):
        return False, "no get_debug_benchmark"
    benchmark = bench_fn()
    try:
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
            dump1 = obs1.model_dump() if hasattr(obs1, "model_dump") else str(obs1)
            dump2 = obs2.model_dump() if hasattr(obs2, "model_dump") else str(obs2)
            ok = dump1 == dump2
        except Exception as e:
            return False, str(e)
        finally:
            for t in (t1, t2):
                if t is not None:
                    try:
                        t.close()
                    except Exception:
                        pass
        return ok, "" if ok else "first observation differed between two resets"
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


def run_debug_agent(
    benchmark: "Benchmark",
    infra: "InfraConfig",
    *,
    run_id: str | None = None,
) -> dict:
    """Pre-flight smoke test for a (benchmark, infra) pair.

    Mirrors the three-step check from resource_lifecycle.md §5:

    1. **Provision check** — queries the ProvisionStore for every resource in
       ``benchmark.list_resources()``. Reports missing registrations with
       actionable instructions and aborts if any are missing.

    2. **Capability check** — verifies ``infra.can_serve(resource)`` for every
       resource. Fails fast with a clear message if requirements are not met
       (e.g. "requires kvm but infra does not support it").

    3. **Launch check** — calls ``infra.launch()`` on the first task-scoped
       resource. Probes the HTTP endpoint (/screenshot + /execute uname -a) to
       confirm the guest agent is reachable. Tears down the resource on exit.

    A clean run guarantees that the infra is ready for a full evaluation run.
    Reuses ``run_debug_suite`` for the functional episode check when the benchmark
    exposes a debug module (``get_debug_benchmark`` + ``make_debug_agent``).

    Args:
        benchmark:  An instantiated Benchmark (not yet setup).
        infra:      An InfraConfig to test against.
        run_id:     Optional run identifier; generated if not provided.

    Returns:
        dict with keys: resources_checked, provision_ok, capabilities_ok,
        launch_ok, endpoint, error.
    """
    import uuid

    from cube.resource import InfraConfig, ResourceConfig

    run_id = run_id or str(uuid.uuid4())
    report: dict = {
        "run_id": run_id,
        "resources_checked": 0,
        "provision_ok": False,
        "capabilities_ok": False,
        "launch_ok": False,
        "endpoint": None,
        "error": None,
    }

    resources: list[ResourceConfig] = benchmark.list_resources()

    if not resources:
        logger.warning(
            "[run_debug_agent] %r defines no resources — nothing to check. "
            "Override list_resources() to declare dependencies.",
            benchmark.name,
        )
        report["provision_ok"] = True
        report["capabilities_ok"] = True
        report["launch_ok"] = True
        return report

    report["resources_checked"] = len(resources)

    # ── Step 1: provision check ───────────────────────────────────────────────

    missing = [r for r in resources if infra.provision_status(r) != "ready"]
    if missing:
        names = ", ".join(repr(r.name) for r in missing)
        msg = (
            f"[run_debug_agent] {len(missing)} resource(s) not registered for "
            f"{infra.fingerprint()!r}: {names}.\n"
            f"  Run: infra.register(resource, {{...}})   # manual\n"
            f"  Or:  infra.provision(resource)          # automated, if supported"
        )
        logger.error(msg)
        report["error"] = msg
        return report

    report["provision_ok"] = True
    logger.info(
        "[run_debug_agent] provision check passed (%d resource(s))", len(resources)
    )

    # ── Step 2: capability check ──────────────────────────────────────────────

    incompatible = [r for r in resources if not infra.can_serve(r)]
    if incompatible:
        details = "; ".join(
            f"{r.name!r} requires {r.requirements()} but infra provides {infra.capabilities()}"
            for r in incompatible
        )
        msg = f"[run_debug_agent] capability mismatch: {details}"
        logger.error(msg)
        report["error"] = msg
        return report

    report["capabilities_ok"] = True
    logger.info("[run_debug_agent] capability check passed")

    # ── Step 3: launch check ──────────────────────────────────────────────────
    # Use the first task-scoped resource for the smoke test.

    task_resources = [r for r in resources if r.scope == "task"]
    if not task_resources:
        logger.info(
            "[run_debug_agent] no task-scoped resources to launch — skipping launch check"
        )
        report["launch_ok"] = True
        return report

    probe_resource = task_resources[0]
    logger.info(
        "[run_debug_agent] launching %r on %r for smoke test …",
        probe_resource.name,
        infra.fingerprint(),
    )

    handle = None
    try:
        handle = infra.launch(probe_resource, run_id=run_id)
        report["endpoint"] = handle.endpoint
        logger.info("[run_debug_agent] endpoint: %s", handle.endpoint)

        if handle.endpoint:
            _probe_endpoint(handle.endpoint)

        report["launch_ok"] = True
        logger.info("[run_debug_agent] launch check passed — infra is ready")

    except Exception as exc:
        msg = f"[run_debug_agent] launch check failed: {type(exc).__name__}: {exc}"
        logger.error(msg)
        report["error"] = msg
    finally:
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass

    return report


def _probe_endpoint(endpoint: str, timeout: int = 120) -> None:
    """Hit /screenshot and /execute on a CUBE guest agent endpoint.

    Reuses the same probe logic as the experiment backends (_common.probe).
    Raises RuntimeError if the endpoint is not responsive within timeout seconds.
    """
    import time

    try:
        import requests as _requests
    except ImportError:
        logger.warning("[run_debug_agent] 'requests' not installed — skipping HTTP probe")
        return

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = _requests.get(f"{endpoint}/screenshot", timeout=5)
            if r.status_code == 200 and len(r.content) > 0:
                logger.info(
                    "[run_debug_agent] /screenshot → HTTP 200, %d bytes", len(r.content)
                )
                r2 = _requests.post(
                    f"{endpoint}/execute",
                    json={"command": ["uname", "-a"]},
                    timeout=10,
                )
                stdout = r2.json().get("stdout", "").strip()
                logger.info("[run_debug_agent] /execute → %s", stdout)
                return
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError(f"Guest agent at {endpoint} not responsive after {timeout}s")


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
