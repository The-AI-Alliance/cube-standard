"""
CUBE testing utilities — framework-level harness for debug episodes.

Public API
----------
run_debug_episode(task, agent, *, max_steps)  →  dict
run_debug_suite(benchmark_name, module, *, max_steps)  →  list[dict]
assert_debug_tasks_reward_one(module, *, max_steps)  →  None

Module protocol (for assert_debug_tasks_reward_one)
----------------------------------------------------------------------
The ``module`` argument must expose two callables:

    get_debug_task_configs() -> list[TaskConfig]
        Return one TaskConfig per debug task. Each config must have a
        ``.task_id`` attribute and a ``.make()`` method that returns a Task.

    make_debug_agent(task_id: str) -> Callable[[Observation, list[ActionSchema]], Action]
        Return a deterministic agent for the given task_id.

Optionally, the module may also expose:

    setup_debug_suite() -> None
        Called once before any debug episodes run (e.g. start a local server).

    teardown_debug_suite() -> None
        Called once after all debug episodes finish, even if they fail.

Example usage in a test file::

    def test_debug_tasks():
        from cube.testing import assert_debug_tasks_reward_one
        import osworld_cube.debug_agent as _mod
        assert_debug_tasks_reward_one(_mod)
"""

from __future__ import annotations

import json
import logging
import time
import types
from collections.abc import Callable

from cube.core import Action, ActionSchema, Observation
from cube.task import Task

logger = logging.getLogger(__name__)


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
    }

    episode_start = time.perf_counter()
    try:
        logger.info(f"[run_debug_episode] task={task_id!r}  calling reset() …")
        t0 = time.perf_counter()
        obs, info = task.reset()
        reset_time = time.perf_counter() - t0
        logger.info(f"[run_debug_episode] task={task_id!r}  reset done in {reset_time:.1f}s  info={info}")

        env_out = None
        while report["steps"] < max_steps:
            action = agent(obs, task.action_set)

            t_step = time.perf_counter()
            env_out = task.step(action)
            step_time = time.perf_counter() - t_step

            report["step_times_s"].append(round(step_time, 3))
            report["steps"] += 1

            obs = env_out.obs
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
        report["episode_time_s"] = round(time.perf_counter() - episode_start, 2)
        logger.info(
            f"[run_debug_episode] task={task_id!r}  DONE  reward={report['reward']:.3f}  steps={report['steps']}  "
            f"episode_time={report['episode_time_s']:.1f}s  error={report['error']}"
        )

    return report


def run_debug_suite(
    benchmark_name: str,
    module: types.ModuleType,
    *,
    max_steps: int = 20,
) -> list[dict]:
    """
    Run all debug tasks for a benchmark and print a JSON report.

    Args:
        benchmark_name: Label used in the JSON output (e.g. ``"osworld-cube"``).
        module:         A module exposing ``get_debug_task_configs()`` and
                        ``make_debug_agent(task_id)``.
        max_steps:      Safety cap passed to ``run_debug_episode`` (default 20).

    Returns:
        List of per-episode report dicts (same schema as ``run_debug_episode``).
        The caller is responsible for exit-code handling.
    """
    setup = getattr(module, "setup_debug_suite", None)
    teardown = getattr(module, "teardown_debug_suite", None)

    if setup is not None:
        logger.info(f"[run_debug_suite] benchmark={benchmark_name!r}  calling setup_debug_suite()")
        setup()

    try:
        task_configs = {tc.task_id: tc for tc in module.get_debug_task_configs()}
        logger.info(
            f"[run_debug_suite] benchmark={benchmark_name!r}  running {len(task_configs)} task(s): {list(task_configs)}"
        )
        results = []
        for tid, tc in task_configs.items():
            try:
                task = tc.make()
            except ImportError as exc:
                raise ImportError(
                    f"{exc}\n\n"
                    f"Hint: '{benchmark_name}' may require an optional tool package that is not installed.\n"
                    f"Check the benchmark's optional extras in its pyproject.toml"
                ) from exc
            results.append(run_debug_episode(task, module.make_debug_agent(tid), max_steps=max_steps))
    finally:
        if teardown is not None:
            logger.info(f"[run_debug_suite] benchmark={benchmark_name!r}  calling teardown_debug_suite()")
            teardown()

    output = {"benchmark": benchmark_name, "debug_episodes": results}
    print(json.dumps(output, indent=2))
    return results


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
            import osworld_cube.debug_agent as mod
            from cube.testing import assert_debug_tasks_reward_one
            assert_debug_tasks_reward_one(mod)

    Args:
        module:    A module exposing ``get_debug_task_configs()`` and
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
