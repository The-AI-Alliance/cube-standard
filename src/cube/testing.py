"""
CUBE testing utilities — framework-level harness for debug episodes.

Public API
----------
run_debug_episode(task, agent, *, max_steps)  →  dict
run_debug_suite(benchmark_name, task_ids, run_fn)  →  list[dict]
"""

from __future__ import annotations

import json
import logging
import time
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
    logger.info("[run_debug_episode] Starting episode for task=%r", task_id)

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
        logger.info("[run_debug_episode] task=%r  calling reset() …", task_id)
        t0 = time.perf_counter()
        obs, info = task.reset()
        reset_time = time.perf_counter() - t0
        logger.info(
            "[run_debug_episode] task=%r  reset done in %.1fs  info=%s",
            task_id, reset_time, info,
        )

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
                "[run_debug_episode] task=%r  step=%d  action=%s  "
                "reward=%.3f  done=%s  step_time=%.3fs",
                task_id,
                report["steps"],
                action.name,
                env_out.reward,
                env_out.done,
                step_time,
            )

            if env_out.done:
                break

        if env_out is not None:
            report["done"] = env_out.done
            report["reward"] = env_out.reward

    except Exception as exc:
        logger.exception("[run_debug_episode] task=%r  episode failed: %s", task_id, exc)
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        task.close()
        report["episode_time_s"] = round(time.perf_counter() - episode_start, 2)
        logger.info(
            "[run_debug_episode] task=%r  DONE  reward=%.3f  steps=%d  "
            "episode_time=%.1fs  error=%s",
            task_id,
            report["reward"],
            report["steps"],
            report["episode_time_s"],
            report["error"],
        )

    return report


def run_debug_suite(
    benchmark_name: str,
    task_ids: list[str],
    run_fn: Callable[[str], dict],
) -> list[dict]:
    """
    Run all debug tasks for a benchmark and print a JSON report.

    This is the shared orchestration layer so each benchmark's ``__main__``
    does not duplicate JSON formatting and benchmark labelling. Task
    construction remains benchmark-specific via ``run_fn``.

    Args:
        benchmark_name: Label used in the JSON output (e.g. ``"osworld-cube"``).
        task_ids:       Ordered list of task IDs to run.
        run_fn:         Callable ``(task_id: str) -> dict`` that constructs the
                        task + agent and delegates to ``run_debug_episode``.

    Returns:
        List of per-episode report dicts (same schema as ``run_debug_episode``).
        The caller is responsible for exit-code handling.
    """
    logger.info(
        "[run_debug_suite] benchmark=%r  running %d task(s): %s",
        benchmark_name, len(task_ids), task_ids,
    )
    results = [run_fn(tid) for tid in task_ids]
    output = {"benchmark": benchmark_name, "debug_episodes": results}
    print(json.dumps(output, indent=2))
    return results
