"""Deterministic debug agent for testing CounterBenchmark end-to-end without an LLM.

Each debug task has a hardcoded action sequence that completes it successfully.
Used to validate the CUBE task loop in CI or local development.

Public API
----------
make_debug_agent(task_id)       → DebugAgent
get_debug_task_configs()        → list[CounterTaskConfig]

Usage::

    # Run all debug tasks and print a JSON report
    python -m counter_cube.debug_agent
"""

from __future__ import annotations

import logging

from cube.core import Action, ActionSchema, Observation
from counter_cube.benchmark import CounterBenchmark
from counter_cube.task import CounterTaskConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardcoded action sequences per task ID
#
# Each sequence must drive the task to done=True, reward=1.0.
# These also serve as a minimal sanity check that the task logic is correct.
# ---------------------------------------------------------------------------

_TASK_ACTIONS: dict[str, list[Action]] = {
    "count-to-3": [
        Action(name="increment", arguments={}),
        Action(name="increment", arguments={}),
        Action(name="increment", arguments={}),
    ],
    "count-to-3-with-decrement": [
        Action(name="increment", arguments={}),
        Action(name="increment", arguments={}),
        Action(name="decrement", arguments={}),  # go back to 1 to show decrement works
        Action(name="increment", arguments={}),
        Action(name="increment", arguments={}),
        Action(name="increment", arguments={}),
    ],
    "count-by-2": [
        Action(name="increment_by", arguments={"value": 2}),  # 0 → 2
        Action(name="increment_by", arguments={"value": 2}),  # 2 → 4 (done)
    ],
}


# ---------------------------------------------------------------------------
# DebugAgent
# ---------------------------------------------------------------------------


class DebugAgent:
    """Deterministic debug agent that replays a fixed action sequence.

    Interface matches cube.testing (stress_test_specs.md §1.2):
        agent = make_debug_agent(task_id)
        action = agent(obs, action_set)   # __call__ shorthand

    Args:
        task_id: Must match a key in _TASK_ACTIONS.

    Raises:
        ValueError: If task_id has no registered action sequence.
    """

    def __init__(self, task_id: str) -> None:
        if task_id not in _TASK_ACTIONS:
            raise ValueError(f"No debug actions registered for task {task_id!r}. Known tasks: {list(_TASK_ACTIONS)}")
        self._task_id = task_id
        self._step = 0
        self._actions = list(_TASK_ACTIONS[task_id])
        logger.debug("[DebugAgent] task=%r  %d actions loaded", task_id, len(self._actions))

    def get_action(self, obs: Observation) -> Action:
        """Return the next predetermined action."""
        if self._step >= len(self._actions):
            raise StopIteration(f"[DebugAgent] task={self._task_id!r}: all {len(self._actions)} actions exhausted")
        action = self._actions[self._step]
        logger.info(
            "[DebugAgent] task=%r  step=%d/%d  action=%s",
            self._task_id,
            self._step + 1,
            len(self._actions),
            action.name,
        )
        self._step += 1
        return action

    def __call__(self, obs: Observation, action_set: list[ActionSchema]) -> Action:
        """Callable shorthand — delegates to get_action() for task-loop compatibility."""
        return self.get_action(obs)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def make_debug_agent(task_id: str) -> DebugAgent:
    """Return a fresh DebugAgent for the given task_id."""
    return DebugAgent(task_id)


def get_debug_task_configs() -> list[CounterTaskConfig]:
    """Return CounterTaskConfig objects for all registered debug tasks."""
    task_metadata = CounterBenchmark.task_metadata
    return [CounterTaskConfig(task_id=tid) for tid in _TASK_ACTIONS if tid in task_metadata]


# ---------------------------------------------------------------------------
# __main__ — run all debug tasks, print JSON report
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import counter_cube.debug as _mod
    from cube.testing import run_debug_suite

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    results = run_debug_suite("counter-cube", _mod)

    failed = [r for r in results if r["error"] or not r["done"] or r["reward"] <= 0]
    sys.exit(1 if failed else 0)
