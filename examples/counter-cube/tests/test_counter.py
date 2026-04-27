"""Tests for counter-cube.

Each test exercises one distinct interaction pattern with the CUBE stack.
Read them alongside tool.py, task.py, and benchmark.py.
"""

import pytest

from cube.core import Action, Observation
from counter_cube import CounterBenchmarkConfig, CounterTaskConfig, CounterTool, CounterToolConfig
from counter_cube.pluggable_tool import CounterToolPluggable
from counter_cube.task import CounterTaskMetadata

INCREMENT = Action(name="increment", arguments={})
DECREMENT = Action(name="decrement", arguments={})


@pytest.fixture
def benchmark():
    with CounterBenchmarkConfig().make() as bench:
        yield bench


@pytest.fixture
def task_configs(benchmark):
    return {c.task_id: c for c in benchmark.config.get_task_configs()}


# ---------------------------------------------------------------------------
# Tool-level tests — exercise CounterTool and add_tool_action directly
# ---------------------------------------------------------------------------


def test_default_action_set():
    """Default tool has only increment; decrement and increment_by are absent."""
    tool = CounterToolConfig().make()
    names = [a.name for a in tool.action_set]
    assert "increment" in names
    assert "decrement" not in names
    assert "increment_by" not in names


def test_enable_decrement():
    """enable_decrement=True adds decrement to the action set."""
    tool = CounterToolConfig(enable_decrement=True).make()
    names = [a.name for a in tool.action_set]
    assert "decrement" in names


def test_disable_decrement():
    """enable_decrement=False keeps decrement out of the action set."""
    tool = CounterToolConfig(enable_decrement=False).make()
    names = [a.name for a in tool.action_set]
    assert "decrement" not in names


def test_enable_increment_by():
    """enable_increment_by=True adds increment_by to the action set."""
    tool = CounterToolConfig(enable_increment_by=True).make()
    names = [a.name for a in tool.action_set]
    assert "increment_by" in names


def test_add_tool_action_custom():
    """add_tool_action attaches a user-supplied function as a discoverable action."""
    tool = CounterToolPluggable(CounterToolConfig())

    def reset_counter(env) -> str:
        """Reset counter to zero."""
        env.counter = 0
        return "Counter reset"

    tool.add_tool_action(reset_counter)

    names = [a.name for a in tool.action_set]
    assert "reset_counter" in names

    # Execute it and verify the counter changes
    tool._env.counter = 5
    result = tool.execute_action(Action(name="reset_counter", arguments={}))
    assert isinstance(result, Observation)
    assert tool._env.counter == 0


def test_tool_reset():
    """reset() brings counter back to 0."""
    tool = CounterToolConfig().make()
    tool.execute_action(INCREMENT)
    tool.execute_action(INCREMENT)
    assert tool._env.counter == 2
    tool.reset()
    assert tool._env.counter == 0


# ---------------------------------------------------------------------------
# Task-level tests — exercise the full episode loop
# ---------------------------------------------------------------------------


def test_single_step(task_configs):
    """task.step() drives the counter and returns EnvironmentOutput."""
    task = task_configs["count-to-3"].make()
    obs, info = task.reset()

    assert info["target"] == 3
    assert "increment" in obs.contents[0].data

    for _ in range(3):
        env_out = task.step(INCREMENT)

    assert env_out.done
    assert env_out.reward == 1.0
    assert isinstance(task.tool, CounterTool)
    assert task.tool._env.counter == 3
    task.close()


def test_multi_step_batch(task_configs):
    """task.step([...]) executes a batch; observations are concatenated."""
    task = task_configs["count-to-3"].make()
    task.reset()

    env_out = task.step([INCREMENT, INCREMENT, INCREMENT])
    assert len(env_out.obs.contents) == 3
    assert task.tool._env.counter == 3
    task.close()


def test_task_isolation(task_configs):
    """Two tasks from the same config have independent tool state."""
    task_a = task_configs["count-to-3"].make()
    task_b = task_configs["count-to-3"].make()
    task_a.reset()
    task_b.reset()

    task_a.step(INCREMENT)
    task_a.step(INCREMENT)

    assert task_a.tool._env.counter == 2
    assert task_b.tool._env.counter == 0
    task_a.close()
    task_b.close()


def test_partial_reward(task_configs):
    """evaluate() returns partial reward (< 1.0) when target not yet reached."""
    task = task_configs["count-to-3"].make()
    obs, _ = task.reset()

    task.step(INCREMENT)  # counter = 1, target = 3
    reward, info = task.evaluate(obs)

    assert reward < 1.0
    assert not info["solved"]
    task.close()


# ---------------------------------------------------------------------------
# ToolConfig-per-task tests
# ---------------------------------------------------------------------------


def test_decrement_task(task_configs):
    """count-to-3-with-decrement exposes decrement in the action set."""
    task = task_configs["count-to-3-with-decrement"].make()
    task.reset()

    names = [a.name for a in task.action_set]
    assert "decrement" in names

    task.step(INCREMENT)  # 0 → 1
    assert task.tool._env.counter == 1
    task.close()


def test_increment_by_task(task_configs):
    """count-by-2 exposes increment_by; two calls with value=2 reach target 4."""
    task = task_configs["count-by-2"].make()
    task.reset()

    names = [a.name for a in task.action_set]
    assert "increment_by" in names

    task.step(Action(name="increment_by", arguments={"value": 2}))
    assert task.tool._env.counter == 2

    env_out = task.step(Action(name="increment_by", arguments={"value": 2}))
    assert task.tool._env.counter == 4
    assert env_out.done
    task.close()


def test_toolconfig_override():
    """Explicit tool_config on TaskConfig takes precedence over metadata defaults."""
    cfg = CounterTaskConfig(
        task_id="count-to-3",
        metadata=CounterTaskMetadata(id="count-to-3", target=3),
        tool_config=CounterToolConfig(enable_increment_by=True),
    )
    task = cfg.make()
    task.reset()

    names = [a.name for a in task.action_set]
    assert "increment_by" in names
    task.close()


# ---------------------------------------------------------------------------
# Low-level API
# ---------------------------------------------------------------------------


def test_execute_action_directly(task_configs):
    """tool.execute_action() works without going through the task."""
    task = task_configs["count-to-3"].make()
    task.reset()

    result = task.tool.execute_action(INCREMENT)
    assert isinstance(result, Observation)
    assert task.tool._env.counter == 1
    task.close()


def test_unknown_action_raises(task_configs):
    """Executing an unknown action name raises ValueError."""
    task = task_configs["count-to-3"].make()
    task.reset()

    with pytest.raises(ValueError, match="nonexistent"):
        task.tool.execute_action(Action(name="nonexistent", arguments={}))
    task.close()
