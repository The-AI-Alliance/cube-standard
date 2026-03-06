"""Tests for counter-cube.

Each test exercises one distinct interaction pattern with the CUBE stack.
Read them alongside tool.py, task.py, and benchmark.py.
"""

import pytest

from cube.core import Action, Observation
from counter_cube import CounterBenchmark, CounterEnvironment, CounterEnvironmentConfig, CounterTaskConfig
from counter_cube.pluggable_actions import CounterEnvironmentPluggable, CounterEnvironmentPluggableConfig

INCREMENT = Action(name="increment", arguments={})
DECREMENT = Action(name="decrement", arguments={})


@pytest.fixture
def benchmark():
    bench = CounterBenchmark()
    bench.setup()
    yield bench
    bench.close()


@pytest.fixture
def task_configs(benchmark):
    return {c.task_id: c for c in benchmark.get_task_configs()}


# ---------------------------------------------------------------------------
# Environment-level tests — exercise CounterEnvironment and add_environment_action directly
# ---------------------------------------------------------------------------


def test_default_action_set():
    """Default environment has only increment; decrement and increment_by are absent."""
    env = CounterEnvironmentConfig().make()
    names = [a.name for a in env.action_set]
    assert "increment" in names
    assert "decrement" not in names
    assert "increment_by" not in names


def test_enable_decrement():
    """enable_decrement=True adds decrement to the action set."""
    env = CounterEnvironmentConfig(enable_decrement=True).make()
    names = [a.name for a in env.action_set]
    assert "decrement" in names


def test_disable_decrement():
    """enable_decrement=False keeps decrement out of the action set."""
    env = CounterEnvironmentConfig(enable_decrement=False).make()
    names = [a.name for a in env.action_set]
    assert "decrement" not in names


def test_enable_increment_by():
    """enable_increment_by=True adds increment_by to the action set."""
    env = CounterEnvironmentConfig(enable_increment_by=True).make()
    names = [a.name for a in env.action_set]
    assert "increment_by" in names


def test_add_environment_action_custom():
    """add_environment_action attaches a user-supplied function as a discoverable action."""
    env = CounterEnvironmentPluggable(CounterEnvironmentPluggableConfig())

    def reset_counter(e) -> str:
        """Reset counter to zero."""
        e.counter = 0
        return "Counter reset"

    env.add_environment_action(reset_counter)

    names = [a.name for a in env.action_set]
    assert "reset_counter" in names

    # Execute it and verify the counter changes
    env.counter = 5
    result = env.execute_action(Action(name="reset_counter", arguments={}))
    assert isinstance(result, Observation)
    assert env.counter == 0


def test_environment_reset():
    """reset() brings counter back to 0."""
    env = CounterEnvironmentConfig().make()
    env.execute_action(INCREMENT)
    env.execute_action(INCREMENT)
    assert env.counter == 2
    env.reset()
    assert env.counter == 0


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
    assert isinstance(task.env, CounterEnvironment)
    assert task.env.counter == 3
    task.close()


def test_multi_step_batch(task_configs):
    """task.step([...]) executes a batch; observations are concatenated."""
    task = task_configs["count-to-3"].make()
    task.reset()

    env_out = task.step([INCREMENT, INCREMENT, INCREMENT])
    assert len(env_out.obs.contents) == 3
    assert task.env.counter == 3
    task.close()


def test_task_isolation(task_configs):
    """Two tasks from the same config have independent environment state."""
    task_a = task_configs["count-to-3"].make()
    task_b = task_configs["count-to-3"].make()
    task_a.reset()
    task_b.reset()

    task_a.step(INCREMENT)
    task_a.step(INCREMENT)

    assert task_a.env.counter == 2
    assert task_b.env.counter == 0
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
# EnvironmentConfig-per-task tests
# ---------------------------------------------------------------------------


def test_decrement_task(task_configs):
    """count-to-3-with-decrement exposes decrement in the action set."""
    task = task_configs["count-to-3-with-decrement"].make()
    task.reset()

    names = [a.name for a in task.action_set]
    assert "decrement" in names

    task.step(INCREMENT)
    assert task.env.counter == 1
    task.step(DECREMENT)
    assert task.env.counter == 0
    task.close()


def test_increment_by_task(task_configs):
    """count-by-2 exposes increment_by; two calls with value=2 reach target 4."""
    task = task_configs["count-by-2"].make()
    task.reset()

    names = [a.name for a in task.action_set]
    assert "increment_by" in names

    task.step(Action(name="increment_by", arguments={"value": 2}))
    assert task.env.counter == 2

    env_out = task.step(Action(name="increment_by", arguments={"value": 2}))
    assert task.env.counter == 4
    assert env_out.done
    task.close()


def test_env_config_override():
    """Explicit env_config on TaskConfig takes precedence over metadata defaults."""
    cfg = CounterTaskConfig(
        task_id="count-to-3",
        env_config=CounterEnvironmentConfig(enable_increment_by=True),
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
    """env.execute_action() works without going through the task."""
    task = task_configs["count-to-3"].make()
    task.reset()

    result = task.env.execute_action(INCREMENT)
    assert isinstance(result, Observation)
    assert task.env.counter == 1
    task.close()


def test_unknown_action_raises(task_configs):
    """Executing an unknown action name raises ValueError."""
    task = task_configs["count-to-3"].make()
    task.reset()

    with pytest.raises(ValueError, match="nonexistent"):
        task.env.execute_action(Action(name="nonexistent", arguments={}))
    task.close()
