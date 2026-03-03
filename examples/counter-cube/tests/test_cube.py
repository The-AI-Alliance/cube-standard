from cube.testing import run_debug_episode, assert_debug_tasks_reward_one
import counter_cube.debug as debug_mod


def test_all_debug_tasks_reward_one():
    """All debug tasks complete with reward == 1.0 using the deterministic agent."""
    assert_debug_tasks_reward_one(debug_mod)


def test_run_debug_episode_report_schema():
    """run_debug_episode returns a dict with the expected keys and correct values."""
    task_configs = {tc.task_id: tc for tc in debug_mod.get_debug_task_configs()}
    task = task_configs["count-to-3"].make()
    agent = debug_mod.make_debug_agent("count-to-3")

    report = run_debug_episode(task, agent, max_steps=20)

    assert set(report) == {"task_id", "done", "reward", "steps", "episode_time_s", "step_times_s", "error"}
    assert report["task_id"] == "count-to-3"
    assert report["done"] is True
    assert report["reward"] == 1.0
    assert report["steps"] == 3
    assert report["error"] is None
    assert len(report["step_times_s"]) == report["steps"]


def test_run_debug_episode_max_steps_cap():
    """max_steps caps the episode before completion."""
    from cube.core import Action, ActionSchema, Observation
    from counter_cube.task import CounterTaskConfig

    # Build a task with a high target so the agent can't finish in 2 steps
    cfg = CounterTaskConfig(task_id="count-to-3")
    task = cfg.make()

    def slow_agent(obs: Observation, action_set: list[ActionSchema]) -> Action:
        return Action(name="increment", arguments={})

    report = run_debug_episode(task, slow_agent, max_steps=2)

    assert report["steps"] == 2
    assert report["done"] is False
    assert report["error"] is None
