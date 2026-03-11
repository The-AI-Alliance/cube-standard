"""Tests for cube.testing — run_debug_episode, run_debug_suite, assert_debug_tasks_reward_one."""

from types import ModuleType
from unittest.mock import patch

import pytest
from pydantic import PrivateAttr

from cube.benchmark import RuntimeContext  # noqa: F401 – required for Task Pydantic resolution
from cube.container import Container
from cube.core import Action, Observation
from cube.task import STOP_ACTION, Task, TaskConfig, TaskMetadata
from cube.testing import assert_debug_tasks_reward_one, run_debug_episode, run_debug_suite
from cube.tool import Tool, ToolConfig, tool_action

# ── Shared test infrastructure ────────────────────────────────────────────────


class NoopTool(Tool):
    @tool_action
    def noop(self) -> str:
        """Do nothing."""
        return "ok"


class NoopToolConfig(ToolConfig):
    def make(self, container: Container | None = None) -> NoopTool:
        return NoopTool()


class DoneTask(Task):
    """Task that evaluates to reward=1.0."""

    _close_calls: int = PrivateAttr(default=0)

    def reset(self):
        return Observation.from_text("ready"), {}

    def evaluate(self, obs: Observation):
        return 1.0, {}

    def close(self):
        self._close_calls += 1
        super().close()


class FailOnResetTask(Task):
    """Task whose reset() always raises."""

    _close_calls: int = PrivateAttr(default=0)

    def reset(self):
        raise RuntimeError("reset failed")

    def evaluate(self, obs: Observation):
        return 0.0, {}

    def close(self):
        self._close_calls += 1
        super().close()


class DoneTaskConfig(TaskConfig):
    def make(self, runtime_context=None, container_backend=None) -> DoneTask:
        return DoneTask(metadata=TaskMetadata(id=self.task_id), tool_config=NoopToolConfig())


class FailTaskConfig(TaskConfig):
    def make(self, runtime_context=None, container_backend=None) -> FailOnResetTask:
        return FailOnResetTask(metadata=TaskMetadata(id=self.task_id), tool_config=NoopToolConfig())


def stop_agent(obs, action_set):
    return Action(name=STOP_ACTION.name, arguments={})


def noop_agent(obs, action_set):
    return Action(name="noop", arguments={})


def _make_module(task_ids=("t1",), *, with_setup=False, with_teardown=False, fail=False):
    """Return (module, call_log). Tasks complete immediately unless fail=True."""
    mod = ModuleType("fake_debug")
    call_log: list[str] = []

    config_cls = FailTaskConfig if fail else DoneTaskConfig
    mod.get_debug_task_configs = lambda: [config_cls(task_id=tid) for tid in task_ids]
    mod.make_debug_agent = lambda tid: stop_agent

    if with_setup:

        def _setup():
            call_log.append("setup")

        mod.setup_debug_suite = _setup

    if with_teardown:

        def _teardown():
            call_log.append("teardown")

        mod.teardown_debug_suite = _teardown

    return mod, call_log


# ── run_debug_episode — report structure ──────────────────────────────────────


def test_episode_stop_action_completes_with_reward_one():
    task = DoneTask(metadata=TaskMetadata(id="t1"), tool_config=NoopToolConfig())
    report = run_debug_episode(task, stop_agent)
    assert report == {
        "task_id": "t1",
        "done": True,
        "reward": 1.0,
        "steps": 1,
        "episode_time_s": report["episode_time_s"],
        "step_times_s": report["step_times_s"],
        "error": None,
    }


# ── run_debug_episode — close() is always called ──────────────────────────────


def test_episode_close_called_on_success():
    task = DoneTask(metadata=TaskMetadata(id="t1"), tool_config=NoopToolConfig())
    run_debug_episode(task, stop_agent)
    assert task._close_calls == 1


def test_episode_close_called_when_reset_raises():
    task = FailOnResetTask(metadata=TaskMetadata(id="t1"), tool_config=NoopToolConfig())
    run_debug_episode(task, stop_agent)
    assert task._close_calls == 1


# ── run_debug_episode — error handling ────────────────────────────────────────


def test_episode_error_set_when_reset_raises():
    task = FailOnResetTask(metadata=TaskMetadata(id="t1"), tool_config=NoopToolConfig())
    report = run_debug_episode(task, stop_agent)
    assert "RuntimeError" in report["error"]
    assert "reset failed" in report["error"]


def test_episode_done_false_when_reset_raises():
    task = FailOnResetTask(metadata=TaskMetadata(id="t1"), tool_config=NoopToolConfig())
    report = run_debug_episode(task, stop_agent)
    assert report["done"] is False
    assert report["reward"] == 0.0


# ── run_debug_episode — max_steps ─────────────────────────────────────────────


def test_episode_respects_max_steps():
    task = DoneTask(metadata=TaskMetadata(id="t1"), tool_config=NoopToolConfig())
    report = run_debug_episode(task, noop_agent, max_steps=3)
    assert report["steps"] == 3
    assert report["done"] is False
    assert len(report["step_times_s"]) == 3


# ── run_debug_suite — basic behaviour ────────────────────────────────────────


def test_suite_returns_one_report_per_task():
    mod, _ = _make_module(task_ids=("t1", "t2", "t3"))
    results = run_debug_suite("bench", mod)
    assert len(results) == 3


def test_suite_reports_contain_task_ids():
    mod, _ = _make_module(task_ids=("alpha", "beta"))
    results = run_debug_suite("bench", mod)
    assert {r["task_id"] for r in results} == {"alpha", "beta"}


# ── run_debug_suite — setup / teardown hooks ─────────────────────────────────


def test_suite_no_error_without_hooks():
    mod, calls = _make_module()
    run_debug_suite("bench", mod)
    assert calls == []


def test_suite_both_hooks_called_in_order():
    mod, calls = _make_module(with_setup=True, with_teardown=True)
    run_debug_suite("bench", mod)
    assert calls == ["setup", "teardown"]


def test_suite_teardown_called_even_when_get_configs_raises():
    mod, calls = _make_module(with_teardown=True)
    mod.get_debug_task_configs = lambda: (_ for _ in ()).throw(RuntimeError("config error"))
    with pytest.raises(RuntimeError, match="config error"):
        run_debug_suite("bench", mod)
    assert "teardown" in calls


# ── assert_debug_tasks_reward_one ────────────────────────────────────────────


def test_assert_passes_when_all_tasks_succeed():
    mod, _ = _make_module(task_ids=("t1", "t2"))
    assert_debug_tasks_reward_one(mod)  # should not raise


def test_assert_raises_when_error_present():
    mod, _ = _make_module(fail=True)
    with pytest.raises(AssertionError, match="Episode error"):
        assert_debug_tasks_reward_one(mod)


def test_assert_raises_when_reward_less_than_one():
    mod, _ = _make_module()
    report = {
        "task_id": "t1",
        "done": True,
        "reward": 0.5,
        "steps": 1,
        "episode_time_s": 0.0,
        "step_times_s": [],
        "error": None,
    }
    with patch("cube.testing.run_debug_suite", return_value=[report]):
        with pytest.raises(AssertionError, match="reward=1.0"):
            assert_debug_tasks_reward_one(mod)


def test_assert_raises_when_not_done():
    mod, _ = _make_module()
    report = {
        "task_id": "t1",
        "done": False,
        "reward": 1.0,
        "steps": 1,
        "episode_time_s": 0.0,
        "step_times_s": [],
        "error": None,
    }
    with patch("cube.testing.run_debug_suite", return_value=[report]):
        with pytest.raises(AssertionError, match="did not complete"):
            assert_debug_tasks_reward_one(mod)
