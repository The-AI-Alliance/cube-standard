"""Tests for cube.testing — run_debug_episode, run_debug_suite, assert_debug_tasks_reward_one."""

from __future__ import annotations

from types import ModuleType
from typing import ClassVar
from unittest.mock import patch

import pytest
from pydantic import PrivateAttr

from cube.benchmark import Benchmark, BenchmarkConfig, BenchmarkMetadata
from cube.container import Container
from cube.core import Action, Observation
from cube.task import STOP_ACTION, Task, TaskConfig, TaskMetadata
from cube.testing import (
    aggregate_profiling,
    assert_debug_tasks_reward_one,
    check_reset_reproducibility,
    format_observation_diff,
    run_debug_episode,
    run_debug_suite,
)
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

    def evaluate(self, obs: Observation | None = None):
        return 1.0, {}

    def close(self):
        self._close_calls += 1
        super().close()


class FailOnResetTask(Task):
    """Task whose reset() always raises."""

    _close_calls: int = PrivateAttr(default=0)

    def reset(self):
        raise RuntimeError("reset failed")

    def evaluate(self, obs: Observation | None = None):
        return 0.0, {}

    def close(self):
        self._close_calls += 1
        super().close()


class DoneTaskConfig(TaskConfig):
    def make(self, runtime_context=None, container_backend=None) -> DoneTask:
        return DoneTask(metadata=self.metadata, tool_config=NoopToolConfig())


class FailTaskConfig(TaskConfig):
    def make(self, runtime_context=None, container_backend=None) -> FailOnResetTask:
        return FailOnResetTask(metadata=self.metadata, tool_config=NoopToolConfig())


class DoneBenchmark(Benchmark):
    """Runtime pair used by the test fixtures; counts setup/close calls."""

    def __init__(self, config, infra=None):
        super().__init__(config, infra=infra)
        self.setup_calls = 0
        self.close_calls = 0

    def _setup(self) -> None:
        self.setup_calls += 1

    def close(self) -> None:
        self.close_calls += 1


class DoneBenchmarkConfig(BenchmarkConfig):
    benchmark_metadata: ClassVar = BenchmarkMetadata(name="test-bench", version="0.1", description="test")
    task_metadata: ClassVar[dict[str, TaskMetadata]] = {
        "t1": TaskMetadata(id="t1"),
        "t2": TaskMetadata(id="t2"),
        "t3": TaskMetadata(id="t3"),
        "alpha": TaskMetadata(id="alpha"),
        "beta": TaskMetadata(id="beta"),
    }
    task_config_class: ClassVar = DoneTaskConfig
    benchmark_class: ClassVar = DoneBenchmark

    _install_count: ClassVar[int] = 0

    @classmethod
    def install(cls) -> None:
        cls._install_count += 1


class FailingDoneBenchmarkConfig(DoneBenchmarkConfig):
    """Mirror of DoneBenchmarkConfig whose tasks fail on reset()."""

    task_config_class: ClassVar = FailTaskConfig


def stop_agent(obs, action_set):
    return Action(name=STOP_ACTION.name, arguments={})


def noop_agent(obs, action_set):
    return Action(name="noop", arguments={})


def _make_module(task_ids=("t1",), *, fail=False) -> tuple[ModuleType, DoneBenchmarkConfig]:
    """Return (module, config). Tasks complete immediately unless fail=True."""
    mod = ModuleType("fake_debug")
    config_cls = FailingDoneBenchmarkConfig if fail else DoneBenchmarkConfig
    config = config_cls().subset_from_list(list(task_ids))

    def get_debug_benchmark():
        return config

    mod.get_debug_benchmark = get_debug_benchmark  # type: ignore[attr-defined]
    mod.make_debug_agent = lambda tid: stop_agent  # type: ignore[attr-defined]
    return mod, config


# ── run_debug_episode — report structure ──────────────────────────────────────


def test_episode_stop_action_completes_with_reward_one():
    task = DoneTask(metadata=TaskMetadata(id="t1"), tool_config=NoopToolConfig())
    report = run_debug_episode(task, stop_agent)
    assert report["task_id"] == "t1"
    assert report["done"] is True
    assert report["reward"] == 1.0
    assert report["steps"] == 1
    assert report["error"] is None
    assert "episode_time_s" in report
    assert len(report["step_times_s"]) == 1
    assert report.get("tools_list_ok") is True
    assert report.get("close_idempotent_ok") is True


# ── run_debug_episode — close() is always called ──────────────────────────────


def test_episode_close_called_on_success():
    task = DoneTask(metadata=TaskMetadata(id="t1"), tool_config=NoopToolConfig())
    run_debug_episode(task, stop_agent)
    # close() is called once normally and once again for close_idempotent check
    assert task._close_calls == 2


def test_episode_close_called_when_reset_raises():
    task = FailOnResetTask(metadata=TaskMetadata(id="t1"), tool_config=NoopToolConfig())
    run_debug_episode(task, stop_agent)
    assert task._close_calls == 2


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
    results = run_debug_suite("bench", mod, print_json=False)
    assert len(results) == 3


def test_suite_reports_contain_task_ids():
    mod, _ = _make_module(task_ids=("alpha", "beta"))
    results = run_debug_suite("bench", mod, print_json=False)
    assert {r["task_id"] for r in results} == {"alpha", "beta"}


def test_suite_workers_preserves_get_task_configs_order():
    mod, _ = _make_module(task_ids=("t1", "t2", "t3"))
    seq = run_debug_suite("bench", mod, print_json=False, workers=1)
    par = run_debug_suite("bench", mod, print_json=False, workers=2)
    assert [r["task_id"] for r in seq] == ["t1", "t2", "t3"]
    assert [r["task_id"] for r in par] == ["t1", "t2", "t3"]


def test_suite_workers_must_be_non_negative():
    mod, _ = _make_module()
    with pytest.raises(ValueError, match="workers must be >= 0"):
        run_debug_suite("bench", mod, print_json=False, workers=-1)


def test_suite_parallel_workers_collects_all_episode_results_when_first_task_raises():
    """Every future must get .result() so a failure in an earlier task does not swallow later ones."""

    class FirstFailsTaskConfig(TaskConfig):
        def make(self, runtime_context=None, container_backend=None):
            if self.task_id == "t1":
                raise RuntimeError("t1 make failed")
            return DoneTask(metadata=self.metadata, tool_config=NoopToolConfig())

    class FirstFailsBenchmarkConfig(DoneBenchmarkConfig):
        task_config_class: ClassVar = FirstFailsTaskConfig

    mod = ModuleType("fake_debug")
    config = FirstFailsBenchmarkConfig().subset_from_list(["t1", "t2"])
    mod.get_debug_benchmark = lambda: config  # type: ignore[attr-defined]
    mod.make_debug_agent = lambda tid: stop_agent  # type: ignore[attr-defined]

    results = run_debug_suite("bench", mod, print_json=False, workers=2)
    assert len(results) == 2
    assert results[0]["task_id"] == "t1"
    assert results[0]["error"] is not None and "t1 make failed" in results[0]["error"]
    assert results[1]["task_id"] == "t2"
    assert results[1].get("error") in (None, "")


# ── run_debug_suite — benchmark lifecycle ────────────────────────────────────


def test_suite_calls_install_then_make_then_close():
    """install() runs once; make() returns a live Benchmark; close() is called in finally."""
    captured: list[DoneBenchmark] = []

    class CapturingBenchmark(DoneBenchmark):
        def _setup(self) -> None:
            super()._setup()
            captured.append(self)

    class CapturingConfig(DoneBenchmarkConfig):
        benchmark_class: ClassVar = CapturingBenchmark
        _install_count: ClassVar[int] = 0  # shadow parent so test is isolated

        @classmethod
        def install(cls) -> None:
            cls._install_count += 1

    mod = ModuleType("fake_debug")
    mod.get_debug_benchmark = lambda: CapturingConfig().subset_from_list(["t1"])  # type: ignore[attr-defined]
    mod.make_debug_agent = lambda tid: stop_agent  # type: ignore[attr-defined]

    run_debug_suite("bench", mod, print_json=False)

    assert CapturingConfig._install_count == 1
    assert len(captured) == 1
    bench = captured[0]
    assert isinstance(bench, DoneBenchmark)
    assert bench.setup_calls == 1
    assert bench.close_calls == 1


def test_suite_benchmark_closed_even_when_get_task_configs_raises():
    captured: list["FailingTaskConfigsBenchmark"] = []

    class FailingTaskConfigsBenchmark(Benchmark):
        def __init__(self, config, infra=None):
            super().__init__(config, infra=infra)
            self.close_calls = 0

        def _setup(self) -> None:
            captured.append(self)

        def close(self) -> None:
            self.close_calls += 1

    class FailingTaskConfigsConfig(BenchmarkConfig):
        benchmark_metadata: ClassVar = BenchmarkMetadata(name="fail-bench", version="0.1", description="test")
        task_metadata: ClassVar[dict[str, TaskMetadata]] = {"t1": TaskMetadata(id="t1")}
        task_config_class: ClassVar = DoneTaskConfig
        benchmark_class: ClassVar = FailingTaskConfigsBenchmark

        def get_task_configs(self):
            raise RuntimeError("config error")

    mod = ModuleType("fake_debug")
    mod.get_debug_benchmark = lambda: FailingTaskConfigsConfig()  # type: ignore[attr-defined]
    mod.make_debug_agent = lambda tid: stop_agent  # type: ignore[attr-defined]

    with pytest.raises(RuntimeError, match="config error"):
        run_debug_suite("bench", mod, print_json=False)
    assert len(captured) == 1
    assert captured[0].close_calls == 1


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


# ── aggregate_profiling ───────────────────────────────────────────────────────


def test_aggregate_profiling_float_values():
    reports = [{"profiling": [{"evaluate": 0.04, "obs_postprocess": 0.01}]}]
    assert aggregate_profiling(reports) == {"step/evaluate": 0.04, "step/obs_postprocess": 0.01}


def test_aggregate_profiling_dict_values():
    reports = [{"profiling": [{"tool_execute": {"total": 0.12, "avg_per_action": 0.06, "n_actions": 2}}]}]
    assert aggregate_profiling(reports) == {
        "step/tool_execute/total": 0.12,
        "step/tool_execute/avg_per_action": 0.06,
        "step/tool_execute/n_actions": 2.0,
    }


def test_aggregate_profiling_legacy_tuple_values():
    reports = [{"profiling": [{"container_exec": (1000.0, 1000.05)}]}]
    assert aggregate_profiling(reports) == {"step/container_exec": pytest.approx(0.05)}


def test_aggregate_profiling_averages_across_steps():
    reports = [{"profiling": [{"evaluate": 0.02}, {"evaluate": 0.06}]}]
    assert aggregate_profiling(reports) == {"step/evaluate": pytest.approx(0.04)}


def test_aggregate_profiling_averages_across_episodes():
    reports = [
        {"profiling": [{"evaluate": 0.02}]},
        {"profiling": [{"evaluate": 0.06}]},
    ]
    assert aggregate_profiling(reports) == {"step/evaluate": pytest.approx(0.04)}


def test_aggregate_profiling_empty_returns_empty():
    assert aggregate_profiling([]) == {}
    assert aggregate_profiling([{"profiling": []}]) == {}


def test_aggregate_profiling_populated_in_episode_report():
    task = DoneTask(metadata=TaskMetadata(id="t1"), tool_config=NoopToolConfig())
    report = run_debug_episode(task, noop_agent, max_steps=2)
    assert len(report["profiling"]) == 2
    for step_prof in report["profiling"]:
        assert set(step_prof.keys()) == {"tool_execute", "obs_postprocess"}
        assert set(step_prof["tool_execute"].keys()) == {"total", "avg_per_action", "n_actions"}
        assert step_prof["tool_execute"]["n_actions"] == 1


# ── check_reset_reproducibility ───────────────────────────────────────────────


class _FakeObs:
    def __init__(self, payload: dict):
        self._payload = payload

    def model_dump(self):
        return dict(self._payload)


class _FakeTaskForReset:
    def __init__(self, payload: dict):
        self._payload = payload

    def reset(self):
        return _FakeObs(self._payload), {}

    def close(self):
        pass


class _FakeTaskConfig:
    def __init__(self):
        self._makes = 0

    def make(self, **kwargs):
        self._makes += 1
        payload = {"token": self._makes}
        return _FakeTaskForReset(payload)


class _FakeBenchmark:
    def __init__(self):
        self._runtime_context: dict = {}

    def close(self):
        pass


class _FakeBenchmarkConfig:
    def __init__(self):
        self._tc = _FakeTaskConfig()
        self.container_backend = None

    def install(self):
        pass

    def make(self, infra=None):
        return _FakeBenchmark()

    def get_task_configs(self):
        return [self._tc]


def test_check_reset_reproducibility_returns_unified_diff_when_obs_differ():
    mod = ModuleType("fake_reset")
    mod.get_debug_benchmark = lambda: _FakeBenchmarkConfig()

    ok, msg, diff = check_reset_reproducibility(mod)
    assert ok is False
    assert "differed" in msg
    assert "Observation differences" in diff
    assert "token" in diff
    assert "first:" in diff and "second:" in diff


def test_check_reset_reproducibility_ok_and_empty_diff_when_matching():
    class _SameTC:
        def make(self, **kwargs):
            return _FakeTaskForReset({"x": 1})

    class _SameBenchmark:
        def __init__(self):
            self._runtime_context: dict = {}

        def close(self):
            pass

    class _SameBenchmarkConfig:
        def __init__(self):
            self.container_backend = None

        def install(self):
            pass

        def make(self, infra=None):
            return _SameBenchmark()

        def get_task_configs(self):
            return [_SameTC()]

    mod = ModuleType("fake_reset_ok")
    mod.get_debug_benchmark = lambda: _SameBenchmarkConfig()

    ok, msg, diff = check_reset_reproducibility(mod)
    assert ok is True
    assert msg == ""
    assert diff == ""


def test_check_reset_reproducibility_errors_return_empty_diff():
    mod = ModuleType("no_bench")
    ok, msg, diff = check_reset_reproducibility(mod)
    assert ok is False
    assert "get_debug_benchmark" in msg
    assert diff == ""


def test_format_observation_diff_key_paths_and_truncates_leaves():
    a = {"hint": "ok", "html": "<div>" + "x" * 500 + "</div>"}
    b = {"hint": "ok", "html": "<div>" + "y" * 500 + "</div>"}
    diff = format_observation_diff(a, b)
    assert "html" in diff
    assert "Observation differences" in diff
    assert len(diff) < 900
    assert "x" * 200 not in diff


def test_format_observation_diff_truncates_long_data_urls():
    a = {"screenshot": "data:image/png;base64," + "A" * 300}
    b = {"screenshot": "data:image/png;base64," + "B" * 300}
    diff = format_observation_diff(a, b)
    assert "screenshot" in diff
    assert len(diff) < 700
