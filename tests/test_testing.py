"""Tests for cube.testing — run_debug_episode, run_debug_suite, assert_debug_tasks_reward_one."""

import json
from types import ModuleType
from unittest.mock import patch

import pytest
from PIL import Image as PILImage
from pydantic import PrivateAttr

from cube.benchmark import Benchmark, BenchmarkMetadata, RuntimeContext  # noqa: F401
from cube.container import Container
from cube.core import Action, ImageContent, Observation, TextContent
from cube.task import STOP_ACTION, Task, TaskConfig, TaskMetadata
from cube.testing import (
    ResetReproducibilityConfig,
    assert_debug_tasks_reward_one,
    check_reset_reproducibility,
    collect_stress_compliance,
    run_debug_episode,
    run_debug_suite,
    run_stress_test,
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


class DoneBenchmark(Benchmark):
    benchmark_metadata = BenchmarkMetadata(name="test-bench", version="0.1", description="test")
    task_metadata = {}
    task_config_class = DoneTaskConfig

    _install_calls: int = PrivateAttr(default=0)
    _setup_calls: int = PrivateAttr(default=0)
    _close_calls: int = PrivateAttr(default=0)

    def install(self) -> None:
        self._install_calls += 1

    def _setup(self) -> None:
        self._setup_calls += 1

    def close(self) -> None:
        self._close_calls += 1
        super().close()


def stop_agent(obs, action_set):
    return Action(name=STOP_ACTION.name, arguments={})


def noop_agent(obs, action_set):
    return Action(name="noop", arguments={})


def _make_module(task_ids=("t1",), *, fail=False):
    """Return (module, benchmark). Tasks complete immediately unless fail=True."""
    mod = ModuleType("fake_debug")

    benchmark = DoneBenchmark()
    config_cls = FailTaskConfig if fail else DoneTaskConfig
    task_meta = {tid: TaskMetadata(id=tid) for tid in task_ids}
    object.__setattr__(benchmark, "task_metadata", task_meta)
    object.__setattr__(benchmark, "task_config_class", config_cls)
    mod.get_debug_benchmark = lambda: benchmark  # type: ignore[attr-defined]
    mod.make_debug_agent = lambda tid: stop_agent  # type: ignore[attr-defined]

    return mod, benchmark


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
    # Stress-test report extras (tools_list, close_idempotent)
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
    # close() is called once in finally and once for close_idempotent check
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
    results = run_debug_suite("bench", mod)
    assert len(results) == 3


def test_suite_reports_contain_task_ids():
    mod, _ = _make_module(task_ids=("alpha", "beta"))
    results = run_debug_suite("bench", mod)
    assert {r["task_id"] for r in results} == {"alpha", "beta"}


# ── run_debug_suite — benchmark lifecycle ────────────────────────────────────


def test_double_setup_metadata_preserved():
    """Multiple setup() calls must not overwrite task_metadata; subset_from_list still works."""
    task_ids = ("t1", "t2")

    class DoubleSetupBenchmark(Benchmark):
        benchmark_metadata = BenchmarkMetadata(name="double-setup-bench", version="0.1", description="test")
        task_metadata = {}
        task_config_class = DoneTaskConfig
        _setup_calls: int = PrivateAttr(default=0)

        def _setup(self) -> None:
            self._setup_calls += 1
            # Idempotent:Only populate if not already set (correct pattern for multiple setup() calls)
            if not self.task_metadata:
                object.__setattr__(
                    self,
                    "task_metadata",
                    {tid: TaskMetadata(id=tid) for tid in task_ids},
                )

        def close(self) -> None:
            pass

    benchmark = DoubleSetupBenchmark()
    benchmark.install()
    benchmark.setup()
    configs_first = list(benchmark.get_task_configs())
    assert len(configs_first) == 2
    assert {c.task_id for c in configs_first} == set(task_ids)

    benchmark.setup()  # second call must not overwrite
    configs_second = list(benchmark.get_task_configs())
    assert len(configs_second) == 2
    assert {c.task_id for c in configs_second} == set(task_ids)
    assert benchmark._setup_calls == 2

    # subset_from_list must still work after double setup
    subset = benchmark.subset_from_list(["t1"])
    subset_configs = list(subset.get_task_configs())
    assert len(subset_configs) == 1
    assert subset_configs[0].task_id == "t1"


def test_suite_benchmark_setup_and_close_called():
    mod, benchmark = _make_module()
    run_debug_suite("bench", mod)
    assert benchmark._install_calls == 1
    assert benchmark._setup_calls == 1
    assert benchmark._close_calls == 1


def test_suite_benchmark_closed_even_when_get_task_configs_raises():
    class FailingTaskConfigsBenchmark(Benchmark):
        benchmark_metadata = BenchmarkMetadata(name="test-bench", version="0.1", description="test")
        task_metadata = {}
        task_config_class = DoneTaskConfig
        _close_calls: int = PrivateAttr(default=0)

        def _setup(self) -> None:
            pass

        def close(self) -> None:
            self._close_calls += 1

        def get_task_configs(self):
            raise RuntimeError("config error")

    mod = ModuleType("fake_debug")
    benchmark = FailingTaskConfigsBenchmark()
    mod.get_debug_benchmark = lambda: benchmark  # type: ignore[attr-defined]
    mod.make_debug_agent = lambda tid: stop_agent  # type: ignore[attr-defined]

    with pytest.raises(RuntimeError, match="config error"):
        run_debug_suite("bench", mod)
    assert benchmark._close_calls == 1


# ── run_stress_test / collect_stress_compliance ───────────────────────────────


def test_run_stress_test_returns_outcome_with_passing_compliance():
    mod, _ = _make_module(task_ids=("t1",))
    out = run_stress_test(mod, print_json=False)
    assert out.report.benchmark == "fake_debug"
    assert len(out.episodes) == 1
    assert not out.failed_episodes
    assert "test_full_episode" in out.report.compliance["passed"]


def test_run_stress_test_raises_typeerror_on_missing_protocol():
    bad = ModuleType("nope")
    with pytest.raises(TypeError, match="get_debug_benchmark"):
        run_stress_test(bad)


def test_run_stress_test_save_delegates_to_report(tmp_path):
    mod, _ = _make_module(task_ids=("t1",))
    out = run_stress_test(mod, print_json=False)
    p = tmp_path / "stress.json"
    out.save(str(p))
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["benchmark"] == "fake_debug"
    assert "compliance" in data


def test_collect_stress_compliance_marks_full_episode_failed():
    mod, _ = _make_module()
    results = [
        {
            "task_id": "t1",
            "error": "boom",
            "done": False,
            "reward": 0.0,
            "steps": 0,
            "episode_time_s": 0.0,
            "step_times_s": [],
            "tools_list_ok": True,
            "close_idempotent_ok": True,
        }
    ]
    passed, failed = collect_stress_compliance(results, mod)
    assert "test_full_episode" in failed
    assert "test_debug_tasks_exist" in passed


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


# ── check_reset_reproducibility ───────────────────────────────────────────────

_nonce_seq = 0


class MinorDriftTextTask(Task):
    """Two instances produce long common prefix with a short differing suffix (simulates volatile ids)."""

    _close_calls: int = PrivateAttr(default=0)

    def reset(self):
        global _nonce_seq
        _nonce_seq += 1
        return Observation.from_text(f"{'y' * 100} id={_nonce_seq:04d}"), {}

    def evaluate(self, obs: Observation):
        return 1.0, {}

    def close(self):
        self._close_calls += 1
        super().close()


class MinorDriftTaskConfig(TaskConfig):
    def make(self, runtime_context=None, container_backend=None) -> MinorDriftTextTask:
        return MinorDriftTextTask(metadata=TaskMetadata(id=self.task_id), tool_config=NoopToolConfig())


_conflict_flip = 0


class ConflictingTextTask(Task):
    _close_calls: int = PrivateAttr(default=0)

    def reset(self):
        global _conflict_flip
        _conflict_flip += 1
        text = "alpha instruction" if _conflict_flip % 2 else "beta different"
        return Observation.from_text(text), {}

    def evaluate(self, obs: Observation):
        return 1.0, {}

    def close(self):
        self._close_calls += 1
        super().close()


class ConflictingTaskConfig(TaskConfig):
    def make(self, runtime_context=None, container_backend=None) -> ConflictingTextTask:
        return ConflictingTextTask(metadata=TaskMetadata(id=self.task_id), tool_config=NoopToolConfig())


_img_seq = 0


class SlightlyDifferentImageTask(Task):
    _close_calls: int = PrivateAttr(default=0)

    def reset(self):
        global _img_seq
        _img_seq += 1
        img = PILImage.new("RGB", (80, 80), (200, 200, 200))
        img.putpixel((20 + _img_seq, 30), (10, 10, 10))
        return Observation(contents=[ImageContent(data=img)]), {}

    def evaluate(self, obs: Observation):
        return 1.0, {}

    def close(self):
        self._close_calls += 1
        super().close()


class SlightlyDifferentImageTaskConfig(TaskConfig):
    def make(self, runtime_context=None, container_backend=None) -> SlightlyDifferentImageTask:
        return SlightlyDifferentImageTask(metadata=TaskMetadata(id=self.task_id), tool_config=NoopToolConfig())


def _module_with_task_config(task_config_cls, **benchmark_attrs):
    mod = ModuleType("fake_reset_repro")
    benchmark = DoneBenchmark()
    object.__setattr__(benchmark, "task_metadata", {"t1": TaskMetadata(id="t1")})
    object.__setattr__(benchmark, "task_config_class", task_config_cls)
    for k, v in benchmark_attrs.items():
        object.__setattr__(benchmark, k, v)
    mod.get_debug_benchmark = lambda: benchmark  # type: ignore[attr-defined]
    mod.make_debug_agent = lambda tid: stop_agent  # type: ignore[attr-defined]
    return mod


def test_check_reset_reproducibility_passes_identical_text():
    ok, msg = check_reset_reproducibility(_make_module()[0])
    assert ok is True
    assert msg == ""


def test_check_reset_reproducibility_passes_minor_text_drift():
    global _nonce_seq
    _nonce_seq = 0
    mod = _module_with_task_config(MinorDriftTaskConfig)
    ok, msg = check_reset_reproducibility(mod)
    assert ok is True, msg


def test_check_reset_reproducibility_fails_conflicting_text():
    global _conflict_flip
    _conflict_flip = 0
    mod = _module_with_task_config(ConflictingTaskConfig)
    ok, msg = check_reset_reproducibility(mod)
    assert ok is False
    assert msg


def test_check_reset_reproducibility_passes_similar_images():
    global _img_seq
    _img_seq = 0
    mod = _module_with_task_config(SlightlyDifferentImageTaskConfig)
    ok, msg = check_reset_reproducibility(mod)
    assert ok is True, msg


def test_check_reset_reproducibility_invalid_config_type():
    mod = _module_with_task_config(DoneTaskConfig, reset_reproducibility_config="bad")
    ok, msg = check_reset_reproducibility(mod)
    assert ok is False
    assert "ResetReproducibilityConfig" in msg


def test_check_reset_reproducibility_custom_config_strict_image_fails():
    global _img_seq
    _img_seq = 0
    mod = _module_with_task_config(
        SlightlyDifferentImageTaskConfig,
        reset_reproducibility_config=ResetReproducibilityConfig(image_max_mae=0.0),
    )
    ok, _ = check_reset_reproducibility(mod)
    assert ok is False


def test_observation_content_count_mismatch_fails():
    from cube.testing import _observations_equivalent_for_reset

    a = Observation(contents=[TextContent(data="hi")])
    b = Observation(contents=[TextContent(data="hi"), TextContent(data="there")])
    ok, msg = _observations_equivalent_for_reset(a, b, ResetReproducibilityConfig())
    assert ok is False
    assert "content count" in msg
