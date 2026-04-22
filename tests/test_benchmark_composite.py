"""Tests for CompositeBenchmarkConfig / CompositeBenchmark / CompositeTaskConfig.

Covers task-id prefixing, uniqueness enforcement, spawn() routing, and
serialization round-trip of nested composites.
"""

from __future__ import annotations

import pytest

from cube.benchmark import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkMetadata,
    CompositeBenchmark,
    CompositeBenchmarkConfig,
)
from cube.core import Observation
from cube.task import CompositeTaskConfig, Task, TaskConfig, TaskMetadata
from cube.tool import Tool, ToolConfig, tool_action

# ── Minimal fixtures ──────────────────────────────────────────────────────────


class _Tool(Tool):
    @tool_action
    def noop(self) -> str:
        """No-op."""
        return "ok"


class _ToolConfig(ToolConfig):
    def make(self, container=None):
        return _Tool()


class _Task(Task):
    """Returns a reset message that includes the task id so spawn routing is observable."""

    def reset(self):
        return Observation.from_text(f"reset:{self.metadata.id}"), {}

    def evaluate(self, obs=None):
        return 0.0, {}


class _TaskConfig(TaskConfig):
    def make(self, runtime_context=None, container_backend=None):
        return _Task(
            metadata=TaskMetadata(id=self.task_id),
            tool_config=self.tool_config or _ToolConfig(),
        )


class _BenchmarkA(Benchmark):
    def _setup(self):
        self._runtime_context = {"source": "A"}

    def close(self):
        pass


class _BenchmarkB(Benchmark):
    def _setup(self):
        self._runtime_context = {"source": "B"}

    def close(self):
        pass


class ConfigA(BenchmarkConfig):
    benchmark_metadata = BenchmarkMetadata(name="bench-a", version="1", description="A")
    task_metadata = {
        "task-1": TaskMetadata(id="task-1"),
        "task-2": TaskMetadata(id="task-2"),
    }
    task_config_class = _TaskConfig
    benchmark_class = _BenchmarkA


class ConfigB(BenchmarkConfig):
    benchmark_metadata = BenchmarkMetadata(name="bench-b", version="1", description="B")
    # Intentionally shares the task id "task-1" with ConfigA to prove prefixing disambiguates.
    task_metadata = {
        "task-1": TaskMetadata(id="task-1"),
    }
    task_config_class = _TaskConfig
    benchmark_class = _BenchmarkB


# ── Construction / duplicate-name guard ──────────────────────────────────────


def test_duplicate_sub_benchmark_names_raise():
    with pytest.raises(ValueError, match="unique sub-benchmark names"):
        CompositeBenchmarkConfig(sub_configs=[ConfigA(), ConfigA()])


def test_composite_metadata_reflects_subs():
    suite = CompositeBenchmarkConfig(
        sub_configs=[ConfigA(), ConfigB()],
        composite_name="my-suite",
        composite_version="0.2.1",
        composite_description="A+B",
    )
    meta = suite.benchmark_metadata
    assert meta.name == "my-suite"
    assert meta.version == "0.2.1"
    assert meta.description == "A+B"
    assert meta.num_tasks == 3  # 2 from A + 1 from B


def test_composite_num_tasks_matches_sum():
    suite = CompositeBenchmarkConfig(sub_configs=[ConfigA(), ConfigB()])
    assert suite.num_tasks == 3


# ── task_metadata prefixing ──────────────────────────────────────────────────


def test_task_metadata_is_prefixed():
    suite = CompositeBenchmarkConfig(sub_configs=[ConfigA(), ConfigB()])
    keys = list(suite.task_metadata.keys())
    assert keys == ["bench-a/task-1", "bench-a/task-2", "bench-b/task-1"]


def test_prefixing_disambiguates_duplicate_inner_ids():
    """Both ConfigA and ConfigB declare task_id='task-1'; prefixing keeps them distinct."""
    suite = CompositeBenchmarkConfig(sub_configs=[ConfigA(), ConfigB()])
    assert "bench-a/task-1" in suite.task_metadata
    assert "bench-b/task-1" in suite.task_metadata


# ── get_task_configs emits CompositeTaskConfig wrappers ──────────────────────


def test_get_task_configs_emits_wrappers():
    suite = CompositeBenchmarkConfig(sub_configs=[ConfigA(), ConfigB()])
    configs = list(suite.get_task_configs())

    assert len(configs) == 3
    assert all(isinstance(c, CompositeTaskConfig) for c in configs)

    ids = [c.task_id for c in configs]
    assert ids == ["bench-a/task-1", "bench-a/task-2", "bench-b/task-1"]

    # Inner configs keep their native un-prefixed ids so sub-benchmark ClassVar lookups still work.
    inner_ids = [c.inner.task_id for c in configs]
    assert inner_ids == ["task-1", "task-2", "task-1"]

    # sub_name is set correctly
    assert [c.sub_name for c in configs] == ["bench-a", "bench-a", "bench-b"]


def test_task_ids_subset_on_composite():
    """Setting task_ids on the composite filters emitted configs by prefixed id."""
    suite = CompositeBenchmarkConfig(sub_configs=[ConfigA(), ConfigB()])
    suite = suite.model_copy(update={"task_ids": ["bench-a/task-2", "bench-b/task-1"]})
    configs = list(suite.get_task_configs())
    assert [c.task_id for c in configs] == ["bench-a/task-2", "bench-b/task-1"]


# ── make() / spawn() routing ─────────────────────────────────────────────────


def test_make_returns_composite_benchmark_with_all_subs_ready():
    suite = CompositeBenchmarkConfig(sub_configs=[ConfigA(), ConfigB()])
    bench = suite.make()
    try:
        assert isinstance(bench, CompositeBenchmark)
        assert set(bench.sub_benchmarks.keys()) == {"bench-a", "bench-b"}
        # Sub-benchmarks had setup() run — runtime_context is populated
        assert bench.sub_benchmarks["bench-a"]._runtime_context == {"source": "A"}
        assert bench.sub_benchmarks["bench-b"]._runtime_context == {"source": "B"}
    finally:
        bench.close()


def test_spawn_routes_to_correct_sub_benchmark():
    suite = CompositeBenchmarkConfig(sub_configs=[ConfigA(), ConfigB()])
    configs = list(suite.get_task_configs())
    with suite.make() as bench:
        for tc in configs:
            task = bench.spawn(tc)
            obs, _ = task.reset()
            # Inner task sees its native un-prefixed id
            assert obs == Observation.from_text(f"reset:{tc.inner.task_id}")
            task.close()


def test_spawn_rejects_non_composite_task_config():
    suite = CompositeBenchmarkConfig(sub_configs=[ConfigA(), ConfigB()])
    with suite.make() as bench:
        with pytest.raises(ValueError, match="CompositeTaskConfig"):
            bench.spawn(_TaskConfig(task_id="task-1"))


def test_spawn_rejects_unknown_sub_name():
    suite = CompositeBenchmarkConfig(sub_configs=[ConfigA()])
    with suite.make() as bench:
        bogus = CompositeTaskConfig(
            task_id="bench-x/task-1",
            sub_name="bench-x",
            inner=_TaskConfig(task_id="task-1"),
        )
        with pytest.raises(ValueError, match="Unknown sub-benchmark"):
            bench.spawn(bogus)


def test_composite_close_closes_every_sub():
    closed: list[str] = []

    class RecordingBench(Benchmark):
        name_tag: str

        def __init__(self, config, *, tag):
            super().__init__(config)
            self.name_tag = tag

        def _setup(self):
            pass

        def close(self):
            closed.append(self.name_tag)

    class ConfigX(BenchmarkConfig):
        benchmark_metadata = BenchmarkMetadata(name="x", version="1", description="x")
        task_metadata = {"x": TaskMetadata(id="x")}
        task_config_class = _TaskConfig
        benchmark_class = RecordingBench

    class ConfigY(BenchmarkConfig):
        benchmark_metadata = BenchmarkMetadata(name="y", version="1", description="y")
        task_metadata = {"y": TaskMetadata(id="y")}
        task_config_class = _TaskConfig
        benchmark_class = RecordingBench

    # Manually wire tag because our subclass expects a kwarg; simpler to wrap via make path:
    # Instead, test via CompositeBenchmark.close() directly on pre-wired sub_benchmarks.
    suite = CompositeBenchmarkConfig(sub_configs=[ConfigX(), ConfigY()])
    composite = CompositeBenchmark(config=suite)
    composite.sub_benchmarks["x"] = RecordingBench(ConfigX(), tag="x")
    composite.sub_benchmarks["y"] = RecordingBench(ConfigY(), tag="y")
    composite.close()
    assert set(closed) == {"x", "y"}


def test_composite_close_continues_after_sub_failure():
    closed: list[str] = []

    class OKBench(Benchmark):
        def _setup(self):
            pass

        def close(self):
            closed.append("ok")

    class BadBench(Benchmark):
        def _setup(self):
            pass

        def close(self):
            closed.append("bad-attempted")
            raise RuntimeError("close failed")

    class ConfigOK(BenchmarkConfig):
        benchmark_metadata = BenchmarkMetadata(name="ok", version="1", description="ok")
        task_metadata = {"x": TaskMetadata(id="x")}
        task_config_class = _TaskConfig
        benchmark_class = OKBench

    class ConfigBad(BenchmarkConfig):
        benchmark_metadata = BenchmarkMetadata(name="bad", version="1", description="bad")
        task_metadata = {"y": TaskMetadata(id="y")}
        task_config_class = _TaskConfig
        benchmark_class = BadBench

    suite = CompositeBenchmarkConfig(sub_configs=[ConfigBad(), ConfigOK()])
    bench = suite.make()
    # close() must not re-raise; every sub gets a close attempt
    bench.close()
    assert closed == ["bad-attempted", "ok"]


# ── Serialization round-trip ─────────────────────────────────────────────────


def test_composite_json_round_trip():
    suite = CompositeBenchmarkConfig(
        sub_configs=[ConfigA(), ConfigB().subset_from_list(["task-1"])],
        composite_name="multi",
    )
    payload = suite.model_dump_json()
    reloaded = CompositeBenchmarkConfig.model_validate_json(payload)

    assert reloaded.composite_name == "multi"
    assert len(reloaded.sub_configs) == 2
    assert reloaded.sub_configs[0].name == "bench-a"
    assert reloaded.sub_configs[1].name == "bench-b"
    assert reloaded.sub_configs[1].task_ids == ["task-1"]
    # Merged task_metadata keys are identical
    assert list(reloaded.task_metadata.keys()) == list(suite.task_metadata.keys())


def test_composite_of_composite():
    """Composites nest: a CompositeBenchmarkConfig can be a sub_config of another."""
    inner = CompositeBenchmarkConfig(
        sub_configs=[ConfigA(), ConfigB()],
        composite_name="inner-suite",
    )
    outer = CompositeBenchmarkConfig(
        sub_configs=[inner, ConfigA().subset_from_list(["task-1"])],
        composite_name="outer-suite",
    )
    # Outer merges inner (prefixed by inner-suite) and ConfigA (prefixed by bench-a).
    # But inner-suite contains bench-a under "bench-a/…". Inner is not a name collision
    # with the second ConfigA because inner.name == "inner-suite" != "bench-a".
    assert set(outer.task_metadata.keys()) == {
        "inner-suite/bench-a/task-1",
        "inner-suite/bench-a/task-2",
        "inner-suite/bench-b/task-1",
        "bench-a/task-1",
    }

    # JSON round-trip still works
    payload = outer.model_dump_json()
    reloaded = CompositeBenchmarkConfig.model_validate_json(payload)
    assert set(reloaded.task_metadata.keys()) == set(outer.task_metadata.keys())


# ── install() / uninstall() delegation ────────────────────────────────────────


def test_composite_install_delegates_to_subs():
    call_log: list[str] = []

    class InstallingConfig(BenchmarkConfig):
        benchmark_metadata = BenchmarkMetadata(name="installable", version="1", description="x")
        task_metadata = {"x": TaskMetadata(id="x")}
        task_config_class = _TaskConfig
        benchmark_class = _BenchmarkA

        @classmethod
        def install(cls) -> None:
            call_log.append("installable")

    class AnotherInstallingConfig(BenchmarkConfig):
        benchmark_metadata = BenchmarkMetadata(name="other", version="1", description="x")
        task_metadata = {"y": TaskMetadata(id="y")}
        task_config_class = _TaskConfig
        benchmark_class = _BenchmarkA

        @classmethod
        def install(cls) -> None:
            call_log.append("other")

    suite = CompositeBenchmarkConfig(sub_configs=[InstallingConfig(), AnotherInstallingConfig()])
    suite.install()
    assert call_log == ["installable", "other"]
