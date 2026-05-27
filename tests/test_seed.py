"""Tests for cube.seed - BasicSeedGenerator."""

import json

from cube.benchmark import Benchmark, BenchmarkConfig, BenchmarkMetadata
from cube.core import Observation, TaskResult
from cube.seed import AbstractSeedGenerator, BasicSeedGenerator
from cube.task import Task, TaskConfig, TaskMetadata
from cube.tool import Tool, ToolConfig, tool_action


def test_basic_seed_generator_returns_list_of_ints():
    seeds = BasicSeedGenerator(n_seed=3, meta_seed=42)(TaskMetadata(id="task-1"))
    assert isinstance(seeds, list)
    assert len(seeds) == 3
    assert all(isinstance(s, int) for s in seeds)


def test_basic_seed_generator_is_deterministic():
    gen = BasicSeedGenerator(n_seed=4, meta_seed=0)
    tm = TaskMetadata(id="repro-task")
    assert gen(tm) == gen(tm)


def test_basic_seed_generator_different_tasks_produce_different_seeds():
    gen = BasicSeedGenerator(n_seed=3, meta_seed=99)
    assert gen(TaskMetadata(id="task-alpha")) != gen(TaskMetadata(id="task-beta"))


def test_basic_seed_generator_different_meta_seeds_produce_different_seeds():
    tm = TaskMetadata(id="same-task")
    assert BasicSeedGenerator(n_seed=3, meta_seed=1)(tm) != BasicSeedGenerator(n_seed=3, meta_seed=2)(tm)


def test_basic_seed_generator_seeds_in_valid_numpy_range():
    for seed in BasicSeedGenerator(n_seed=10, meta_seed=1)(TaskMetadata(id="range-task")):
        assert 0 <= seed < 2**32


# ── Full BenchmarkConfig → JSON → worker pipeline ───────────────────────────


class _CustomSeedGenerator(AbstractSeedGenerator):
    """Cube-defined seed generator with a subclass-specific field."""

    offset: int = 100

    def __call__(self, task_metadata: TaskMetadata) -> list[int]:
        return [self.offset, self.offset + len(task_metadata.id)]


class _Tool(Tool):
    @tool_action
    def noop(self) -> str:
        """No-op."""
        return "ok"


class _ToolConfig(ToolConfig):
    def make(self, container=None):
        return _Tool()


class _Task(Task):
    def reset(self):
        return Observation.from_text("ready"), {}

    def evaluate(self, obs=None) -> TaskResult:
        return TaskResult(reward=0.0, checks=[], info={})


class _TaskConfig(TaskConfig):
    def make(self, runtime_context=None, container_backend=None):
        return _Task(metadata=self.metadata, tool_config=self.tool_config or _ToolConfig())


class _Benchmark(Benchmark):
    def _setup(self):
        pass

    def close(self):
        pass


class _SeededBenchmarkConfig(BenchmarkConfig):
    benchmark_metadata = BenchmarkMetadata(name="seeded", version="1.0", description="seed pipeline test", num_tasks=2)
    task_metadata = {
        "task-a": TaskMetadata(id="task-a"),
        "task-b": TaskMetadata(id="task-b"),
    }
    task_config_class = _TaskConfig
    benchmark_class = _Benchmark


def test_seed_generator_expands_tasks_in_get_task_configs():
    """Each task with a seed_generator emits one TaskConfig per seed."""
    cfg = _SeededBenchmarkConfig(seed_generator=_CustomSeedGenerator(offset=100))
    configs = list(cfg.get_task_configs())
    # 2 tasks × 2 seeds each = 4 emitted configs
    assert len(configs) == 4
    seeds_for_a = sorted(c.seed for c in configs if c.task_id == "task-a")
    assert seeds_for_a == [100, 106]


def test_seed_generator_survives_full_json_round_trip_pipeline():
    """BenchmarkConfig → JSON → worker round-trip preserves the custom seed generator
    subclass and its subclass-specific fields, and the rehydrated config emits the
    same expanded TaskConfigs the original would.
    """
    original = _SeededBenchmarkConfig(seed_generator=_CustomSeedGenerator(offset=42))

    # Driver-side: serialize to JSON (the cross-process boundary).
    payload = original.model_dump_json()
    assert "_type" in json.loads(payload)["seed_generator"]  # discriminator preserved

    # Worker-side: rehydrate via TypedBaseModel polymorphic dispatch.
    restored = BenchmarkConfig.model_validate_json(payload)
    assert isinstance(restored, _SeededBenchmarkConfig)
    assert isinstance(restored.seed_generator, _CustomSeedGenerator)
    assert restored.seed_generator.offset == 42  # subclass-specific field preserved

    # Expanded TaskConfigs match between driver and worker side.
    original_configs = [(c.task_id, c.seed) for c in original.get_task_configs()]
    restored_configs = [(c.task_id, c.seed) for c in restored.get_task_configs()]
    assert original_configs == restored_configs
