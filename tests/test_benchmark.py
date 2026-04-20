"""Tests for cube.benchmark - Benchmark, BenchmarkMetadata, subsetting."""

import json

import pytest

from cube.benchmark import Benchmark, BenchmarkMetadata
from cube.core import Observation
from cube.task import Task, TaskConfig, TaskMetadata
from cube.tool import Tool, ToolConfig, tool_action

# --- Minimal fixtures ---


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

    def evaluate(self, obs=None):
        return 0.0, {}


class _TaskConfig(TaskConfig):
    def make(self, runtime_context=None, container_backend=None):
        return _Task(
            metadata=TaskMetadata(id=self.task_id),
            tool_config=self.tool_config or _ToolConfig(),
        )


class MyBenchmark(Benchmark):
    benchmark_metadata = BenchmarkMetadata(
        name="MyBenchmark",
        version="2.0.0",
        description="Test benchmark",
        num_tasks=4,
    )
    task_metadata = {
        "t1": TaskMetadata(id="t1", split="train", extra_info={"difficulty": "easy"}),
        "t2": TaskMetadata(id="t2", split="train", extra_info={"difficulty": "hard"}),
        "t3": TaskMetadata(id="t3", split="val", extra_info={"difficulty": "easy"}),
        "t4": TaskMetadata(id="t4", split="test", extra_info={"difficulty": "hard"}),
    }
    task_config_class = _TaskConfig

    def _setup(self):
        pass

    def close(self):
        pass


# --- BenchmarkMetadata ---


def test_benchmark_metadata_defaults():
    bm = BenchmarkMetadata(name="foo", version="1.0", description="bar")
    assert bm.authors == []
    assert bm.tags == []
    assert bm.num_tasks == 0
    assert bm.license == ""
    assert bm.requirements == {}
    assert bm.extra_info == {}


# --- get_task_configs ---


def test_get_task_configs_yields_one_per_task():
    configs = list(MyBenchmark().get_task_configs())
    assert len(configs) == 4
    assert {c.task_id for c in configs} == {"t1", "t2", "t3", "t4"}


# --- subset_from_list ---


def test_subset_from_list_by_metadata_objects():
    bench = MyBenchmark()
    sub = bench.subset_from_list([bench.task_metadata["t2"], bench.task_metadata["t4"]])
    assert set(sub.task_metadata.keys()) == {"t2", "t4"}


def test_subset_from_list_invalid_ids_raise():
    with pytest.raises(ValueError, match="do not exist"):
        MyBenchmark().subset_from_list(["t1", "nonexistent"])


def test_subset_preserves_subclass_type_and_instance_fields():
    """subset_from_list returns the same subclass with instance fields and metadata intact."""

    class BenchmarkWithField(Benchmark):
        benchmark_metadata = BenchmarkMetadata(
            name="FieldBench",
            version="1.0",
            description="benchmark with extra field",
            num_tasks=3,
            authors=["Alice"],
            tags=["test"],
        )
        task_metadata = {
            "f1": TaskMetadata(id="f1"),
            "f2": TaskMetadata(id="f2"),
            "f3": TaskMetadata(id="f3"),
        }
        task_config_class = _TaskConfig
        a: str = "hello"

        def _setup(self):
            pass

        def close(self):
            pass

    bench = BenchmarkWithField()
    sub = bench.subset_from_list(["f1", "f2"], benchmark_name_suffix="small")

    # Subclass type and instance field are preserved
    assert isinstance(sub, BenchmarkWithField)
    assert sub.a == "hello"

    # task_metadata reflects the subset
    assert set(sub.task_metadata.keys()) == {"f1", "f2"}

    # benchmark_metadata: name and count updated, all other fields preserved
    assert sub.benchmark_metadata.name == "FieldBench_small"
    assert sub.benchmark_metadata.num_tasks == 2
    assert sub.benchmark_metadata.version == "1.0"
    assert sub.benchmark_metadata.authors == ["Alice"]
    assert sub.benchmark_metadata.tags == ["test"]

    # Original benchmark is not mutated by the subset operation
    assert bench.benchmark_metadata.name == "FieldBench"
    assert bench.benchmark_metadata.num_tasks == 3
    assert set(bench.task_metadata.keys()) == {"f1", "f2", "f3"}


# --- subset_from_glob ---


def test_subset_from_glob_by_split():
    sub = MyBenchmark().subset_from_glob("split", "train")
    assert set(sub.task_metadata.keys()) == {"t1", "t2"}


def test_subset_from_glob_by_extra_info():
    sub = MyBenchmark().subset_from_glob("extra_info.difficulty", "easy")
    assert set(sub.task_metadata.keys()) == {"t1", "t3"}


def test_subset_from_glob_no_match_raises():
    with pytest.raises(ValueError, match="No tasks found"):
        MyBenchmark().subset_from_glob("split", "nonexistent_split")


# --- File loading ---


def test_benchmark_metadata_from_json(tmp_path):
    p = tmp_path / "bm.json"
    p.write_text(json.dumps({"name": "json-bench", "version": "3.0", "description": "from JSON"}))
    bm = Benchmark.benchmark_metadata_from_json(p)
    assert bm.name == "json-bench"
    assert bm.version == "3.0"
    assert bm.description == "from JSON"


def test_benchmark_metadata_from_csv(tmp_path):
    p = tmp_path / "bm.csv"
    p.write_text(
        'name,version,description,num_tasks,authors,tags\ncsv-bench,4.0,from CSV,5,"[""Alice""]","[""toy""]"\n'
    )
    bm = Benchmark.benchmark_metadata_from_csv(p)
    assert bm.name == "csv-bench"
    assert bm.version == "4.0"
    assert bm.num_tasks == 5
    assert bm.authors == ["Alice"]
    assert bm.tags == ["toy"]


def test_task_metadata_from_json(tmp_path):
    p = tmp_path / "tasks.json"
    p.write_text(
        json.dumps(
            [
                {"id": "task-a", "split": "train"},
                {"id": "task-b", "split": "test", "extra_info": {"difficulty": "hard"}},
            ]
        )
    )
    tasks = Benchmark.task_metadata_from_json(p)
    assert set(tasks.keys()) == {"task-a", "task-b"}
    assert tasks["task-a"].split == "train"
    assert tasks["task-b"].extra_info == {"difficulty": "hard"}


def test_task_metadata_from_csv(tmp_path):
    p = tmp_path / "tasks.csv"
    p.write_text("id,split,abstract_description\ntask-a,train,do something\ntask-b,test,do other\n")
    tasks = Benchmark.task_metadata_from_csv(p)
    assert set(tasks.keys()) == {"task-a", "task-b"}
    assert tasks["task-a"].split == "train"
    assert tasks["task-b"].split == "test"


# --- spawn ---


def test_spawn_returns_ready_task():
    bench = MyBenchmark()
    bench.setup()
    task = bench.spawn(_TaskConfig(task_id="t1"))
    assert isinstance(task, Task)
    obs, _ = task.reset()
    assert obs.contents[0].data == "ready"


def test_spawn_unknown_task_raises():
    bench = MyBenchmark()
    bench.setup()
    with pytest.raises(ValueError, match="not found"):
        bench.spawn(_TaskConfig(task_id="nonexistent"))
