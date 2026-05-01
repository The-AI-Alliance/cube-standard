"""Tests for cube.task - Task, TaskMetadata, STOP_ACTION."""

import json

import pytest

from cube.container import Container
from cube.core import Action, EnvironmentOutput, Observation, StepError, TextContent
from cube.task import STOP_ACTION, Task, TaskConfig, TaskExecutionInfo, TaskMetadata
from cube.tool import Tool, ToolConfig, tool_action


class GreetTool(Tool):
    @tool_action
    def greet(self, name: str) -> str:
        """Greet someone."""
        return f"Hello, {name}!"

    @tool_action
    def fail(self) -> str:
        """Always raises."""
        raise ValueError("action failed")


class GreetToolConfig(ToolConfig):
    def make(self, container: Container | None = None) -> GreetTool:
        return GreetTool()


class SimpleTask(Task):
    def reset(self):
        return Observation.from_text("ready"), {}

    def evaluate(self, obs: Observation | None = None):
        return 0.5, {"score": 0.5}


def make_task(**kwargs) -> SimpleTask:
    return SimpleTask(
        metadata=TaskMetadata(id="simple-task"),
        tool_config=GreetToolConfig(),
        **kwargs,
    )


# --- TaskMetadata ---


def test_task_metadata_defaults():
    tm = TaskMetadata(id="my-task")
    assert tm == TaskMetadata(
        id="my-task",
        split="test",
        abstract_description="",
        recommended_max_steps=None,
        container_config=None,
    )


# --- Task.reset ---


def test_task_reset():
    obs, info = make_task().reset()
    assert obs.contents == [TextContent(data="ready")]
    assert info == {}


# --- Task.step ---


def test_task_step_stop_action_marks_done():
    out = make_task().step(Action(name=STOP_ACTION.name, arguments={}))
    assert isinstance(out, EnvironmentOutput)
    assert out.done is True
    assert out.error is None


def test_task_step_regular_action():
    out = make_task().step(Action(name="greet", arguments={"name": "World"}))
    assert isinstance(out, EnvironmentOutput)
    assert out.done is False
    assert out.obs.contents == [TextContent(data="Hello, World!")]


def test_task_step_action_error_sets_done_and_error():
    out = make_task().step(Action(name="fail", arguments={}))
    assert out.done is True
    assert isinstance(out.error, StepError)
    assert out.error.error_type == "ValueError"


def test_task_validate_per_step_triggers_evaluate():
    out = make_task(validate_per_step=True).step(Action(name="greet", arguments={"name": "Alice"}))
    assert out.reward == 0.5
    assert out.info["score"] == 0.5
    assert "profiling" in out.info


def test_task_action_set_comes_from_tool():
    names = {a.name for a in make_task().action_set}
    assert names == {"greet", "fail"}


# --- TaskExecutionInfo ---


class _MyExecutionInfo(TaskExecutionInfo):
    instruction: str
    patch: str = ""


def test_task_execution_info_subclass_round_trip():
    """Subclasses round-trip through JSON via the ``_type`` discriminator."""
    info = _MyExecutionInfo(instruction="solve it", patch="diff --git ...")
    payload = info.model_dump_json()
    restored = TaskExecutionInfo.model_validate_json(payload)
    assert isinstance(restored, _MyExecutionInfo)
    assert restored == info


def test_task_execution_info_default_on_task_is_none():
    assert make_task().execution_info is None


def test_task_execution_info_field_round_trips_via_task():
    """The ``execution_info`` slot on Task preserves subclass fields through JSON."""
    info = _MyExecutionInfo(instruction="solve it")
    task = make_task(execution_info=info)
    reloaded = SimpleTask.model_validate_json(task.model_dump_json())
    assert isinstance(reloaded.execution_info, _MyExecutionInfo)
    assert reloaded.execution_info.instruction == "solve it"


# --- TaskConfig cache helpers ---


class _CacheTaskConfig(TaskConfig):
    def make(self, runtime_context=None, container_backend=None):
        return SimpleTask(metadata=self.metadata, tool_config=GreetToolConfig())


def _cache_task_config(task_id: str = "task-1") -> _CacheTaskConfig:
    return _CacheTaskConfig(metadata=TaskMetadata(id=task_id), tool_config=GreetToolConfig())


def test_task_execution_cache_dir_falls_back_to_package_when_unowned(monkeypatch, tmp_path):
    """Without a back-stamp, the default keys on the top-level Python package."""
    import cube as cube_pkg

    monkeypatch.setattr(cube_pkg, "_CUBE_CACHE_ROOT", tmp_path)
    # _CacheTaskConfig is defined in this test module: top-level package "tests".
    expected = tmp_path / "tests" / "tasks_execution_info"
    assert _CacheTaskConfig.task_execution_cache_dir() == expected


def test_task_execution_cache_dir_uses_back_stamped_benchmark_name(monkeypatch, tmp_path):
    """When BenchmarkConfig back-stamps ``_benchmark_cache_name``, the cache keys on it."""
    import cube as cube_pkg

    monkeypatch.setattr(cube_pkg, "_CUBE_CACHE_ROOT", tmp_path)
    monkeypatch.setattr(_CacheTaskConfig, "_benchmark_cache_name", "my-bench")
    assert _CacheTaskConfig.task_execution_cache_dir() == tmp_path / "my-bench" / "tasks_execution_info"


def test_load_task_execution_info_raises_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(_CacheTaskConfig, "task_execution_cache_dir", classmethod(lambda cls: tmp_path))
    with pytest.raises(RuntimeError, match="No execution data"):
        _cache_task_config("missing-task").load_task_execution_info()


def test_load_task_execution_info_returns_dict_when_present(monkeypatch, tmp_path):
    monkeypatch.setattr(_CacheTaskConfig, "task_execution_cache_dir", classmethod(lambda cls: tmp_path))
    (tmp_path / "task-1.json").write_text(json.dumps({"instruction": "solve it"}))
    assert _cache_task_config("task-1").load_task_execution_info() == {"instruction": "solve it"}


def test_verify_installed_default_is_noop():
    """Base ``verify_installed`` is a no-op so cubes opt in only when they want fail-fast."""
    assert _cache_task_config().verify_installed() is None
