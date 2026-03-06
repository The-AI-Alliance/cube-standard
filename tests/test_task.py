"""Tests for cube.task - Task, TaskMetadata, STOP_ACTION."""

from cube.benchmark import RuntimeContext  # noqa: F401 – required for Task.model_rebuild()
from cube.container import Container
from cube.core import Action, EnvironmentOutput, Observation, StepError, TextContent
from cube.environment import Environment, EnvironmentConfig, environment_action
from cube.task import STOP_ACTION, Task, TaskMetadata


class GreetEnvironment(Environment):
    @environment_action
    def greet(self, name: str) -> str:
        """Greet someone."""
        return f"Hello, {name}!"

    @environment_action
    def fail(self) -> str:
        """Always raises."""
        raise ValueError("action failed")


class GreetEnvironmentConfig(EnvironmentConfig):
    def make(self, container: Container | None = None) -> GreetEnvironment:
        return GreetEnvironment()


class SimpleTask(Task):
    def reset(self):
        return Observation.from_text("ready"), {}

    def evaluate(self, obs: Observation):
        return 0.5, {"score": 0.5}


def make_task(**kwargs) -> SimpleTask:
    return SimpleTask(
        metadata=TaskMetadata(id="simple-task"),
        env_config=GreetEnvironmentConfig(),
        **kwargs,
    )


# --- TaskMetadata ---


def test_task_metadata_defaults():
    tm = TaskMetadata(id="my-task")
    assert tm.split == "test"
    assert tm.abstract_description == ""
    assert tm.recommended_max_steps is None
    assert tm.container_config is None
    assert tm.extra_info == {}


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
    assert out.info == {"score": 0.5}


def test_task_action_set_comes_from_environment():
    names = {a.name for a in make_task().action_set}
    assert names == {"greet", "fail"}
