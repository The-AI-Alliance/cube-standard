"""Tests for cube.environment - Environment, EnvironmentConfig, environment_action."""

import pytest

from cube.core import Action, Observation, StepError, TextContent
from cube.environment import Environment, environment_action


class EchoEnvironment(Environment):
    @environment_action
    def echo(self, text: str) -> str:
        """Echo the given text."""
        return text

    @environment_action
    def add(self, a: float, b: float) -> str:
        """Add two numbers."""
        return str(a + b)

    @environment_action
    def crash(self) -> str:
        """Always raises."""
        raise RuntimeError("intentional error")

    def not_an_action(self) -> str:
        """Not decorated."""
        return "hidden"


def test_environment_action_decorator_sets_flag():
    env = EchoEnvironment()
    assert getattr(env.echo, "_is_action", False) is True
    assert getattr(env.not_an_action, "_is_action", False) is False


def test_environment_action_set_discovers_only_decorated_methods():
    action_names = {a.name for a in EchoEnvironment().action_set}
    assert action_names == {"echo", "add", "crash"}


def test_environment_action_set_schemas_have_descriptions():
    for schema in EchoEnvironment().action_set:
        assert schema.description != ""


def test_environment_execute_action_returns_observation():
    result = EchoEnvironment().execute_action(Action(name="echo", arguments={"text": "hello"}))
    assert isinstance(result, Observation)
    assert result.contents == [TextContent(data="hello")]


def test_environment_execute_action_none_returns_success():
    """A @environment_action returning None should yield an Observation with 'Success'."""

    class SilentEnvironment(Environment):
        @environment_action
        def do_nothing(self) -> None:
            """Does nothing."""

    result = SilentEnvironment().execute_action(Action(name="do_nothing", arguments={}))
    assert isinstance(result, Observation)
    assert result.contents == [TextContent(data="Success")]


def test_environment_execute_action_unknown_method_raises():
    with pytest.raises(ValueError, match="does not exist"):
        EchoEnvironment().execute_action(Action(name="nonexistent", arguments={}))


def test_environment_execute_action_non_action_method_raises():
    with pytest.raises(ValueError, match="not decorated"):
        EchoEnvironment().execute_action(Action(name="not_an_action", arguments={}))


def test_environment_execute_action_exception_returns_step_error():
    result = EchoEnvironment().execute_action(Action(name="crash", arguments={}))
    assert isinstance(result, StepError)
    assert result.error_type == "RuntimeError"
    assert result.exception_str == "intentional error"
