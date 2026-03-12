"""Tests for cube.tool - Tool, ToolConfig, tool_action."""

import inspect

import pytest

from cube.core import Action, Observation, StepError, TextContent
from cube.tool import Tool, tool_action


def assert_tool_docstrings_valid(tool_cls: type) -> None:
    """Validate that all @tool_action methods on a Tool class have parsable docstrings.

    Checks that every action has a non-empty description and that every
    parameter (excluding 'self') has a non-empty description.

    Raises AssertionError with a descriptive message on the first failure.
    """
    from cube.utils import function_to_dict

    actions = [
        (name, func)
        for name, func in inspect.getmembers(tool_cls, predicate=callable)
        if not name.startswith("_") and getattr(func, "_is_action", False)
    ]

    assert actions, f"{tool_cls.__name__} has no @tool_action methods"

    for name, func in actions:
        result = function_to_dict(func)
        assert result["description"], f"{tool_cls.__name__}.{name}: missing function description"
        for param_name, param_info in result["parameters"]["properties"].items():
            if param_name == "self":
                continue
            assert "description" in param_info and param_info["description"], (
                f"{tool_cls.__name__}.{name}: parameter '{param_name}' missing description"
            )


class EchoTool(Tool):
    @tool_action
    def echo(self, text: str) -> str:
        """Echo the given text."""
        return text

    @tool_action
    def add(self, a: float, b: float) -> str:
        """Add two numbers."""
        return str(a + b)

    @tool_action
    def crash(self) -> str:
        """Always raises."""
        raise RuntimeError("intentional error")

    def not_an_action(self) -> str:
        """Not decorated."""
        return "hidden"


def test_tool_action_decorator_sets_flag():
    tool = EchoTool()
    assert getattr(tool.echo, "_is_action", False) is True
    assert getattr(tool.not_an_action, "_is_action", False) is False


def test_tool_action_set_discovers_only_decorated_methods():
    action_names = {a.name for a in EchoTool().action_set}
    assert action_names == {"echo", "add", "crash"}


def test_tool_action_set_schemas_have_descriptions():
    for schema in EchoTool().action_set:
        assert schema.description != ""


def test_tool_execute_action_returns_observation():
    result = EchoTool().execute_action(Action(name="echo", arguments={"text": "hello"}))
    assert isinstance(result, Observation)
    assert result.contents == [TextContent(data="hello")]


def test_tool_execute_action_none_returns_success():
    """A @tool_action returning None should yield an Observation with 'Success'."""

    class SilentTool(Tool):
        @tool_action
        def do_nothing(self) -> None:
            """Does nothing."""

    result = SilentTool().execute_action(Action(name="do_nothing", arguments={}))
    assert isinstance(result, Observation)
    assert result.contents == [TextContent(data="Success")]


def test_tool_execute_action_unknown_method_raises():
    with pytest.raises(ValueError, match="does not exist"):
        EchoTool().execute_action(Action(name="nonexistent", arguments={}))


def test_tool_execute_action_non_action_method_raises():
    with pytest.raises(ValueError, match="not decorated"):
        EchoTool().execute_action(Action(name="not_an_action", arguments={}))


def test_tool_execute_action_exception_returns_step_error():
    result = EchoTool().execute_action(Action(name="crash", arguments={}))
    assert isinstance(result, StepError)
    assert result.error_type == "RuntimeError"
    assert result.exception_str == "intentional error"


# --- assert_tool_docstrings_valid ---


class WellDocumentedTool(Tool):
    @tool_action
    def greet(self, name: str) -> str:
        """Greet someone by name.

        Args:
            name: The name of the person to greet.
        """
        return f"Hello, {name}!"

    @tool_action
    def add(self, a: float, b: float) -> str:
        """Add two numbers together.

        Parameters
        ----------
        a : float
            First number.
        b : float
            Second number.
        """
        return str(a + b)


class MissingParamDescriptionTool(Tool):
    @tool_action
    def do_thing(self, value: int) -> str:
        """Do the thing."""  # no Args section
        return str(value)


class MissingFunctionDescriptionTool(Tool):
    @tool_action
    def do_thing(self) -> str:
        pass  # no docstring


def test_assert_tool_docstrings_valid_passes():
    assert_tool_docstrings_valid(WellDocumentedTool)


def test_assert_tool_docstrings_valid_catches_missing_param_description():
    with pytest.raises(AssertionError, match="missing description"):
        assert_tool_docstrings_valid(MissingParamDescriptionTool)


def test_assert_tool_docstrings_valid_catches_missing_function_description():
    with pytest.raises(AssertionError, match="missing function description"):
        assert_tool_docstrings_valid(MissingFunctionDescriptionTool)
