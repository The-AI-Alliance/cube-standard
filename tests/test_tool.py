"""Tests for cube.tool - Tool, AsyncTool, ToolConfig, tool_action."""

import inspect

import pytest

from cube.core import Action, ActionSchema, Observation, StepError, TextContent
from cube.tool import AsyncTool, AsyncToolbox, Tool, Toolbox, tool_action


def assert_tool_docstrings_valid(tool_cls: type) -> None:
    """Validate that all @tool_action methods on a Tool class have parsable docstrings.

    Checks that every action has a non-empty description and that every
    parameter (excluding 'self') has a non-empty description.

    Raises AssertionError with a descriptive message on the first failure.
    """
    actions = [
        (name, func)
        for name, func in inspect.getmembers(tool_cls, predicate=callable)
        if not name.startswith("_") and getattr(func, "_is_action", False)
    ]

    assert actions, f"{tool_cls.__name__} has no @tool_action methods"

    for name, func in actions:
        try:
            schema = ActionSchema.from_function(func)
        except ValueError as e:
            raise AssertionError(f"{tool_cls.__name__}.{name}: {e}") from e
        ok, msg = schema.validate_param_descriptions()
        assert ok, f"{tool_cls.__name__}.{name}: {msg}"


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


class AsyncEchoTool(AsyncTool):
    @tool_action
    async def echo(self, text: str) -> str:
        """Echo the given text."""
        return text

    @tool_action
    async def add(self, a: float, b: float) -> str:
        """Add two numbers."""
        return str(a + b)

    @tool_action
    async def crash(self) -> str:
        """Always raises."""
        raise RuntimeError("intentional error")

    async def not_an_action(self) -> str:
        """Not decorated."""
        return "hidden"


# ── sync Tool ──────────────────────────────────────────────────────────────────


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


def test_tool_execute_action_missing_required_arg_returns_observation() -> None:
    """Missing required argument → recoverable error Observation, not StepError."""
    result = EchoTool().execute_action(Action(name="echo", arguments={}))
    assert isinstance(result, Observation)
    assert "Bad arguments" in result.contents[0].data
    assert "echo" in result.contents[0].data


def test_tool_execute_action_unknown_kwarg_returns_observation() -> None:
    """Unknown keyword argument → recoverable error Observation, not StepError."""
    result = EchoTool().execute_action(Action(name="echo", arguments={"text": "hi", "typo": "x"}))
    assert isinstance(result, Observation)
    assert "Bad arguments" in result.contents[0].data


# ── AsyncTool ─────────────────────────────────────────────────────────────────


def test_async_tool_action_decorator_sets_flag():
    tool = AsyncEchoTool()
    assert getattr(tool.echo, "_is_action", False) is True
    assert getattr(tool.not_an_action, "_is_action", False) is False


def test_async_tool_action_set_discovers_only_decorated_methods():
    action_names = {a.name for a in AsyncEchoTool().action_set}
    assert action_names == {"echo", "add", "crash"}


def test_async_tool_action_set_schemas_have_descriptions():
    for schema in AsyncEchoTool().action_set:
        assert schema.description != ""


@pytest.mark.asyncio
async def test_async_tool_execute_action_returns_observation():
    result = await AsyncEchoTool().execute_action(Action(name="echo", arguments={"text": "hello"}))
    assert isinstance(result, Observation)
    assert result.contents == [TextContent(data="hello")]


@pytest.mark.asyncio
async def test_async_tool_execute_action_none_returns_success():
    """A @tool_action returning None should yield an Observation with 'Success'."""

    class SilentAsyncTool(AsyncTool):
        @tool_action
        async def do_nothing(self) -> None:
            """Does nothing."""

    result = await SilentAsyncTool().execute_action(Action(name="do_nothing", arguments={}))
    assert isinstance(result, Observation)
    assert result.contents == [TextContent(data="Success")]


@pytest.mark.asyncio
async def test_async_tool_execute_action_unknown_method_raises():
    with pytest.raises(ValueError, match="does not exist"):
        await AsyncEchoTool().execute_action(Action(name="nonexistent", arguments={}))


@pytest.mark.asyncio
async def test_async_tool_execute_action_non_action_method_raises():
    with pytest.raises(ValueError, match="not decorated"):
        await AsyncEchoTool().execute_action(Action(name="not_an_action", arguments={}))


@pytest.mark.asyncio
async def test_async_tool_execute_action_exception_returns_step_error():
    result = await AsyncEchoTool().execute_action(Action(name="crash", arguments={}))
    assert isinstance(result, StepError)
    assert result.error_type == "RuntimeError"
    assert result.exception_str == "intentional error"


@pytest.mark.asyncio
async def test_async_tool_missing_required_arg_returns_observation() -> None:
    """Missing required argument → recoverable error Observation, not StepError."""
    result = await AsyncEchoTool().execute_action(Action(name="echo", arguments={}))
    assert isinstance(result, Observation)
    assert "Bad arguments" in result.contents[0].data
    assert "echo" in result.contents[0].data


@pytest.mark.asyncio
async def test_async_tool_unknown_kwarg_returns_observation() -> None:
    """Unknown keyword argument → recoverable error Observation, not StepError."""
    result = await AsyncEchoTool().execute_action(Action(name="echo", arguments={"text": "hi", "typo": "x"}))
    assert isinstance(result, Observation)
    assert "Bad arguments" in result.contents[0].data


def test_async_tool_rejects_sync_action_at_class_definition():
    """AsyncTool raises TypeError at class definition if a @tool_action method is sync."""
    with pytest.raises(TypeError, match="not async"):

        class _BadAsyncTool(AsyncTool):
            @tool_action
            def sync_action(self) -> str:
                """This should not be allowed."""
                return "oops"


# ── assert_tool_docstrings_valid ─────────────────────────────────────────────────────────────────


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
    with pytest.raises(AssertionError, match="A docstring is required to extract parameter information"):
        assert_tool_docstrings_valid(MissingFunctionDescriptionTool)


# ── Toolbox fixtures ───────────────────────────────────────────────────────────


class UpperTool(Tool):
    @tool_action
    def upper(self, text: str) -> str:
        """Return the text in uppercase."""
        return text.upper()


class AsyncUpperTool(AsyncTool):
    @tool_action
    async def upper(self, text: str) -> str:
        """Return the text in uppercase."""
        return text.upper()


# ── Toolbox (sync) ─────────────────────────────────────────────────────────────


def test_toolbox_action_set_is_union_of_tools():
    box = Toolbox(tools=[EchoTool(), UpperTool()])
    names = {a.name for a in box.action_set}
    assert names == {"echo", "add", "crash", "upper"}


def test_toolbox_execute_action_delegates_to_correct_tool():
    box = Toolbox(tools=[EchoTool(), UpperTool()])
    result = box.execute_action(Action(name="upper", arguments={"text": "hello"}))
    assert isinstance(result, Observation)
    assert result.contents == [TextContent(data="HELLO")]


def test_toolbox_execute_action_unknown_raises():
    box = Toolbox(tools=[EchoTool()])
    with pytest.raises(ValueError, match="not supported"):
        box.execute_action(Action(name="nonexistent", arguments={}))


def test_toolbox_duplicate_action_name_raises_on_construction():
    with pytest.raises(ValueError, match="Duplicate action name"):
        Toolbox(tools=[EchoTool(), EchoTool()])


def test_toolbox_find_tool_returns_correct_instance():
    echo = EchoTool()
    upper = UpperTool()
    box = Toolbox(tools=[echo, upper])
    assert box.find_tool(UpperTool) is upper


def test_toolbox_find_tool_returns_none_when_absent():
    box = Toolbox(tools=[EchoTool()])
    assert box.find_tool(UpperTool) is None


def test_toolbox_reset_calls_reset_on_all_tools():
    resets = 0

    class TrackingTool(Tool):
        def reset(self) -> None:
            nonlocal resets
            resets += 1

    Toolbox(tools=[TrackingTool(), TrackingTool()]).reset()
    assert resets == 2


def test_toolbox_close_calls_close_on_all_tools():
    closed = 0

    class TrackingTool(Tool):
        def close(self) -> None:
            nonlocal closed
            closed += 1

    Toolbox(tools=[TrackingTool(), TrackingTool()]).close()
    assert closed == 2


# ── AsyncToolbox ───────────────────────────────────────────────────────────────


def test_async_toolbox_action_set_is_union_of_tools():
    box = AsyncToolbox(tools=[AsyncEchoTool(), AsyncUpperTool()])
    names = {a.name for a in box.action_set}
    assert names == {"echo", "add", "crash", "upper"}


@pytest.mark.asyncio
async def test_async_toolbox_execute_action_delegates_to_correct_tool():
    box = AsyncToolbox(tools=[AsyncEchoTool(), AsyncUpperTool()])
    result = await box.execute_action(Action(name="upper", arguments={"text": "hello"}))
    assert isinstance(result, Observation)
    assert result.contents == [TextContent(data="HELLO")]


@pytest.mark.asyncio
async def test_async_toolbox_execute_action_unknown_raises():
    box = AsyncToolbox(tools=[AsyncEchoTool()])
    with pytest.raises(ValueError, match="not supported"):
        await box.execute_action(Action(name="nonexistent", arguments={}))


def test_async_toolbox_duplicate_action_name_raises_on_construction():
    with pytest.raises(ValueError, match="Duplicate action name"):
        AsyncToolbox(tools=[AsyncEchoTool(), AsyncEchoTool()])


def test_async_toolbox_find_tool_returns_correct_instance():
    echo = AsyncEchoTool()
    upper = AsyncUpperTool()
    box = AsyncToolbox(tools=[echo, upper])
    assert box.find_tool(AsyncUpperTool) is upper


def test_async_toolbox_find_tool_returns_none_when_absent():
    box = AsyncToolbox(tools=[AsyncEchoTool()])
    assert box.find_tool(AsyncUpperTool) is None


@pytest.mark.asyncio
async def test_async_toolbox_reset_calls_reset_on_all_tools():
    resets = 0

    class TrackingAsyncTool(AsyncTool):
        async def reset(self) -> None:
            nonlocal resets
            resets += 1

    await AsyncToolbox(tools=[TrackingAsyncTool(), TrackingAsyncTool()]).reset()
    assert resets == 2


@pytest.mark.asyncio
async def test_async_toolbox_close_calls_close_on_all_tools():
    closed = 0

    class TrackingAsyncTool(AsyncTool):
        async def close(self) -> None:
            nonlocal closed
            closed += 1

    await AsyncToolbox(tools=[TrackingAsyncTool(), TrackingAsyncTool()]).close()
    assert closed == 2
