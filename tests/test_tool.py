"""Tests for cube.tool — Tool, Toolbox, ToolConfig, tool_action."""

import asyncio
import inspect

import pytest

from cube.core import Action, ActionSchema, AgentStop, Observation, TextContent
from cube.tool import Tool, Toolbox, tool_action


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


class AsyncEchoTool(Tool):
    """A `Tool` whose `@tool_action` methods are all `async def`."""

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
    assert action_names == {"echo", "add", "crash", "final_step"}  # final_step is universal


def test_tool_final_step_is_universal_and_raises_agent_stop():
    # final_step is a real @tool_action on the Tool base — every tool exposes it, and
    # executing it raises AgentStop (no STOP special-casing in the dispatch path).
    assert "final_step" in {a.name for a in EchoTool().action_set}
    with pytest.raises(AgentStop):
        EchoTool().execute_action(Action(name="final_step", arguments={}))


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


def test_tool_execute_action_exception_folds_into_observation():
    # Errors-as-observations: a tool exception folds into the returned Observation — the
    # structured StepError on obs.error, and the error text in contents (the agent reads
    # it and retries; non-terminal).
    result = EchoTool().execute_action(Action(name="crash", arguments={}))
    assert isinstance(result, Observation)
    assert result.error is not None
    assert result.error.error_type == "RuntimeError"
    assert result.error.exception_str == "intentional error"
    assert "intentional error" in result.to_markdown()


def test_tool_execute_action_unknown_kwarg_returns_observation():
    """LLM-side typos like an extra `timeout=` should not kill the episode."""
    result = EchoTool().execute_action(Action(name="echo", arguments={"text": "hi", "timeout": 120}))
    assert isinstance(result, Observation)
    assert "Invalid arguments for echo" in result.contents[0].data
    assert "['text']" in result.contents[0].data


def test_tool_execute_action_missing_required_returns_observation():
    """Missing a required argument is also a recoverable, agent-correctable error."""
    result = EchoTool().execute_action(Action(name="add", arguments={"a": 1.0}))
    assert isinstance(result, Observation)
    assert "Invalid arguments for add" in result.contents[0].data


# ── Tool with async @tool_action methods ──────────────────────────────────────


def test_async_tool_action_decorator_sets_flag():
    tool = AsyncEchoTool()
    assert getattr(tool.echo, "_is_action", False) is True
    assert getattr(tool.not_an_action, "_is_action", False) is False


def test_async_tool_action_set_discovers_only_decorated_methods():
    action_names = {a.name for a in AsyncEchoTool().action_set}
    assert action_names == {"echo", "add", "crash", "final_step"}  # final_step is universal


@pytest.mark.asyncio
async def test_async_tool_final_step_raises_agent_stop():
    with pytest.raises(AgentStop):
        await AsyncEchoTool().async_execute_action(Action(name="final_step", arguments={}))


def test_sync_bridge_forwards_agent_stop_from_async_action():
    # An async @tool_action that raises AgentStop, executed from a SYNC call-site, must
    # propagate AgentStop through the thread-bridge — NOT deadlock. Regression: the bridge
    # runner caught only Exception, so the AgentStop (a BaseException) was dropped and
    # `fut.result()` blocked forever.
    class _AsyncStopTool(Tool):
        @tool_action
        async def stop_async(self) -> str:
            """Async action that ends the episode."""
            raise AgentStop()

    with pytest.raises(AgentStop):
        _AsyncStopTool().execute_action(Action(name="stop_async", arguments={}))


def test_async_tool_action_set_schemas_have_descriptions():
    for schema in AsyncEchoTool().action_set:
        assert schema.description != ""


@pytest.mark.asyncio
async def test_async_tool_execute_action_returns_observation():
    result = await AsyncEchoTool().async_execute_action(Action(name="echo", arguments={"text": "hello"}))
    assert isinstance(result, Observation)
    assert result.contents == [TextContent(data="hello")]


@pytest.mark.asyncio
async def test_async_tool_execute_action_none_returns_success():
    """An async @tool_action returning None should yield an Observation with 'Success'."""

    class SilentAsyncTool(Tool):
        @tool_action
        async def do_nothing(self) -> None:
            """Does nothing."""

    result = await SilentAsyncTool().async_execute_action(Action(name="do_nothing", arguments={}))
    assert isinstance(result, Observation)
    assert result.contents == [TextContent(data="Success")]


@pytest.mark.asyncio
async def test_async_tool_execute_action_unknown_method_raises():
    with pytest.raises(ValueError, match="does not exist"):
        await AsyncEchoTool().async_execute_action(Action(name="nonexistent", arguments={}))


@pytest.mark.asyncio
async def test_async_tool_execute_action_non_action_method_raises():
    with pytest.raises(ValueError, match="not decorated"):
        await AsyncEchoTool().async_execute_action(Action(name="not_an_action", arguments={}))


@pytest.mark.asyncio
async def test_async_tool_execute_action_exception_folds_into_observation():
    result = await AsyncEchoTool().async_execute_action(Action(name="crash", arguments={}))
    assert isinstance(result, Observation)
    assert result.error is not None
    assert result.error.error_type == "RuntimeError"
    assert result.error.exception_str == "intentional error"


@pytest.mark.asyncio
async def test_async_tool_execute_action_unknown_kwarg_returns_observation():
    result = await AsyncEchoTool().async_execute_action(Action(name="echo", arguments={"text": "hi", "timeout": 120}))
    assert isinstance(result, Observation)
    assert "Invalid arguments for echo" in result.contents[0].data


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


class AsyncUpperTool(Tool):
    @tool_action
    async def upper(self, text: str) -> str:
        """Return the text in uppercase."""
        return text.upper()


# ── Toolbox — sync dispatch ────────────────────────────────────────────────────


def test_toolbox_action_set_is_union_of_tools():
    box = Toolbox(tools=[EchoTool(), UpperTool()])
    names = [a.name for a in box.action_set]
    assert set(names) == {"echo", "add", "crash", "upper", "final_step"}
    assert names.count("final_step") == 1  # the universal final_step dedups to one


def test_toolbox_execute_action_delegates_to_correct_tool():
    box = Toolbox(tools=[EchoTool(), UpperTool()])
    result = box.execute_action(Action(name="upper", arguments={"text": "hello"}))
    assert isinstance(result, Observation)
    assert result.contents == [TextContent(data="HELLO")]


def test_toolbox_execute_action_unknown_raises():
    box = Toolbox(tools=[EchoTool()])
    with pytest.raises(ValueError, match="not supported"):
        box.execute_action(Action(name="nonexistent", arguments={}))


def test_toolbox_identical_actions_dedup_on_construction():
    # Two tools sharing identical actions (here every action, incl. the universal final_step)
    # dedup to the first owner — NOT an error. This is what makes inherited final_step work
    # across composed tools.
    box = Toolbox(tools=[EchoTool(), EchoTool()])
    names = [a.name for a in box.action_set]
    assert sorted(set(names)) == ["add", "crash", "echo", "final_step"]
    assert len(names) == len(set(names))  # no duplicates


def test_toolbox_conflicting_action_schema_raises_on_construction():
    # Same action name, DIFFERENT schema across tools is a real conflict.
    class EchoIntTool(Tool):
        @tool_action
        def echo(self, count: int) -> str:
            """A different 'echo' — incompatible schema."""
            return str(count)

    with pytest.raises(ValueError, match="Conflicting action 'echo'"):
        Toolbox(tools=[EchoTool(), EchoIntTool()])


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


# ── Toolbox — async dispatch over mixed leaves ────────────────────────────────


@pytest.mark.asyncio
async def test_toolbox_async_execute_action_routes_async_leaf():
    box = Toolbox(tools=[AsyncEchoTool(), AsyncUpperTool()])
    result = await box.async_execute_action(Action(name="upper", arguments={"text": "hello"}))
    assert isinstance(result, Observation)
    assert result.contents == [TextContent(data="HELLO")]


@pytest.mark.asyncio
async def test_toolbox_async_execute_action_unknown_raises():
    box = Toolbox(tools=[AsyncEchoTool()])
    with pytest.raises(ValueError, match="not supported"):
        await box.async_execute_action(Action(name="nonexistent", arguments={}))


@pytest.mark.asyncio
async def test_toolbox_async_execute_action_handles_mixed_sync_and_async_leaves():
    """A single `Toolbox` may hold a mix of sync and async leaves;
    `async_execute_action` routes per the leaf's action method's kind.
    """
    box = Toolbox(tools=[EchoTool(), AsyncUpperTool()])
    sync_result = await box.async_execute_action(Action(name="echo", arguments={"text": "hello"}))
    assert isinstance(sync_result, Observation)
    assert sync_result.contents[0].data == "hello"
    async_result = await box.async_execute_action(Action(name="upper", arguments={"text": "hi"}))
    assert isinstance(async_result, Observation)
    assert async_result.contents[0].data == "HI"


@pytest.mark.asyncio
async def test_async_execute_action_default_on_sync_tool_runs_sync_body():
    """`AbstractTool.async_execute_action` default delegates to sync
    `execute_action` directly — no thread hop. Sync tools are usable
    from async call-sites uniformly."""
    tool = EchoTool()
    action = Action(name="echo", arguments={"text": "hi"})
    result = await tool.async_execute_action(action)
    assert isinstance(result, Observation)
    assert result.contents[0].data == "hi"


# ── Tool consolidation — 4-cell dispatch matrix ───────────────────────────────


class MixedActionTool(Tool):
    """Tool with both sync and async @tool_action methods on one class."""

    @tool_action
    def fast_sync(self, x: int) -> str:
        """Sync action."""
        return f"sync:{x * 2}"

    @tool_action
    async def slow_async(self, x: int) -> str:
        """Async action."""
        return f"async:{x * 2}"


def test_sync_caller_sync_action_direct():
    """Cell (sync caller, sync action): no overhead, direct dispatch."""
    tool = MixedActionTool()
    result = tool.execute_action(Action(name="fast_sync", arguments={"x": 5}))
    assert isinstance(result, Observation)
    assert result.contents[0].data == "sync:10"


def test_sync_caller_async_action_bridge():
    """Cell (sync caller, async action): bridge via thread+loop.

    Returns the awaited value through a `concurrent.futures.Future`.
    Slow but correct.
    """
    tool = MixedActionTool()
    result = tool.execute_action(Action(name="slow_async", arguments={"x": 5}))
    assert isinstance(result, Observation)
    assert result.contents[0].data == "async:10"


def test_sync_caller_async_action_bridge_inside_running_loop():
    """The bridge MUST work even when the sync caller is itself running
    inside an event loop (Agent._run inside Episode's asyncio.run is the
    motivating use case). Verifies the bridge doesn't try `asyncio.run`
    on the current thread."""

    async def harness():
        tool = MixedActionTool()
        return tool.execute_action(Action(name="slow_async", arguments={"x": 7}))

    result = asyncio.run(harness())
    assert isinstance(result, Observation)
    assert result.contents[0].data == "async:14"


@pytest.mark.asyncio
async def test_async_caller_sync_action_to_thread():
    """Cell (async caller, sync action): hops to_thread.

    Pooled worker enables real parallelism inside `asyncio.gather`.
    Result is returned through the awaitable.
    """
    tool = MixedActionTool()
    result = await tool.async_execute_action(Action(name="fast_sync", arguments={"x": 3}))
    assert isinstance(result, Observation)
    assert result.contents[0].data == "sync:6"


@pytest.mark.asyncio
async def test_async_caller_async_action_direct():
    """Cell (async caller, async action): direct await, no thread hop."""
    tool = MixedActionTool()
    result = await tool.async_execute_action(Action(name="slow_async", arguments={"x": 3}))
    assert isinstance(result, Observation)
    assert result.contents[0].data == "async:6"


@pytest.mark.asyncio
async def test_async_caller_parallel_gather_over_sync_actions():
    """Real parallelism: N concurrent sync actions via gather + to_thread.

    Wall-clock should be roughly max(latency_i), not sum(latency_i).
    """
    import time

    class SleepyTool(Tool):
        @tool_action
        def sleep_a_bit(self, ms: int) -> str:
            """Sleep `ms` milliseconds then return."""
            time.sleep(ms / 1000)
            return f"slept-{ms}"

    tool = SleepyTool()
    actions = [Action(name="sleep_a_bit", arguments={"ms": 100}) for _ in range(4)]
    start = time.time()
    results = await asyncio.gather(*[tool.async_execute_action(a) for a in actions])
    elapsed = time.time() - start
    # 4 × 100ms sequential = 400ms. Parallel should be well under 250ms.
    assert elapsed < 0.25, f"parallel-gather over sync actions ran sequentially? elapsed={elapsed:.3f}s"
    assert all(isinstance(r, Observation) for r in results)
