"""
Tool configuration for CUBE benchmarks.

This module defines AbstractTool, Tool, AsyncTool, ToolConfig, and the @tool_action decorator
for implementing and configuring agent action interfaces. ToolConfig allows researchers
to swap tool implementations for experimentation, enabling research on different tool
sets and configurations without modifying benchmark code.

Abstract classes:
    AbstractTool — subclasses must implement:
        - execute_action(action: Action) -> Observation | StepError
        - action_set (property) -> list[ActionSchema]
    AbstractAsyncTool — same contract but fully async:
        - async execute_action(action: Action) -> Observation | StepError
        - action_set (property) -> list[ActionSchema]
    Tool is a concrete subclass of AbstractTool that implements both automatically
    via the @tool_action decorator — subclass Tool instead of AbstractTool directly.
    AsyncTool is a concrete subclass of AbstractAsyncTool — all @tool_action methods
    must be async def. A TypeError is raised at class definition time otherwise.

    ToolConfig — subclasses must implement:
        - make(container) -> AbstractTool | AbstractAsyncTool

Example — defining a custom sync tool and its config:

    from cube.tool import Tool, ToolConfig, tool_action
    from cube.container import Container

    class BrowserTool(Tool):
        base_url: str

        @tool_action
        def navigate(self, url: str) -> str:
            '''Navigate to a URL and return the page title.'''
            ...

        @tool_action
        def click(self, selector: str) -> str:
            '''Click on an element identified by a CSS selector.'''
            ...

    class BrowserToolConfig(ToolConfig):
        base_url: str = "http://localhost:9222"

        def make(self, container: Container | None = None) -> BrowserTool:
            url = container.get_url(port=9222) if container else self.base_url
            return BrowserTool(base_url=url)

Example — defining an async tool:

    class AsyncBrowserTool(AsyncTool):
        @tool_action
        async def navigate(self, url: str) -> str:
            '''Navigate to a URL and return the page title.'''
            ...

The BrowserToolConfig can then be passed to a Task or Benchmark, letting
harness users swap browser backends without touching benchmark logic.
"""

import asyncio
import concurrent.futures
import contextvars
import inspect
import logging
import threading
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, List

from cube.container import Container
from cube.core import Action, ActionSchema, Content, Observation, StepError, ValidatedConfig

logger = logging.getLogger(__name__)


class AbstractTool(ABC):
    """
    Abstract interface for objects that can react on a list of actions.
    List defined by the Protocol that tool inherits.
    """

    def reset(self) -> None:
        """Optional: reset the tool to its initial state."""
        pass

    def close(self) -> None:
        """Optional: clean up tool resources (connections, processes, files, etc.)."""
        pass

    @abstractmethod
    def execute_action(self, action: Action) -> Any:
        """Execute a single action and return the result."""
        pass

    async def async_execute_action(self, action: Action) -> Any:
        """Async-shaped facade over `execute_action`.

        Default: call sync `execute_action` directly on the current
        thread (no `to_thread` hop — the call is fully synchronous
        underneath, just packaged as a coroutine so async callers can
        `await` uniformly).

        Tools with truly async I/O should subclass `AbstractAsyncTool`
        (or `AsyncTool`) so their own async `execute_action` becomes
        the canonical implementation; `async_execute_action` is then
        the unified call-site for any caller that wants an awaitable,
        regardless of whether the inner tool is sync or async.
        """
        return self.execute_action(action)

    @property
    @abstractmethod
    def action_set(self) -> List[ActionSchema]:
        """
        Returns list of actions supported by that tool.
        Tool definitions in litellm-compatible format.

        Returns a JSON-serializable list of tool descriptors, each with:
        - type: "function"
        - function: {name, description, parameters (JSON Schema)}

        This format is compatible with litellm/OpenAI function calling.
        Agents use this to discover available actions without knowing
        tool implementations in advance.

        Example return value:
        [
            {
                "type": "function",
                "function": {
                    "name": "click",
                    "description": "Click on a web element",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "selector": {"type": "string", "description": "CSS selector"}
                        },
                        "required": ["selector"]
                    }
                }
            }
        ]
        """
        pass


class AbstractAsyncTool(AbstractTool):
    """Deprecated. Subclass `AbstractTool` directly.

    `AbstractTool` now declares both `execute_action` (sync) and
    `async_execute_action` (async) — no class-level split is needed.

    This subclass preserves the async-`execute_action` shape so
    legacy `class Foo(AbstractAsyncTool): async def execute_action(...)`
    code keeps working through one release window. Subclassing emits a
    `DeprecationWarning`.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Skip the warning when our own deprecated shim (`AsyncTool`) is the
        # immediate subclass — its own __init_subclass__ already emits one,
        # and we don't want the warning to fire at module import time.
        if cls.__module__ == __name__ and cls.__name__ == "AsyncTool":
            return
        warnings.warn(
            f"Subclassing AbstractAsyncTool is deprecated (see {cls.__name__}); "
            f"subclass AbstractTool directly. AbstractTool now declares both "
            f"`execute_action` (sync) and `async_execute_action` (async).",
            DeprecationWarning,
            stacklevel=2,
        )

    @abstractmethod
    async def execute_action(self, action: Action) -> Any:  # type: ignore[override]
        """Legacy async execute_action. Subclasses MUST override."""

    async def async_execute_action(self, action: Action) -> Any:
        """Delegate to the legacy async `execute_action`."""
        return await self.execute_action(action)


class ToolConfig(ValidatedConfig, ABC):
    """
    Configuration for creating task-specific tools.

    ToolConfig enables research on tool variability by allowing researchers to:
    - Swap out different tool implementations (e.g., Playwright vs Selenium)
    - Provide different tool sets (e.g., basic vs advanced browser tools)
    - Configure tool behavior (e.g., browser types, shell environments)
    """

    @abstractmethod
    def make(self, container: Container | None = None) -> AbstractTool:
        """
        Instantiate Tool from configuration data.

        Args:
            container: The launched container for this task, if any. Use it to
                       extract connection info (host, ports) to configure the
                       tool's endpoint. None if the task needs no container.

        Returns:
            AbstractTool instance
        """
        pass


class AsyncToolConfig(ValidatedConfig, ABC):
    """Configuration for creating async task-specific tools.

    Mirrors ToolConfig but make() is a coroutine, allowing async resource
    acquisition (browser launch, network connections, etc.) before the tool
    is handed to the caller.
    """

    @abstractmethod
    async def make(self, container: Container | None = None) -> AbstractAsyncTool:
        """Instantiate AsyncTool from configuration data."""
        pass


def tool_action(func: Callable) -> Callable:
    """
    Decorator to mark a method as an action in a Tool or AsyncTool.

    This decorator automatically registers methods as actions that will be
    discovered by the tool's action_set property.

    For AsyncTool subclasses, the decorated method must be async def —
    a TypeError is raised at class definition time otherwise.

    Usage:
        class MyTool(Tool):
            @tool_action
            def my_action(self, param: str) -> str:
                '''Action description.'''
                return f"Result: {param}"

        class MyAsyncTool(AsyncTool):
            @tool_action
            async def my_action(self, param: str) -> str:
                '''Action description.'''
                return f"Result: {param}"
    """
    func._is_action = True  # type: ignore[attr-defined]
    return func


class _ToolActionsMixin:
    """
    Shared action discovery and dispatch logic for Tool and AsyncTool.

    Not intended to be subclassed directly.
    """

    def get_action_method(self, action: Action) -> Callable:
        """Return the bound method for an action, or raise ValueError if it is not registered.

        Raises distinct errors for:
        - Method that does not exist on the class at all.
        - Method that exists but is not decorated with @tool_action.
        """
        # Check instance dict first — catches dynamically attached actions (not in any class dict)
        method = self.__dict__.get(action.name)
        if method and callable(method) and getattr(method, "_is_action", False):
            return method
        method = getattr(self, action.name, None)
        if not method:
            raise ValueError(f"Action {action.name} does not exist in {self.__class__.__name__}.")
        is_registered = any(
            getattr(cls.__dict__.get(action.name), "_is_action", False)
            for cls in type(self).__mro__
            if action.name in cls.__dict__
        )
        if not is_registered:
            raise ValueError(
                f"Action {action.name} exists in {self.__class__.__name__} but is not decorated with @tool_action. Add @tool_action to expose it as an action."
            )
        return method

    def _validate_action_args(self, action: Action, method: Callable) -> Observation | None:
        """Pre-validate action.arguments against the bound method's signature.

        Returns an error Observation when the agent passed unknown / mismatched
        kwargs (so the agent can correct itself on the next step), or None when
        the args bind cleanly.

        This is necessary because raising would surface as a fatal StepError
        upstream (Task.step terminates the episode), and a single LLM-side typo
        like ``write_file(path=…, content=…, timeout=120)`` shouldn't end an
        episode — the agent should see the mistake and retry.
        """
        try:
            inspect.signature(method).bind(**action.arguments)
            return None
        except TypeError as e:
            params = list(inspect.signature(method).parameters)
            msg = f"Invalid arguments for {action.name}: {e}. Expected parameters: {params}."
            return Observation(contents=[Content.from_data(msg, tool_call_id=action.id)])

    @property
    def action_set(self) -> List[ActionSchema]:
        """Automatically discover all methods marked with @tool_action decorator."""
        actions = []

        # Introspect the class to find all methods marked as actions
        for attr_name in dir(self):
            # Skip private/protected methods and the action_set property itself
            if attr_name.startswith("_") or attr_name == "action_set":
                continue

            # Skip properties — calling getattr on a property invokes its getter,
            # which may have side effects (e.g. raising if a resource is not yet initialized).
            if any(
                isinstance(cls.__dict__.get(attr_name), property)
                for cls in type(self).__mro__
                if attr_name in cls.__dict__
            ):
                continue

            attr = getattr(self, attr_name)

            # Check if this attr_name is a method marked as an action.
            # We walk up the class hierarchy (method resolution order, MRO)
            # because a subclass may override a method without repeating
            # @tool_action - as long as the decorator appears on the method
            # in any parent class, the override is still treated as an action.
            is_action = any(
                getattr(cls.__dict__.get(attr_name), "_is_action", False)
                for cls in type(self).__mro__
                if attr_name in cls.__dict__
            )
            if callable(attr) and is_action:
                actions.append(ActionSchema.from_function(attr))

        # Also discover instance-level actions attached via setattr (not in any class dict)
        for name, attr in self.__dict__.items():
            if not name.startswith("_") and callable(attr) and getattr(attr, "_is_action", False):
                actions.append(ActionSchema.from_function(attr))

        return actions


class Tool(_ToolActionsMixin, AbstractTool):
    """
    Base class for tools with automatic action discovery via decorators.

    `@tool_action` methods can be **sync OR async** on the same class —
    dispatch routes per the method's `def` keyword. The tool exposes a
    dual call surface so callers can pick the shape that matches their
    own call-site:

      * `tool.execute_action(action)` — sync caller. Sync actions run
        directly (no overhead). Async actions are bridged via a
        one-shot thread with its own event loop (~2-5 ms / call).

      * `await tool.async_execute_action(action)` — async caller. Async
        actions are awaited directly. Sync actions hop through
        `asyncio.to_thread` (pooled worker, enables real OS-thread
        parallelism inside `asyncio.gather`).

    Tools that wrap thread-affine resources (sync Playwright, some DB
    drivers) should override `async_execute_action` to dispatch through
    a per-instance single-threaded executor.

    Example:
        ```python
        from cube.tool import Tool, tool_action, Action

        class CalculatorTool(Tool):
            @tool_action
            def add(self, a: float, b: float) -> str:
                return f"Result: {a + b}"

            @tool_action
            async def slow_add(self, a: float, b: float) -> str:
                await some_io()
                return f"Result: {a + b}"

        calc = CalculatorTool()
        # Sync caller, sync action: direct call.
        calc.execute_action(Action(name="add", arguments={"a": 5, "b": 3}))
        # Sync caller, async action: bridge via thread+loop.
        calc.execute_action(Action(name="slow_add", arguments={"a": 5, "b": 3}))
        # Async caller, both action kinds: works through async_execute_action.
        await calc.async_execute_action(Action(name="slow_add", arguments={"a": 5, "b": 3}))
        ```
    """

    def execute_action(self, action: Action) -> Observation | StepError:
        """Execute an action from a sync call-site.

        Sync action → direct call.
        Async action → bridge via one-shot daemon thread + new event loop
        (~2-5 ms overhead; `contextvars.copy_context()` propagates the
        caller's tracing/OTel context into the worker).
        """
        method = self.get_action_method(action)
        invalid = self._validate_action_args(action, method)
        if invalid is not None:
            return invalid
        if inspect.iscoroutinefunction(method):
            return self._bridge_async_to_sync(action, method)
        return self._dispatch_sync(action, method)

    async def async_execute_action(self, action: Action) -> Observation | StepError:
        """Execute an action from an async call-site.

        Async action → direct await.
        Sync action → `asyncio.to_thread(method)` — pooled worker, no
        per-call thread spawn, enables real OS-thread parallelism when
        wrapped in `asyncio.gather`. Override this method to route
        through a per-instance single-threaded executor for tools that
        wrap thread-affine resources.
        """
        method = self.get_action_method(action)
        invalid = self._validate_action_args(action, method)
        if invalid is not None:
            return invalid
        try:
            if inspect.iscoroutinefunction(method):
                action_result = (await method(**action.arguments)) or "Success"
            else:
                action_result = (await asyncio.to_thread(method, **action.arguments)) or "Success"
        except Exception as e:
            action_result = f"Error executing action {action.name}: {e}"
            logger.exception(action_result)
            return StepError.from_exception(e)
        return Observation(contents=[Content.from_data(action_result, tool_call_id=action.id)])

    # ── Internals ──

    def _dispatch_sync(self, action: Action, method: Callable) -> Observation | StepError:
        """Execute a sync `@tool_action` method directly, with the standard
        success/error wrapping. Shared between `execute_action` (sync) and
        the bridge runner."""
        try:
            action_result = method(**action.arguments) or "Success"
        except Exception as e:
            action_result = f"Error executing action {action.name}: {e}"
            logger.exception(action_result)
            return StepError.from_exception(e)
        return Observation(contents=[Content.from_data(action_result, tool_call_id=action.id)])

    def _bridge_async_to_sync(self, action: Action, method: Callable) -> Observation | StepError:
        """Run an async `@tool_action` from a sync call-site.

        Implementation: spawn a daemon thread, run a new event loop in
        it, block on the result. ~2-5 ms overhead per call. The
        thread is one-shot (not pooled). `contextvars.copy_context()`
        propagates the caller's context (OTel spans, logging state)
        into the worker so cross-thread observability is preserved.
        """
        ctx = contextvars.copy_context()
        fut: concurrent.futures.Future = concurrent.futures.Future()

        def runner() -> None:
            try:
                fut.set_result(ctx.run(asyncio.run, method(**action.arguments)))
            except Exception as e:
                fut.set_exception(e)

        threading.Thread(target=runner, daemon=True).start()
        try:
            action_result = fut.result() or "Success"
        except Exception as e:
            action_result = f"Error executing action {action.name}: {e}"
            logger.exception(action_result)
            return StepError.from_exception(e)
        return Observation(contents=[Content.from_data(action_result, tool_call_id=action.id)])


class AsyncTool(Tool, AbstractAsyncTool):
    """Deprecated alias of `Tool`. Subclass `Tool` directly.

    `Tool` now supports `@tool_action` methods of either sync or async
    kind on the same class — see `Tool`'s docstring.

    This subclass preserves the async `execute_action` interface so
    legacy `await tool.execute_action(action)` callers keep working
    through the deprecation window. Multiple-inherits `AbstractAsyncTool`
    so `isinstance(x, AbstractAsyncTool)` checks on subclasses keep
    returning True. Subclassing `AsyncTool` emits a `DeprecationWarning`.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        warnings.warn(
            f"Subclassing AsyncTool is deprecated (see {cls.__name__}); "
            f"subclass Tool directly. Tool now supports both sync and "
            f"async @tool_action methods on the same class.",
            DeprecationWarning,
            stacklevel=2,
        )

    async def execute_action(self, action: Action) -> Observation | StepError:  # type: ignore[override]
        """Legacy async dispatch — delegates to `Tool.async_execute_action`."""
        return await self.async_execute_action(action)


class Toolbox(Tool):
    """Composite tool that holds a list of `Tool` instances and routes
    actions by name. Both sync and async dispatch supported — leaf
    Tools handle the per-method bridging.

    `@tool_action` methods on member tools may be of either kind
    (sync or async). The toolbox is transparent: `execute_action`
    and `async_execute_action` delegate to the matching leaf's
    same-named dispatch method.
    """

    def __init__(self, tools: list[AbstractTool]):
        self.tools = tools
        self._action_name_to_tool: dict[str, AbstractTool] = {}
        for tool in tools:
            for action in tool.action_set:
                if action.name in self._action_name_to_tool:
                    previous_tool_name = self._action_name_to_tool[action.name].__class__.__name__
                    this_tool_name = tool.__class__.__name__
                    raise ValueError(
                        f"Duplicate action name '{action.name}' found in multiple tools ({previous_tool_name} and {this_tool_name}). Action names must be unique across all tools in the toolbox."
                    )
                self._action_name_to_tool[action.name] = tool

    @property
    def action_set(self) -> list[ActionSchema]:
        """Returns the union of all action sets across contained tools."""
        return [action for tool in self.tools for action in tool.action_set]

    def find_tool(self, tool_cls: type) -> AbstractTool | None:
        """Find a tool of the given class in the toolbox."""
        for tool in self.tools:
            if isinstance(tool, tool_cls):
                return tool
        return None

    def reset(self) -> None:
        """Sync reset. If a leaf's `reset` returns a coroutine (legacy
        `AsyncTool` shim), the coroutine is closed without awaiting —
        call `async_reset` from an async context for proper cleanup."""
        for tool in self.tools:
            r = tool.reset()
            if inspect.iscoroutine(r):
                r.close()

    async def async_reset(self) -> None:
        """Async reset. Awaits coroutine returns from any leaf."""
        for tool in self.tools:
            r = tool.reset()
            if inspect.iscoroutine(r):
                await r

    def close(self) -> None:
        """Sync close. Same coroutine-handling as `reset`."""
        for tool in self.tools:
            c = tool.close()
            if inspect.iscoroutine(c):
                c.close()

    async def async_close(self) -> None:
        """Async close. Awaits coroutine returns from any leaf."""
        for tool in self.tools:
            c = tool.close()
            if inspect.iscoroutine(c):
                await c

    def execute_action(self, action: Action) -> Observation | StepError:
        """Sync dispatch — delegates to leaf's `execute_action`.
        Bridging for async actions happens inside the leaf."""
        if action.name not in self._action_name_to_tool:
            raise ValueError(f"Action '{action.name}' is not supported by any tool in the toolbox.")
        return self._action_name_to_tool[action.name].execute_action(action)

    async def async_execute_action(self, action: Action) -> Observation | StepError:
        """Async dispatch — delegates to leaf's `async_execute_action`.
        Sync leaves hop through `asyncio.to_thread` automatically."""
        if action.name not in self._action_name_to_tool:
            raise ValueError(f"Action '{action.name}' is not supported by any tool in the toolbox.")
        return await self._action_name_to_tool[action.name].async_execute_action(action)


class AsyncToolbox(Toolbox):
    """Deprecated. Use `Toolbox` directly (its `async_execute_action`
    is the canonical async call-site).

    Kept as a thin shim that preserves the async-`execute_action`
    semantic for legacy `await tb.execute_action(action)` callers.
    Each call emits a `DeprecationWarning`.

    `reset()` and `close()` are also async on this shim, mirroring the
    pre-consolidation contract.
    """

    async def execute_action(self, action: Action) -> Observation | StepError:  # type: ignore[override]
        warnings.warn(
            "AsyncToolbox.execute_action is deprecated; use Toolbox.async_execute_action.",
            DeprecationWarning,
            stacklevel=2,
        )
        return await self.async_execute_action(action)

    async def reset(self) -> None:  # type: ignore[override]
        await self.async_reset()

    async def close(self) -> None:  # type: ignore[override]
        await self.async_close()


class ToolboxConfig(ToolConfig):
    """Configuration for a list of tools (sync only)."""

    tool_configs: list[ToolConfig] = []

    def make(self, container: Container | None = None) -> Toolbox:
        tools: list[AbstractTool] = []
        for tc in self.tool_configs:
            result = tc.make(container)
            if isinstance(result, Toolbox):
                tools.extend(result.tools)
            else:
                tools.append(result)
        return Toolbox(tools=tools)


class AsyncToolboxConfig(AsyncToolConfig):
    """Configuration for a list of async only tools."""

    tool_configs: list[AsyncToolConfig] = []

    async def make(self, container: Container | None = None) -> AsyncToolbox:
        tools = [await tc.make(container) for tc in self.tool_configs]
        return AsyncToolbox(tools=tools)
