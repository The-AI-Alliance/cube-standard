"""
Tool configuration for CUBE benchmarks.

This module defines `AbstractTool`, `Tool`, `ToolConfig`, `Toolbox`, and the
`@tool_action` decorator. `ToolConfig` lets researchers swap tool
implementations for experimentation without modifying benchmark code.

`Tool` carries a dual call surface — `execute_action` (sync) and
`async_execute_action` (async) — and routes per the `@tool_action` method's
own `def` keyword. Sync and async methods may coexist on one class; the
framework bridges the impedance mismatch when caller and method differ.

Abstract classes:
    AbstractTool — subclasses must implement:
        - execute_action(action: Action) -> Observation
        - action_set (property) -> list[ActionSchema]
      and inherit a default `async_execute_action` that delegates to
      `execute_action`. Override it for async-native dispatch.

    ToolConfig — subclasses must implement:
        - make(container) -> AbstractTool

Example — defining a tool and its config:

    from cube.tool import Tool, ToolConfig, tool_action
    from cube.container import Container

    class BrowserTool(Tool):
        base_url: str

        @tool_action
        def screenshot(self) -> bytes:
            '''Capture a screenshot of the current page.'''
            ...

        @tool_action
        async def navigate(self, url: str) -> str:
            '''Navigate to a URL and return the page title.'''
            ...

    class BrowserToolConfig(ToolConfig):
        base_url: str = "http://localhost:9222"

        def make(self, container: Container | None = None) -> BrowserTool:
            url = container.get_url(port=9222) if container else self.base_url
            return BrowserTool(base_url=url)

The BrowserToolConfig can then be passed to a Task or Benchmark, letting
harness users swap browser backends without touching benchmark logic.
"""

import asyncio
import concurrent.futures
import contextvars
import inspect
import logging
import threading
from abc import ABC, abstractmethod
from typing import Any, Callable, List

from cube.container import Container
from cube.core import Action, ActionSchema, AgentStop, Content, Observation, StepError, ValidatedConfig

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
        """Async call-site for executing an action.

        Default: delegate to sync `execute_action` on the current thread
        (no thread hop). Subclasses with native async dispatch — most
        commonly `Tool`, which routes per the action method's own
        `def`/`async def` keyword — override this method.
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
    """Configuration for creating task-specific tools whose `make()`
    needs to be a coroutine (async resource acquisition: browser launch,
    network connections, etc.) before the tool is handed to the caller.
    """

    @abstractmethod
    async def make(self, container: Container | None = None) -> AbstractTool:
        """Instantiate a Tool from configuration data."""
        pass


def tool_action(func: Callable) -> Callable:
    """
    Decorator to mark a method as an action on a Tool.

    Decorated methods are discovered by the tool's `action_set` property.
    Methods may be sync `def` or `async def` — `Tool` dispatches per the
    method's own keyword.

    Usage:
        class MyTool(Tool):
            @tool_action
            def my_action(self, param: str) -> str:
                '''Action description.'''
                return f"Result: {param}"

            @tool_action
            async def my_slow_action(self, param: str) -> str:
                '''Async action on the same class.'''
                await some_io()
                return f"Result: {param}"
    """
    func._is_action = True  # type: ignore[attr-defined]
    return func


class _ToolActionsMixin:
    """
    Shared action discovery logic for `Tool`. Not intended to be subclassed
    directly.
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

    @tool_action
    def final_step(self) -> str:
        """Stop the task execution."""
        # STOP is just an action that raises — no special-casing in the dispatch path.
        # AgentStop is a BaseException, so the dispatch's `except Exception` lets it through.
        raise AgentStop()

    def execute_action(self, action: Action) -> Observation:
        """Execute an action from a sync call-site. Always returns an Observation: a tool
        error is folded into it (non-terminal, structured error on ``obs.error``);
        ``final_step`` raises :class:`AgentStop` (a BaseException, so it propagates here).

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

    async def async_execute_action(self, action: Action) -> Observation:
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
            return StepError.from_exception(e).to_observation()
        return Observation(contents=[Content.from_data(action_result, tool_call_id=action.id)])

    # ── Internals ──

    def _dispatch_sync(self, action: Action, method: Callable) -> Observation:
        """Execute a sync `@tool_action` method directly, with the standard
        success/error wrapping. Shared between `execute_action` (sync) and
        the bridge runner."""
        try:
            action_result = method(**action.arguments) or "Success"
        except Exception as e:
            action_result = f"Error executing action {action.name}: {e}"
            logger.exception(action_result)
            return StepError.from_exception(e).to_observation()
        return Observation(contents=[Content.from_data(action_result, tool_call_id=action.id)])

    def _bridge_async_to_sync(self, action: Action, method: Callable) -> Observation:
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
            return StepError.from_exception(e).to_observation()
        return Observation(contents=[Content.from_data(action_result, tool_call_id=action.id)])


def _dedup_actions(tools: list["AbstractTool"]) -> dict[str, "AbstractTool"]:
    """Map action name -> owning tool, deduping identical actions. Same name with an
    *identical* schema (e.g. the universal ``final_step`` every Tool inherits) dedups to the
    first owner; same name with a *different* schema is a real conflict and raises."""
    name_to_tool: dict[str, AbstractTool] = {}
    name_to_schema: dict[str, ActionSchema] = {}
    for tool in tools:
        for action in tool.action_set:
            existing = name_to_schema.get(action.name)
            if existing is not None:
                if existing == action:
                    continue  # identical -> dedup to the first owner
                raise ValueError(
                    f"Conflicting action '{action.name}' across tools "
                    f"({name_to_tool[action.name].__class__.__name__} and {tool.__class__.__name__}): "
                    "same name, different schema. Action names must be unique unless identical."
                )
            name_to_schema[action.name] = action
            name_to_tool[action.name] = tool
    return name_to_tool


def _deduped_action_set(tools: list["AbstractTool"]) -> list[ActionSchema]:
    """Union of all leaves' action sets, deduped by name (first wins). Conflicts are already
    rejected by :func:`_dedup_actions` at construction, so first-wins is safe here."""
    seen: set[str] = set()
    actions: list[ActionSchema] = []
    for tool in tools:
        for action in tool.action_set:
            if action.name not in seen:
                seen.add(action.name)
                actions.append(action)
    return actions


class Toolbox(Tool):
    """Composite tool that holds a list of `Tool` instances and routes
    actions by name. Both sync and async dispatch supported — leaf
    Tools handle the per-method bridging.

    `@tool_action` methods on member tools may be of either kind
    (sync or async). The toolbox is transparent: `execute_action`
    and `async_execute_action` delegate to the matching leaf's
    same-named dispatch method.

    Identical actions across leaves dedup (e.g. the ``final_step`` every Tool
    inherits); same name with a different schema raises (see `_dedup_actions`).
    """

    def __init__(self, tools: list[AbstractTool]):
        self.tools = tools
        self._action_name_to_tool = _dedup_actions(tools)

    @property
    def action_set(self) -> list[ActionSchema]:
        """Union of all leaves' action sets, deduped (identical actions like the inherited
        ``final_step`` collapse to one)."""
        return _deduped_action_set(self.tools)

    def find_tool(self, tool_cls: type) -> AbstractTool | None:
        """Find a tool of the given class in the toolbox."""
        for tool in self.tools:
            if isinstance(tool, tool_cls):
                return tool
        return None

    def reset(self) -> None:
        """Sync reset — calls `reset` on every leaf."""
        for tool in self.tools:
            tool.reset()

    async def async_reset(self) -> None:
        """Async reset. Same body as `reset` — leaves' `reset` is sync."""
        for tool in self.tools:
            tool.reset()

    def close(self) -> None:
        """Sync close — calls `close` on every leaf."""
        for tool in self.tools:
            tool.close()

    async def async_close(self) -> None:
        """Async close. Same body as `close` — leaves' `close` is sync."""
        for tool in self.tools:
            tool.close()

    def execute_action(self, action: Action) -> Observation:
        """Sync dispatch — delegates to leaf's `execute_action`.
        Bridging for async actions happens inside the leaf."""
        if action.name not in self._action_name_to_tool:
            raise ValueError(f"Action '{action.name}' is not supported by any tool in the toolbox.")
        return self._action_name_to_tool[action.name].execute_action(action)

    async def async_execute_action(self, action: Action) -> Observation:
        """Async dispatch — delegates to leaf's `async_execute_action`.
        Sync leaves hop through `asyncio.to_thread` automatically."""
        if action.name not in self._action_name_to_tool:
            raise ValueError(f"Action '{action.name}' is not supported by any tool in the toolbox.")
        return await self._action_name_to_tool[action.name].async_execute_action(action)


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
    """Configuration for a list of tools whose `make()` is async."""

    tool_configs: list[AsyncToolConfig] = []

    async def make(self, container: Container | None = None) -> Toolbox:
        tools = [await tc.make(container) for tc in self.tool_configs]
        return Toolbox(tools=tools)
