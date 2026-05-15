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

import inspect
import logging
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


class AbstractAsyncTool(ABC):
    """
    Async variant of AbstractTool. All execution is async.

    Subclass AsyncTool (not this class directly) to get automatic action
    discovery via @tool_action. All @tool_action methods must be async def.
    """

    async def reset(self) -> None:
        """Optional: reset the tool to its initial state."""
        pass

    async def close(self) -> None:
        """Optional: clean up tool resources (connections, processes, files, etc.)."""
        pass

    @abstractmethod
    async def execute_action(self, action: Action) -> Any:
        """Execute a single action and return the result."""
        pass

    @property
    @abstractmethod
    def action_set(self) -> List[ActionSchema]:
        """Returns list of actions supported by that tool (same format as AbstractTool)."""
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
    Base class for sync tools with automatic action discovery via decorators.

    Tool subclasses should mark their action methods with the @tool_action decorator.
    The action_set property will automatically discover and expose these methods.

    Example:
        ```python
        from cube.tool import Tool, tool_action, Action

        class CalculatorTool(Tool):
            '''Calculator tool with basic arithmetic operations.'''

            @tool_action
            def add(self, a: float, b: float) -> str:
                '''Add two numbers together.'''
                return f"Result: {a + b}"

        # Usage
        calc = CalculatorTool()

        # Automatic discovery of actions
        print("Available actions:")
        for action_schema in calc.action_set:
            print(f"  - {action_schema.name}: {action_schema.description}")
        # Output: - add: Add two numbers together.

        # Execute an action
        action = Action(name="add", arguments={"a": 5.0, "b": 3.0})
        result = calc.execute_action(action)
        print(result.contents[0].data)  # "Result: 8.0"
        ```

    Benefits:
        - Zero boilerplate: Just add @tool_action decorator
        - Single source of truth: Method signature and docstring define the action
        - No duplication: Each function defined exactly once
        - Clear intent: Obvious which methods are actions
    """

    def execute_action(self, action: Action) -> Observation | StepError:
        """Execute an action by name."""
        method = self.get_action_method(action)
        try:
            action_result = method(**action.arguments) or "Success"
        except Exception as e:
            action_result = f"Error executing action {action.name}: {e}"
            logger.exception(action_result)
            return StepError.from_exception(e)
        return Observation(contents=[Content.from_data(action_result, tool_call_id=action.id)])


class AsyncTool(_ToolActionsMixin, AbstractAsyncTool):
    """
    Base class for async tools with automatic action discovery via decorators.

    All @tool_action methods must be async def. A TypeError is raised at class
    definition time if a sync method is decorated with @tool_action.

    Example:
        ```python
        from cube.tool import AsyncTool, tool_action, Action

        class AsyncCalculatorTool(AsyncTool):
            '''Async calculator tool.'''

            @tool_action
            async def add(self, a: float, b: float) -> str:
                '''Add two numbers together.'''
                return f"Result: {a + b}"

        # Usage
        calc = AsyncCalculatorTool()
        action = Action(name="add", arguments={"a": 5.0, "b": 3.0})
        result = await calc.execute_action(action)
        ```
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Validate that all @tool_action methods in AsyncTool subclasses are async def.
        """
        super().__init_subclass__(**kwargs)
        for name, attr in cls.__dict__.items():
            if getattr(attr, "_is_action", False) and not inspect.iscoroutinefunction(attr):
                raise TypeError(
                    f"{cls.__name__}.{name} is decorated with @tool_action but is not async. "
                    f"AsyncTool requires all @tool_action methods to be 'async def'."
                )

    async def execute_action(self, action: Action) -> Observation | StepError:
        """Execute an async action by name."""
        method = self.get_action_method(action)
        try:
            action_result = (await method(**action.arguments)) or "Success"
        except Exception as e:
            action_result = f"Error executing action {action.name}: {e}"
            logger.exception(action_result)
            return StepError.from_exception(e)
        return Observation(contents=[Content.from_data(action_result, tool_call_id=action.id)])


class Toolbox(Tool):
    """Composite sync tool that delegates to a list of AbstractTool instances."""

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
        for tool in self.tools:
            tool.reset()

    def execute_action(self, action: Action) -> Observation | StepError:
        if action.name not in self._action_name_to_tool:
            raise ValueError(f"Action '{action.name}' is not supported by any tool in the toolbox.")
        tool = self._action_name_to_tool[action.name]
        assert isinstance(tool, AbstractTool)
        return tool.execute_action(action)

    def close(self) -> None:
        for tool in self.tools:
            tool.close()


class AsyncToolbox(AsyncTool):
    """Composite async tool that delegates to a list of AbstractAsyncTool instances."""

    def __init__(self, tools: list[AbstractAsyncTool]):
        self.tools = tools
        self._action_name_to_tool: dict[str, AbstractAsyncTool] = {}
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

    def find_tool(self, tool_cls: type) -> AbstractAsyncTool | None:
        """Find a tool of the given class in the toolbox."""
        for tool in self.tools:
            if isinstance(tool, tool_cls):
                return tool
        return None

    async def reset(self) -> None:
        for tool in self.tools:
            await tool.reset()

    async def execute_action(self, action: Action) -> Observation | StepError:
        if action.name not in self._action_name_to_tool:
            raise ValueError(f"Action '{action.name}' is not supported by any tool in the toolbox.")
        return await self._action_name_to_tool[action.name].execute_action(action)

    async def close(self) -> None:
        for tool in self.tools:
            await tool.close()


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
