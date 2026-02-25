"""
Tool configuration for CUBE benchmarks.

This module defines AbstractTool, Tool, ToolConfig, and the @tool_action decorator
for implementing and configuring agent action interfaces. ToolConfig allows researchers
to swap tool implementations for experimentation, enabling research on different tool
sets and configurations without modifying benchmark code.

Abstract classes:
    AbstractTool — subclasses must implement:
        - execute_action(action: Action) -> Observation | StepError
        - action_set (property) -> list[ActionSchema]
    Tool is a concrete subclass of AbstractTool that implements both automatically
    via the @tool_action decorator — subclass Tool instead of AbstractTool directly.

    ToolConfig — subclasses must implement:
        - make(container) -> AbstractTool    instantiate the tool from serialized config data,
                                             connecting to the container if one was launched

Example — defining a custom tool and its config:

    from cube.tool import Tool, ToolConfig, tool_action
    from cube.containers import Container

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

The BrowserToolConfig can then be passed to a Task or Benchmark, letting
harness users swap browser backends without touching benchmark logic.
"""

import logging
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Callable, List

from cube.containers import Container
from cube.core import Action, ActionSchema, Content, Observation, StepError, TypedBaseModel

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


class ToolConfig(TypedBaseModel, ABC):
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


def tool_action(func: Callable) -> Callable:
    """
    Decorator to mark a method as an action in a Tool.

    This decorator automatically registers methods as actions that will be
    discovered by the Tool's action_set property.

    Usage:
        class MyTool(Tool):
            @tool_action
            def my_action(self, param: str) -> str:
                '''Action description.'''
                return f"Result: {param}"
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    # Mark the function as an action
    setattr(wrapper, "_is_action", True)
    return wrapper


class Tool(AbstractTool):
    """
    Base class for tools with automatic action discovery via decorators.

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
        # Get the method
        method = getattr(self, action.name, None)

        if not method:
            raise ValueError(f"Method '{action.name}' does not exist in {self.__class__.__name__}.")
        if not getattr(method, "_is_action", False):
            raise ValueError(
                f"Method '{action.name}' exists in {self.__class__.__name__} but is not decorated with @tool_action. Add @tool_action to expose it as an action."
            )

        try:
            action_result = method(**action.arguments) or "Success"
        except Exception as e:
            action_result = f"Error executing action {action.name}: {e}"
            logger.exception(action_result)
            return StepError.from_exception(e)

        return Observation(contents=[Content.from_data(action_result, tool_call_id=action.id)])

    @property
    def action_set(self) -> List[ActionSchema]:
        """Automatically discover all methods marked with @tool_action decorator."""
        actions = []

        # Introspect the class to find all methods marked as actions
        for attr_name in dir(self):
            # Skip private/protected methods and properties
            if attr_name.startswith("_") or attr_name == "action_set":
                continue

            attr = getattr(self, attr_name)

            # Check if it's marked as an action
            if callable(attr) and getattr(attr, "_is_action", False):
                actions.append(ActionSchema.from_function(attr))

        return actions
