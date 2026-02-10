"""
Tool configuration for CUBE benchmarks.

ToolConfig allows researchers to swap MCP server implementations for experimentation,
enabling research on different tool sets, implementations, and configurations without
modifying benchmark code.
"""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, List, Literal, Self

import litellm
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from typing_extensions import get_protocol_members

from cube.types import Action, Content, Observation, TypedBaseModel

# Forward reference to avoid circular import
if TYPE_CHECKING:
    from cube.task import Task


logger = logging.getLogger(__name__)


class ActionSchema(TypedBaseModel):
    """
    Represents a function specification with a type, name, description and arguments.
    Compatible with OAI, Anthropic and VLLM definitions.

    Attributes:
        type (Literal["function"]): The type of the tool, which is always "function".
        name (str): The name of the function.
        description (str): A brief description of the function.
        parameters (dict): A dictionary containing the parameters of the function.
    """

    type: Literal["function"] = "function"
    name: str
    description: str
    parameters: dict = Field(default_factory=dict)

    @classmethod
    def from_function(cls, func: Callable) -> Self:
        """Create tool object from python function."""
        schema = litellm.utils.function_to_dict(func)
        return cls(**schema)

    def as_dict(self) -> dict[str, Any]:
        """Produce dict that could be passed as tool schema into LLM api."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class AbstractTool(ABC):
    """
    Abstract interface for objects that can react on a list of actions.
    List defined by the ActionSpace that tool inherits.
    """

    @abstractmethod
    def execute_action(self, action: Action) -> Any:
        """Execute a single action and return the result."""
        pass

    @property
    @abstractmethod
    def action_set(self) -> List[ActionSchema]:
        """Returns list of actions supported by that tool."""
        pass


class ToolConfig(TypedBaseModel, ABC):
    """
    Configuration for creating MCP servers with task-specific tools.

    ToolConfig enables research on tool variability by allowing researchers to:
    - Swap out different tool implementations (e.g., Playwright vs Selenium)
    - Provide different tool sets (e.g., basic vs advanced browser tools)
    - Use different MCP server implementations
    - Configure tool behavior (e.g., browser types, shell environments)

    Example:
        >>> class BrowserToolConfig(ToolConfig):
        ...     browser_type: str = "chromium"
        ...     headless: bool = True
        ...
        ...     def create_mcp_server(self, task: WebTask) -> FastMCP:
        ...         mcp = FastMCP(f"Browser: {task.id}")
        ...
        ...         @mcp.tool()
        ...         def navigate(url: str) -> str:
        ...             return task.navigate_with_browser(url, self.browser_type)
        ...
        ...         @mcp.tool()
        ...         def click(selector: str) -> str:
        ...             return task.click_element(selector)
        ...
        ...         return mcp
    """

    @abstractmethod
    def create_mcp_server(self, task: "Task") -> FastMCP:
        """
        Create and configure an MCP server for the given task.

        This method provides full control over MCP server creation:
        - Choose which tools to register
        - Implement tools with different behaviors
        - Configure tool parameters based on research needs

        Args:
            task: The task instance with state and metadata. Task state
                  (e.g., self.counter, self.browser) can be accessed via closure.

        Returns:
            FastMCP server with tools registered

        Example:
            >>> def create_mcp_server(self, task: CounterTask) -> FastMCP:
            ...     mcp = FastMCP(f"Counter: {task.id}")
            ...
            ...     @mcp.tool()
            ...     def increment() -> str:
            ...         task.counter += 1
            ...         return f"Counter is now {task.counter}"
            ...
            ...     if self.enable_decrement:  # Configurable feature
            ...         @mcp.tool()
            ...         def decrement() -> str:
            ...             task.counter -= 1
            ...             return f"Counter is now {task.counter}"
            ...
            ...     return mcp
        """
        raise NotImplementedError("Subclasses must implement create_mcp_server() method.")

    @abstractmethod
    def make(self) -> AbstractTool:
        """
        Instantiate ToolConfig from configuration data.

        This method allows creating ToolConfig instances from serialized data,
        enabling easy swapping of tool configurations in benchmarks.

        Returns:
            AbstractTool instance
        """
        raise NotImplementedError("Subclasses must implement make() method.")


class Tool(AbstractTool):
    """
    Base class for tool that implements an action space protocol.

    :var action_space: Protocol defining the actions this tool supports
    """

    action_space: Any

    def get_action_method(self, action) -> Callable:
        if not getattr(self.action_space, action.name, None):
            raise ValueError(f"Action {action.name} is not a part of {self.action_space}.")
        if not (fn := getattr(self, action.name, None)):
            raise ValueError(f"Action {action.name} is not implemented in {self.__class__.__name__}.")
        return fn

    def execute_action(self, action: Action) -> Observation:
        fn = self.get_action_method(action)

        try:
            action_result = fn(**action.arguments) or "Success"
        except Exception as e:
            action_result = f"Error executing action {action.name}: {e}"
            logger.exception(action_result)

        return Observation(contents=[Content(data=action_result, tool_call_id=action.id)])

    @property
    def action_set(self) -> List[ActionSchema]:
        """Returns list of actions supported by that environment."""
        action_names = get_protocol_members(self.action_space)
        return [ActionSchema.from_function(getattr(self, name)) for name in action_names]
