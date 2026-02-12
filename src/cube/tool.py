"""
Tool configuration for CUBE benchmarks.

ToolConfig allows researchers to swap MCP server implementations for experimentation,
enabling research on different tool sets, implementations, and configurations without
modifying benchmark code.
"""

import logging
import traceback
from abc import ABC, abstractmethod
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, List, Literal, Self

import litellm
from pydantic import ConfigDict, Field

from cube import TypedBaseModel

# Forward reference to avoid circular import
if TYPE_CHECKING:
    pass


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


class Action(TypedBaseModel):
    """
    A class representing a function call.

    Attributes:
        id (str): The identifier for the tool call.
        name (str): The name of the function being called.
        arguments (Any): The arguments to be passed to the function.
    """

    id: str | None = None
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Content(TypedBaseModel):
    """
    Represents a piece of content in an observation.

    This is CUBE's domain model for observation content. While MCP has TextContent,
    ImageContent, etc., CUBE uses a simpler unified Content model since observations
    may contain arbitrary data types beyond MCP's content types.

    For MCP protocol responses (tool results, resources), use MCP's content types directly.

    Attributes:
        type (str): Content type (text, image, etc.) (default: "text")
        tool_call_id (str | None): Content could be result of a tool call (default: None)
        name (str | None): Optional name of the content (default: None)
        data (str | bytes): The actual content data
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: str = Field(default="text", description="Content type (text, image, etc.)")
    tool_call_id: str | None = None  # content could be result of a tool call
    name: str | None = None  # optional name of the content
    data: str | bytes


class Observation(TypedBaseModel):
    """
    Represents an observation from the environment.

    An observation encapsulates the information returned from the environment
    after an action is taken. It can contain multiple pieces of content with
    different types (text, images, etc.).

    Attributes:
        contents (list[Content]): List of content pieces that make up this observation.
    """

    contents: list[Content] = Field(default_factory=list)

    @classmethod
    def from_text(cls, text: str) -> Self:
        return cls(contents=[Content(data=text)])

    def __add__(self, other: Self) -> Self:
        self.contents += other.contents
        return self


class StepError(TypedBaseModel):
    """Represents an error that occurred during a step execution."""

    error_type: str
    exception_str: str
    stack_trace: str

    @classmethod
    def from_exception(cls, exc: Exception) -> "StepError":
        """Create a StepError from an exception object."""
        return cls(
            error_type=type(exc).__name__,
            exception_str=str(exc),
            stack_trace="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )


class EnvironmentOutput(TypedBaseModel):
    """
    Represents the result of an environment step.

    This follows the Gymnasium API standard for environment responses,
    containing the observation, reward, termination flags, and additional info.

    Attributes:
        obs (Observation): The observation from the environment after the step.
        reward (float): The reward received for the step (default: 0.0).
        done (bool): Whether the episode has terminated (default: False).
        truncated (bool): Whether the episode was terminated due to step or time limit (default: False).
        info (dict): Additional information about the step (default: empty dict).
        error (StepError|None): python exception if any (default: None).
    """

    obs: Observation
    reward: float = 0.0
    done: bool = False
    truncated: bool = False
    info: dict = Field(default_factory=dict)
    error: StepError | None = None


class AbstractTool(ABC):
    """
    Abstract interface for objects that can react on a list of actions.
    List defined by the Protocol that tool inherits.
    """

    def reset(self) -> None:
        """Optional reset the tool to its initial state"""
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
    Configuration for creating MCP servers with task-specific tools.

    ToolConfig enables research on tool variability by allowing researchers to:
    - Swap out different tool implementations (e.g., Playwright vs Selenium)
    - Provide different tool sets (e.g., basic vs advanced browser tools)
    - Configure tool behavior (e.g., browser types, shell environments)
    """

    @abstractmethod
    def make(self) -> AbstractTool:
        """
        Instantiate Tool from configuration data.

        This method allows creating Tool instances from serialized data,
        enabling easy swapping of tool configurations in benchmarks.

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

        if not method or not getattr(method, "_is_action", False):
            raise ValueError(f"Action {action.name} is not available in {self.__class__.__name__}")

        try:
            action_result = method(**action.arguments) or "Success"
        except Exception as e:
            action_result = f"Error executing action {action.name}: {e}"
            logger.exception(action_result)
            return StepError.from_exception(e)

        return Observation(contents=[Content(data=action_result, tool_call_id=action.id)])

    @property
    def action_set(self) -> List[ActionSchema]:
        """Automatically discover all methods marked with @tool_action decorator."""
        actions = []

        # Introspect the class to find all methods marked as actions
        for attr_name in dir(self):
            # Skip private/protected methods
            if attr_name.startswith("_"):
                continue

            attr = getattr(self, attr_name)

            # Check if it's marked as an action
            if callable(attr) and getattr(attr, "_is_action", False):
                actions.append(ActionSchema.from_function(attr))

        return actions
