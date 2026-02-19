import importlib
import traceback
from typing import Any, Callable, Literal, Self

import litellm
from pydantic import BaseModel, Field, model_serializer, model_validator


class TypedBaseModel(BaseModel):
    """
    Base class for Pydantic models that can save and load their type information.

    When serialized, includes `_type` field with the fully qualified class name.
    When deserialized, uses `_type` to instantiate the correct subclass.

    This allows saving/loading configs where the field type is an abstract base class
    but the actual value is a concrete subclass (e.g., AgentConfig -> ReactAgentConfig).
    """

    @model_serializer(mode="wrap")
    def _serialize_with_type(self, handler):
        data = handler(self)
        data["_type"] = f"{self.__class__.__module__}.{self.__class__.__name__}"
        return data

    @model_validator(mode="wrap")
    @classmethod
    def _deserialize_with_type(cls, value, handler):
        if isinstance(value, dict) and "_type" in value:
            type_path = value.pop("_type")
            module_name, class_name = type_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            actual_cls = getattr(module, class_name)
            return actual_cls.model_validate(value)
        return handler(value)


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
