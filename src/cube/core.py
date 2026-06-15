"""
Core data models for CUBE.

This module defines the fundamental types used across the framework: Action,
ActionSchema, Content (and subclasses), Observation, StepError, EnvironmentOutput,
and TypedBaseModel.

Abstract classes:
    Content — subclasses must implement:
        to_markdown() -> str          render content as a Markdown string
        to_llm_message() -> dict      render content as an LLM message dict
    Built-in implementations are provided for the most common content types:
    TextContent, StructuredContent, ImageContent, AudioContent, VideoContent.
"""

import base64
import importlib
import inspect
import io
import json
import traceback
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from typing import Any, Callable, ClassVar, Self

from PIL import Image as PILImage
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializeAsAny,
    field_serializer,
    field_validator,
    model_serializer,
    model_validator,
)

from cube.utils import function_to_dict


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
        value = handler(self)
        value["_type"] = f"{self.__class__.__module__}.{self.__class__.__name__}"
        return value

    @model_validator(mode="wrap")
    @classmethod
    def _deserialize_with_type(cls, value, handler):
        if isinstance(value, dict):
            if "_type" in value:
                # Copy before popping: the caller's dict may be reused (e.g. re-validated
                # under validate_assignment=True), and losing _type silently downgrades
                # polymorphism on the second pass.
                value = value.copy()
                type_path = value.pop("_type")
                module_path, class_name = type_path.rsplit(".", 1)
                module = importlib.import_module(module_path)  # nosemgrep: non-literal-import
                actual_cls = getattr(module, class_name)
                if not isinstance(actual_cls, type) or not issubclass(actual_cls, cls):
                    raise ValueError(f"Cannot deserialize '{type_path}': class must be a subclass of '{cls.__name__}'.")
                # When actual_cls is cls, fall through to handler(value) instead of
                # recursing: Pydantic v2 warns about "returning non-self from __init__"
                # if a wrap-validator returns the result of model_validate on the same class.
                if actual_cls is not cls:
                    return actual_cls.model_validate(value)
            elif inspect.isabstract(cls):
                raise ValueError(
                    f"Cannot deserialize abstract class '{cls.__name__}' directly. "
                    "Ensure the input dict includes a '_type' field naming a concrete subclass."
                )
        return handler(value)


class ValidatedConfig(TypedBaseModel):
    """A ``TypedBaseModel`` that validates attribute assignment.

    Plain Pydantic models only validate at construction; assigning a wrongly
    typed value afterwards silently succeeds and fails much later (often deep
    in a worker process). Config objects are routinely tweaked by attribute
    assignment in recipes (``agent.budget.cost_limit = 2.0``), so the failure
    must surface at the point of assignment instead.

    Subclass this instead of ``TypedBaseModel`` for any config a user mutates
    after construction. For validation to reach nested writes
    (``cfg.sub.field = ...``) every model in the tree must subclass this.
    """

    model_config = ConfigDict(validate_assignment=True)


class ConfigRegistry[T: BaseModel](Mapping[str, T]):
    """Maps a name to a canonical config; every lookup returns a deep copy.

    Used for the named-config catalogs that recipes pick from — canonical
    agent configs, per-cube benchmark configs, infra profiles:

        agent = GENNY_CONFIGS["swe"]
        agent.budget.cost_limit = 2.0

    Every lookup returns a fresh ``model_copy(deep=True)``, so a caller can
    never mutate the shared canonical instance and corrupt other recipes in
    the process. Trade-off: bind to a variable before mutating —
    ``REG["x"].field = y`` mutates a throwaway and no-ops.

    A ``Mapping`` (not a ``dict`` subclass) on purpose: ``dict.get`` /
    ``.values`` / ``.items`` / ``**unpack`` would bypass ``__getitem__`` and
    hand out the shared instance. ``Mapping`` routes every read through
    ``__getitem__``, so copy-on-access actually holds.
    """

    def __init__(self, configs: dict[str, T]) -> None:
        self._configs = configs

    def __getitem__(self, name: str) -> T:
        try:
            return self._configs[name].model_copy(deep=True)
        except KeyError:
            raise KeyError(f"Unknown config {name!r}. Available: {sorted(self._configs)}") from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._configs)

    def __len__(self) -> int:
        return len(self._configs)


class ActionSchema(TypedBaseModel):
    """
    Represents a function specification with a type, name, description and arguments.
    Compatible with OAI, Anthropic and VLLM definitions.

    Attributes:
        name (str): The name of the function.
        description (str): A brief description of the function.
        parameters (dict): A dictionary containing the parameters of the function.
    """

    name: str
    description: str
    parameters: dict = Field(default_factory=dict)

    @field_validator("name", "description")
    @classmethod
    def _must_be_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must be a non-empty string")
        return v

    def validate_param_descriptions(self) -> tuple[bool, str]:
        """Check that every parameter (except 'self') has a non-empty description."""
        props = self.parameters.get("properties", {})
        for param_name, param_info in props.items():
            if param_name == "self":
                continue
            if not isinstance(param_info, dict):
                return False, f"parameter '{param_name}' invalid"
            desc = param_info.get("description")
            if not desc or not str(desc).strip():
                return False, f"parameter '{param_name}' missing description"
        return True, ""

    @classmethod
    def from_function(cls, func: Callable) -> Self:
        """Create tool object from python function."""
        schema = function_to_dict(func)
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

    @classmethod
    def from_openai_tool_call(cls, tool_call: dict) -> Self:
        """Create an Action from an OpenAI tool call dict.

        Supports both Chat Completions format:
            {"id": "call_xxx", "type": "function", "function": {"name": "click", "arguments": "{...}"}}
        and flat/Responses API format:
            {"id": "call_xxx", "name": "click", "arguments": "{...}"}
        """
        if "function" in tool_call:
            func = tool_call["function"]
            name = func["name"]
            raw_args = func.get("arguments") or "{}"
        else:
            name = tool_call["name"]
            raw_args = tool_call.get("arguments") or "{}"

        arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        return cls(id=tool_call.get("id") or tool_call.get("call_id"), name=name, arguments=arguments)


class Content(TypedBaseModel, ABC):
    """
    Abstract base class for all content types in an observation.

    Subclasses represent specific data types (text, image, audio, video, structured data)
    and provide `to_markdown()` and `to_llm_message()` implementations.

    Attributes:
        tool_call_id (str | None): Set if this content is the result of a tool call.
        name (str | None): Optional label for the content.
        data (Any): The actual content data; narrowed to a specific type in each subclass.
    """

    tool_call_id: str | None = None
    name: str | None = None
    data: Any

    @abstractmethod
    def to_markdown(self) -> str:
        """Render this content as a Markdown string."""
        ...

    @abstractmethod
    def to_llm_message(self) -> dict:
        """Return a well-formed LLM message dict (OpenAI/litellm format)."""
        ...

    def _build_message(self, msg_content: str | list[dict]) -> dict:
        """Wrap content into an LLM message dict, applying role and tool_call_id."""
        if self.tool_call_id:
            return {"role": "tool", "content": msg_content, "tool_call_id": self.tool_call_id}
        return {"role": "user", "content": msg_content}

    @classmethod
    def from_data(cls, data: Any, **kwargs) -> "Content":
        """Instantiate the appropriate Content subclass based on the type of data.

        Dispatches to TextContent, StructuredContent, or ImageContent based on
        isinstance checks. Raw bytes are rejected — callers must explicitly construct
        AudioContent or VideoContent since the media type cannot be inferred.

        Args:
            data: The content data. Supported types: str, int, float, dict, list, BaseModel, PILImage.Image.
            **kwargs: Additional fields passed to the subclass (e.g. name, tool_call_id).

        Raises:
            TypeError: If data is bytes (ambiguous between audio and video) or an unsupported type.
        """
        if isinstance(data, PILImage.Image):
            return ImageContent(data=data, **kwargs)
        if isinstance(data, (dict, list, BaseModel)):
            return StructuredContent(data=data, **kwargs)
        if isinstance(data, (str, int, float)):
            return TextContent(data=str(data), **kwargs)
        if isinstance(data, bytes):
            raise TypeError(
                "Cannot infer content type from raw bytes. Explicitly construct AudioContent or VideoContent instead."
            )
        raise TypeError(
            f"Unsupported data type for Content.from_data(): {type(data).__name__}. "
            "Supported types: str, int, float, dict, list, BaseModel, PILImage.Image."
        )


class TextContent(Content):
    """Text or numeric content, rendered as plain text."""

    data: str

    @field_validator("data", mode="before")
    @classmethod
    def coerce_to_str(cls, v: Any) -> str:
        if isinstance(v, (int, float)):
            return str(v)
        return v

    def to_markdown(self) -> str:
        return f"## {self.name}\n{self.data}" if self.name else self.data

    def to_llm_message(self) -> dict:
        text = f"## {self.name}\n{self.data}" if self.name else self.data
        return self._build_message(text)


class StructuredContent(Content):
    """Structured data (dict, list, or BaseModel), rendered as a JSON code block."""

    data: dict | list | BaseModel

    @field_validator("data", mode="before")
    @classmethod
    def coerce_base_model_to_dict(cls, v: Any) -> Any:
        if isinstance(v, BaseModel):
            return v.model_dump()
        return v

    def to_markdown(self) -> str:
        block = f"```json\n{json.dumps(self.data, indent=2)}\n```"
        return f"## {self.name}\n{block}" if self.name else block

    def to_llm_message(self) -> dict:
        text = json.dumps(self.data)
        if self.name:
            text = f"## {self.name}\n{text}"
        return self._build_message(text)


class ImageContent(Content):
    """PIL image content, serialized as base64 PNG."""

    model_config = {"arbitrary_types_allowed": True}

    data: PILImage.Image

    _image_prefix: ClassVar[str] = "data:image/png;base64,"

    def as_base64_image_str(self, data: PILImage.Image) -> str:
        byte_arr = io.BytesIO()
        data.save(byte_arr, format="PNG")
        encoded_image = base64.b64encode(byte_arr.getvalue()).decode("utf-8")
        return f"{self._image_prefix}{encoded_image}"

    @field_serializer("data")
    def serialize_data(self, data: PILImage.Image) -> str:
        """Serialize a PIL Image to a base64 data URL string for JSON compatibility.

        PIL Images are not JSON-serializable by default; this converts them to a
        prefixed base64 string that can be stored, transmitted, and later reconstructed.
        """
        return self.as_base64_image_str(data)

    @field_validator("data", mode="before")
    @classmethod
    def deserialize_data(cls, v: Any) -> Any:
        """Deserialize a base64 data URL string back into a PIL Image.

        Detects the image prefix to distinguish serialized images from plain strings.
        Forces eager loading via img.load() to prevent the BytesIO buffer from being
        garbage collected before the pixel data is read (PIL uses lazy loading by default).
        """
        if isinstance(v, str) and v.startswith(cls._image_prefix):
            v = v[len(cls._image_prefix) :]
            # Decode base64 string to bytes
            decoded_image = base64.b64decode(v)
            # Open bytes as PIL Image and load immediately to avoid lazy loading issues
            img = PILImage.open(io.BytesIO(decoded_image))
            img.load()  # Force load to prevent BytesIO buffer from being garbage collected
            return img
        return v  # Return original value if not a string (e.g., already an Image object)

    def to_markdown(self) -> str:
        b64_url = self.as_base64_image_str(self.data)
        alt = self.name or "image"
        img = f"![{alt}]({b64_url})"
        return f"## {self.name}\n{img}" if self.name else img

    def to_llm_message(self) -> dict:
        b64_url = self.as_base64_image_str(self.data)
        msg_content: list[dict] = [{"type": "image_url", "image_url": {"url": b64_url}}]
        if self.name:
            msg_content.insert(0, {"type": "text", "text": f"## {self.name}"})
        return self._build_message(msg_content)


class AudioContent(Content):
    """Raw audio bytes.

    Placeholder class for future work. Rendering to Markdown and LLM message formats
    is not yet defined for audio — it is unclear what the best representation would be
    (e.g. waveform image, transcript, base64 embed). Custom to_markdown() and
    to_llm_message() implementations should be added once a rendering strategy is decided.

    TODO: define custom rendering functions for audio content.
    """

    data: bytes
    duration_seconds: float | None = None

    def to_markdown(self) -> str:
        raise NotImplementedError("Markdown rendering is not supported for audio content.")

    def to_llm_message(self) -> dict:
        raise NotImplementedError("Audio is not supported by LLM message format.")


class VideoContent(Content):
    """Raw video bytes.

    Placeholder class for future work. Rendering to Markdown and LLM message formats
    is not yet defined for video — it is unclear what the best representation would be
    (e.g. thumbnail image, transcript, base64 embed). Custom to_markdown() and
    to_llm_message() implementations should be added once a rendering strategy is decided.

    TODO: define custom rendering functions for video content.
    """

    data: bytes
    duration_seconds: float | None = None

    def to_markdown(self) -> str:
        raise NotImplementedError("Markdown rendering is not supported for video content.")

    def to_llm_message(self) -> dict:
        raise NotImplementedError("Video is not supported by LLM message format.")


class Observation(TypedBaseModel):
    """
    Represents an observation from the environment.

    An observation encapsulates the information returned from the environment
    after an action is taken. It can contain multiple pieces of content with
    different types (text, images, etc.).

    Attributes:
        contents (list[Content]): List of content pieces that make up this observation.
    """

    contents: list[SerializeAsAny[Content]] = Field(default_factory=list)
    # Structured error when this observation reports a failed action. The error text is
    # also in `contents` (the agent reads it); this is the machine-readable copy for
    # telemetry / `EnvironmentOutput.error`. A failed action is NON-terminal.
    error: "StepError | None" = None

    @classmethod
    def from_text(cls, text: str) -> Self:
        return cls(contents=[TextContent(data=text)])

    def to_llm_messages(self) -> list[dict]:
        """Convert observation to a list of messages suitable for sending to LLM."""
        return [content.to_llm_message() for content in self.contents]

    def to_markdown(self) -> str:
        """Render the entire observation as a single Markdown string."""
        return "\n\n".join(content.to_markdown() for content in self.contents)

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

    def to_observation(self) -> Observation:
        """Render this error as an observation — text for the agent + the structured
        error attached (`Observation.error`) for the runtime. A failed action is fed back
        as a normal observation (the agent reads it and retries), never terminal."""
        return Observation(
            contents=[TextContent(data=f"Action failed — {self.error_type}: {self.exception_str}")],
            error=self,
        )


class AgentStop(BaseException):
    """Raised when the agent ends the episode — by executing the STOP action
    (``final_step``, see :data:`STOP_ACTION`), which simply raises this.

    A ``BaseException`` (not ``Exception``) so a tool's / agent's ``except Exception``
    never swallows it. The gym ``Task.step`` view catches it (``done=True``); the
    agent-facing path lets it propagate to the runtime. Carries the terminal observation.
    """

    def __init__(self, observation: "Observation | None" = None) -> None:
        self.observation = (
            observation if observation is not None else Observation.from_text("Task finished by the agent.")
        )
        super().__init__("Agent requested task stop")


# The STOP action every agent can take to end its episode. Implemented as a real tool
# action (``Tool.final_step``) that raises ``AgentStop`` — so there is no special-casing
# in the dispatch path; executing it just raises. Schema is Anthropic-safe (no args).
STOP_ACTION = ActionSchema(
    name="final_step",
    description="Stop the task execution.",
    parameters={"type": "object", "properties": {}},
)

# Resolve Observation.error now that StepError is defined (forward ref above).
Observation.model_rebuild()


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
