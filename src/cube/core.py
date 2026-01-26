from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Protocol, Self, TypeAlias

import litellm.utils
from pydantic import ConfigDict, Field

from cube.apis.benchmark import TaskMetadata, TaskStatus
from cube.base import TypedBaseModel


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
    arguments: Dict[str, Any] = Field(default_factory=dict)


class Content(TypedBaseModel):
    """Represents a piece of content in an observation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tool_call_id: str | None = None  # content could be result of a tool call
    name: str | None = None  # optional name of the content
    data: str | bytes


class Observation(TypedBaseModel):
    """Represents an observation from the environment."""

    contents: list[Content] = Field(default_factory=list)

    @classmethod
    def from_text(cls, text: str) -> Self:
        return cls(contents=[Content(data=text)])

    def __add__(self, other: Self) -> Self:
        self.contents += other.contents
        return self


class EnvironmentOutput(TypedBaseModel):
    """Represents the result of an environment step."""

    obs: Observation
    reward: float = 0.0
    done: bool = False
    step: int = 0
    info: dict = Field(default_factory=dict)


class ActionSpace(Protocol):
    """Base class for action spaces."""

    pass


ActionSubset: TypeAlias = tuple[Callable, ...]


class Task(ABC):
    """Represents a task that an agent must complete in an environment."""

    metadata: TaskMetadata
    status: TaskStatus | None = None  # will get instantiated once we call benchmark.spawn() or cube/spawn
    _tool: Any  # access to the environment tool, initialized in setup()
    validate_per_step: bool = False

    @property
    def id(self) -> str:
        return self.metadata.id
    
    @property
    def seed(self) -> int | None:
        return self.metadata.seed

    @abstractmethod
    def setup(self, tool: Any) -> tuple[Observation, dict]:
        """
        Set up the task in the given environment.

        Returns:
            Tuple of (Observation, dict with additional task info)
        """
        self._tool = tool

    def teardown(self) -> None:
        """Optional clean up after task completion."""
        pass

    @abstractmethod
    def validate_task(self, obs: Observation) -> tuple[float, dict]:
        """Validate the current state of the task and return (reward, info)."""
        pass

    @abstractmethod
    def filter_actions(self, actions: list[ActionSchema]) -> list[ActionSchema]:
        """Allows the task to whitelist subset of all the actions provided by the environment."""
        pass

    def cheat(self):
        """
        Solve the task using a pre-defined solution (optional).
        """
        raise NotImplementedError

    def obs_postprocess(self, obs: Observation) -> Observation:
        """Optional post-processing of observation before returning it to the agent."""
        return obs

    def finished(self) -> bool:
        """Check if the task is finished."""
        return False

    def accept_agent_stop(self) -> bool:
        """Optional, whether the task accepts the agent stopping the task right now. Default is True."""
        return True
