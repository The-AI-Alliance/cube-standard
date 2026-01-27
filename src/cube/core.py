from abc import ABC, abstractmethod
from typing import Any, Callable, Protocol, TypeAlias

from mcp.types import Tool as MCPTool

from cube.types import (
    Observation,
    TaskMetadata,
    TaskStatus,
)



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
    def filter_actions(self, actions: list[MCPTool]) -> list[MCPTool]:
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
