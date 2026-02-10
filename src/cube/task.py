"""
Task Session Management for CUBE.

This module provides the Task base class and TaskSession class which implements
the task-level API for managing individual task instances. It handles both MCP
protocol methods (tools/*, resources/*) and CUBE extensions (cube/*).
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from mcp.types import (
    Tool as MCPTool,
)

from cube.benchmark import RuntimeContext
from cube.containers import Container, ContainerConfig
from cube.tool import ToolConfig
from cube.types import (
    Observation,
    TaskMetadata,
    TypedBaseModel,
)

logger = logging.getLogger(__name__)


class Task(TypedBaseModel, ABC):
    """Represents a task that an agent must complete in an environment."""

    metadata: TaskMetadata
    tool: Any  # access to the environment tool, initialized in setup()
    runtime_context: RuntimeContext | None = None
    container: Container | None = None  # access to the environment container, initialized in setup()
    validate_per_step: bool = False

    @property
    def id(self) -> str:
        return self.metadata.id

    @property
    def seed(self) -> int | None:
        return self.metadata.seed

    @abstractmethod
    def setup(self) -> tuple[Observation, dict]:
        """
        Set up the task in the given environment.

        Returns:
            Tuple of (Observation, dict with additional task info)
        """
        pass

    def teardown(self) -> None:
        """Optional clean up after task completion."""
        pass

    @abstractmethod
    def validate_task(self, obs: Observation) -> tuple[float, dict]:
        """Validate the current state of the task and return (reward, info)."""
        pass

    def filter_actions(self, actions: list[MCPTool]) -> list[MCPTool]:
        """Allows the task to whitelist subset of all the actions provided by the environment."""
        return actions

    def cheat(self):
        """
        Solve the task using a pre-defined solution (optional).
        """
        raise NotImplementedError

    def finished(self) -> bool:
        """Check if the task is finished."""
        return False


class TaskConfig(TypedBaseModel):
    """
    Serializable task configuration (Pydantic BaseModel).

    Must be JSON-serializable to pass to workers.
    Contains references and configs, but NOT task logic/metadata.
    Task logic (intent, eval functions) is retrieved via task_id.
    """

    task_id: str
    # Optional configs (provided by benchmark)
    tool_config: ToolConfig
    container_config: ContainerConfig | None = None

    def make(self, runtime_context: RuntimeContext | None = None) -> Task:
        """
        Instantiate task from config.

        Called on worker after deserialization.

        Steps:
        1. Create tools (if tool_config provided)
        2. Start container (if container_config provided)
        3. Create Task with logic and tools

        Returns: Ready-to-use Task instance

        Note: For RPC, spawn = task_config.make() + make_task_rpc_server()
        RPC support can be added later without changing this API.
        """
        if self.tool_config:
            tool = self.tool_config.make()
        else:
            tool = None
        if self.container_config:
            container = self.container_config.make()
        else:
            container = None
        return Task(tools=tool, container=container, runtime_context=runtime_context)


# Simple exception classes for task session management
class TaskClosedException(Exception):
    """Raised when trying to interact with a closed task session."""

    def __init__(self, session_id: str):
        super().__init__(f"Task session {session_id} has been closed")
        self.session_id = session_id


class ResourceNotFoundException(Exception):
    """Raised when a requested resource is not found."""

    def __init__(self, uri: str):
        super().__init__(f"Resource not found: {uri}")
        self.uri = uri
