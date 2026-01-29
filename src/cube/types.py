"""
Core type definitions for CUBE.

This module contains shared type definitions that are used across both
core domain logic and API schemas. Separating these types prevents
circular import dependencies.
"""

import datetime
import importlib
from typing import Any, Self
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict, model_serializer, model_validator


# =============================================================================
# Base Classes
# =============================================================================


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


# =============================================================================
# Core Domain Models
# =============================================================================


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


class EnvironmentOutput(TypedBaseModel):
    """
    Represents the result of an environment step.

    This follows the Gymnasium API standard for environment responses,
    containing the observation, reward, termination flags, and additional info.

    Attributes:
        obs (Observation): The observation from the environment after the step.
        reward (float): The reward received for the step (default: 0.0).
        terminated (bool): Whether the episode has terminated (default: False).
        truncated (bool): Whether the episode was truncated due to step limit (default: False).
        step (int): The current step number in the episode (default: 0).
        info (dict): Additional information about the step (default: empty dict).
    """

    obs: Observation
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    step: int = 0
    info: dict = Field(default_factory=dict)


# =============================================================================
# Benchmark-Level API Schemas
# =============================================================================

# =============================================================================
# cube/info endpoint
# =============================================================================
class BenchmarkMetadata(BaseModel):
    """
    Metadata describing a benchmark.

    Used by:
    - Benchmark: metadata attribute
    - API endpoint: cube/info

    Attributes:
        name (str): Benchmark name
        version (str): Benchmark version
        description (str): Benchmark description
        authors (list[str]): List of benchmark author names (default: empty list)
        license (str): Benchmark license (default: empty string)
        requirements (dict[str, Any]): Hardware requirements to install and run the benchmark (default: empty dict)
        num_tasks (int): Total number of tasks (default: 0)
        tags (list[str]): Benchmark tags (default: empty list)
        other (dict[str, Any]): Additional metadata (default: empty dict)
    """

    name: str = Field(..., description="Benchmark name")
    version: str = Field(..., description="Benchmark version")
    description: str = Field(..., description="Benchmark description")
    authors: list[str] = Field(default_factory=list, description="List of benchmark author names")
    license: str = Field(default="", description="Benchmark license")
    requirements: dict[str, Any] = Field(
        default_factory=dict,
        description="Hardware requirements to install and run the benchmark"
    )
    num_tasks: int = Field(default=0, description="Total number of tasks")
    tags: list[str] = Field(default_factory=list, description="Benchmark tags")
    other: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    # TODO: discuss adding fields such as homepage, repository, citation, etc.


# =============================================================================
# cube/tasks endpoint
# =============================================================================


class TaskRequest(BaseModel):
    """
    Request schema for cube/tasks endpoint.

    Used by: cube/tasks (parameters for filtering results)

    Attributes:
        task_id (str | None): Unique task identifier. If None, fetches all tasks (default: None)
        offset (int): Offset for pagination (default: 0)
        limit (int): Limit for number of tasks to return. -1 means no limit (default: -1)
        filter (dict[str, Any]): Filter criteria for tasks (default: empty dict)
    """

    task_id: str | None = Field(
        default=None, description="Unique task identifier. If None, fetches all task"
    )
    offset: int = Field(default=0, description="Offset for pagination")
    limit: int = Field(
        default=-1, description="Limit for number od tasks to return. -1 means no limit"
    )
    filter: dict[str, Any] = Field(
        default_factory=dict, description="Filter criteria for tasks"
    )


class TaskMetadata(BaseModel):
    """
    Metadata describing a task.

    Used by:
    - Task: metadata attribute
    - API endpoint: cube/tasks (list of TaskMetadata in response)

    Attributes:
        id (str): Unique task identifier
        seed (int | None): Random seed for the task, if applicable (default: None)
        description (str): Task description (default: empty string)
        tags (list[str]): List of task tags (default: empty list)
        max_steps (int | None): Maximum number of steps allowed (default: None)
        difficulty (str | None): Task difficulty level (default: None)
        domain (str | None): Task domain (e.g., 'web', 'coding') (default: None)
        other (dict[str, Any]): Additional task metadata (default: empty dict)
    """

    id: str = Field(..., description="Unique task identifier")
    seed: int | None = Field(default=None, description="Random seed for the task, if applicable")
    description: str = Field(default="", description="Task description")
    tags: list[str] = Field(default_factory=list, description="List of task tags")
    max_steps: int | None = Field(default=None, description="Maximum number of steps allowed")
    difficulty: str | None = Field(default=None, description="Task difficulty level")
    domain: str | None = Field(default=None, description="Task domain (e.g., 'web', 'coding')")
    other: dict[str, Any] = Field(default_factory=dict, description="Additional task metadata")
    # TODO: discuss adding fields such as created_at, updated_at, etc.

class TaskListResponse(BaseModel):
    """
    Response schema for cube/tasks endpoint.

    Used by: cube/tasks (list of tasks with pagination info)

    Attributes:
        tasks (list[TaskMetadata]): List of tasks
        total (int): Total number of tasks available
        offset (int): Offset used for pagination (default: 0)
        limit (int): Limit used for pagination (default: -1)
    """

    tasks: list[TaskMetadata] = Field(..., description="List of tasks")
    total: int = Field(..., description="Total number of tasks available")
    offset: int = Field(default=0, description="Offset used for pagination")
    limit: int = Field(default=-1, description="Limit used for pagination")


# =============================================================================
# cube/spawn endpoint
# =============================================================================


class SpawnRequest(BaseModel):
    """
    Request schema for cube/spawn endpoint.

    Used by: cube/spawn

    Attributes:
        task_id (str): Task ID to spawn
        seed (int | None): Random seed for reproducibility (default: None)
    """

    task_id: str = Field(..., description="Task ID to spawn")
    seed: int | None = Field(
        default=None, description="Random seed for reproducibility"
    )


class SpawnResponse(BaseModel):
    """
    Response schema for cube/spawn endpoint.

    Used by: cube/spawn

    Attributes:
        url (str): URL endpoint for the spawned task session
        session_id (str): Unique session identifier
        other (dict[str, Any]): Additional session information (default: empty dict)
    """

    url: str = Field(..., description="URL endpoint for the spawned task session")
    session_id: str = Field(..., description="Unique session identifier")
    other: dict[str, Any] = Field(
        default_factory=dict, description="Additional session information"
    )
    # TODO: discuss adding fields such as spawned_time, expiration_time, etc. or keep them in other


# =============================================================================
# cube/status endpoint
# =============================================================================


class TaskStatusEnum(str, Enum):
    """
    Status of a running task session.

    Used by: cube/status
    """

    running = "running"
    stopped = "stopped"
    error = "error"


class StatusRequest(BaseModel):
    """
    Request schema for cube/status endpoint.

    Used by: cube/status (parameters for filtering results)

    Attributes:
        session_id (str | None): Unique task session identifier. If None, fetches all running tasks (default: None)
        offset (int): Offset for pagination (default: 0)
        limit (int): Limit for number of tasks to return. -1 means no limit (default: -1)
        filter (dict[str, Any]): Filter criteria for tasks (default: empty dict)
    """

    session_id: str | None = Field(
        default=None,
        description="Unique task session identifier. If None, fetches all running tasks",
    )
    offset: int = Field(default=0, description="Offset for pagination")
    limit: int = Field(
        default=-1, description="Limit for number od tasks to return. -1 means no limit"
    )
    filter: dict[str, Any] = Field(
        default_factory=dict, description="Filter criteria for tasks"
    )


class TaskStatus(BaseModel):
    """
    Status information for a running task session.

    Used by:
    - Task: status attribute for tracking session state
    - API endpoint: cube/status (in response)

    Attributes:
        session_id (str): Session identifier
        task_id (str): Task identifier
        status (TaskStatusEnum): Task status (running, stopped, error)
        created_at (datetime.datetime): Session creation timestamp
        step_count (int): Number of steps executed (default: 0)
        last_updated (datetime.datetime | None): Last update timestamp (default: None)
        other (dict[str, Any]): Additional status information (default: empty dict)
    """

    session_id: str = Field(..., description="Session identifier")
    task_id: str = Field(..., description="Task identifier")
    status: TaskStatusEnum = Field(..., description="Task status (running, stopped, error)")
    created_at: datetime.datetime = Field(..., description="Session creation timestamp")
    step_count: int = Field(default=0, description="Number of steps executed")
    last_updated: datetime.datetime | None = Field(default=None, description="Last update timestamp")
    other: dict[str, Any] = Field(default_factory=dict, description="Additional status information")
    # TODO: discuss adding fields such as error_message, started_at, ended_at, etc.


class StatusResponse(BaseModel):
    """
    Response schema for cube/status endpoint.

    Used by: cube/status

    Attributes:
        tasks (list[TaskStatus]): List of running task statuses
    """

    tasks: list[TaskStatus] = Field(..., description="List of running task statuses")


# =============================================================================
# cube/shutdown endpoint
# =============================================================================


class ShutdownRequest(BaseModel):
    """
    Request schema for cube/shutdown endpoint.

    Used by: cube/shutdown

    Attributes:
        session_id (str | None): Specific session to shutdown (omit for all) (default: None)
    """

    session_id: str | None = Field(
        default=None, description="Specific session to shutdown (omit for all)"
    )


class ShutdownResponse(BaseModel):
    """
    Response schema for cube/shutdown endpoint.

    Used by: cube/shutdown

    Attributes:
        success (bool): Whether shutdown was successful
        cleaned (list[str]): List of session IDs that were cleaned up
    """

    success: bool = Field(..., description="Whether shutdown was successful")
    cleaned: list[str] = Field(
        ..., description="List of session IDs that were cleaned up"
    )


# =============================================================================
# Task-Level API Schemas
# =============================================================================

# =============================================================================
# tools/list endpoint -- use MCP types
# =============================================================================

from mcp.types import (
    ListToolsRequest as MCPListToolsRequest,
    ListToolsResult as MCPListToolsResult,
)

# =============================================================================
# tools/call endpoint -- use MCP types
# =============================================================================

from mcp.types import (
    CallToolRequest as MCPCallToolRequest,
    CallToolRequestParams as MCPCallToolRequestParams,
    CallToolResult as MCPCallToolResult,
)

# =============================================================================
# resources/list endpoint -- use MCP types
# =============================================================================

from mcp.types import (
    ListResourcesRequest as MCPListResourcesRequest,
    ListResourcesResult as MCPListResourcesResult,
    Resource as MCPResource,
)

# =============================================================================
# resources/read endpoint -- use MCP types
# =============================================================================

from mcp.types import (
    ReadResourceRequest as MCPReadResourceRequest,
    ReadResourceRequestParams as MCPReadResourceRequestParams,
    ReadResourceResult as MCPReadResourceResult,
)

# =============================================================================
# cube/evaluation endpoint
# =============================================================================

class EvaluationResponse(BaseModel):
    """
    Response schema from evaluating the environment state.

    Used by: cube/evaluation

    Attributes:
        response (EnvironmentOutput): Environment output after evaluation
    """

    response: EnvironmentOutput = Field(..., description="Environment output after evaluation")

# =============================================================================
# cube/reset endpoint
# =============================================================================


class ResetRequest(BaseModel):
    """
    Request schema to reset a task.

    Used by: cube/reset

    Attributes:
        seed (int | None): Random seed for reset (default: None)
    """

    seed: int | None = Field(default=None, description="Random seed for reset")


class ResetResponse(BaseModel):
    """
    Response schema from resetting a task.

    Used by: cube/reset

    Attributes:
        response (EnvironmentOutput): Environment output after reset
    """

    response: EnvironmentOutput = Field(..., description="Environment output after reset")


# =============================================================================
# cube/close endpoint
# =============================================================================


class CloseResponse(BaseModel):
    """
    Response schema from closing a task.

    Used by: cube/close

    Attributes:
        success (bool): Whether close was successful
        profiling (dict[str, Any] | None): Optional profiling data (default: empty dict)
    """

    success: bool = Field(..., description="Whether close was successful")
    profiling: dict[str, Any] | None = Field(
        default_factory=dict, description="Optional profiling data"
    )


# =============================================================================
# cube/step endpoint (CUBE extension)
# =============================================================================


class StepRequest(BaseModel):
    """
    Request schema to execute a step (tool call + evaluation).

    Used by: cube/step (CUBE extension)

    Attributes:
        params (MCPCallToolRequestParams): Parameters for tool call
    """

    params: MCPCallToolRequestParams = Field(..., description="Parameters for tool call")

    # TODO: when calling this endpoint, we should create a MCP CallToolRequest and submit it
    # we should then call cube/evaluation internally to get the evaluation response
    # finally we return the reponse as a EnvironmentOutput wrapped in StepResponse

class StepResponse(BaseModel):
    """
    Response schema from executing a step.

    Used by: cube/step (combines tool result and evaluation)

    Attributes:
        response (EnvironmentOutput): Environment output after step
    """

    response: EnvironmentOutput = Field(..., description="Environment output after step")
