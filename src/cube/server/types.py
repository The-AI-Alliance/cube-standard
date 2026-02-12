"""
Core type definitions for CUBE.

This module contains shared type definitions that are used across both
core domain logic and API schemas. Separating these types prevents
circular import dependencies.
"""

import datetime
from enum import Enum
from typing import Any

from mcp.types import (
    CallToolRequestParams as MCPCallToolRequestParams,
)
from pydantic import Field

from cube import TypedBaseModel
from cube.task import TaskMetadata
from cube.tool import EnvironmentOutput

# =============================================================================
# Base Classes
# =============================================================================


class JSONRPCRequest(TypedBaseModel):
    """JSON-RPC 2.0 request format."""

    jsonrpc: str = "2.0"
    method: str
    params: dict[str, Any] | None = None
    id: str | int | None = None


class JSONRPCResponse(TypedBaseModel):
    """JSON-RPC 2.0 response format."""

    jsonrpc: str = "2.0"
    result: Any | None = None
    error: dict[str, Any] | None = None
    id: str | int | None = None


class TaskRequest(TypedBaseModel):
    """
    Request schema for cube/tasks endpoint.

    Used by: cube/tasks (parameters for filtering results)

    Attributes:
        task_id (str | None): Unique task identifier. If None, fetches all tasks (default: None)
        offset (int): Offset for pagination (default: 0)
        limit (int): Limit for number of tasks to return. -1 means no limit (default: -1)
        filter (dict[str, Any]): Filter criteria for tasks (default: empty dict)
    """

    task_id: str | None = Field(default=None, description="Unique task identifier. If None, fetches all task")
    offset: int = Field(default=0, description="Offset for pagination")
    limit: int = Field(default=-1, description="Limit for number od tasks to return. -1 means no limit")
    filter: dict[str, Any] = Field(default_factory=dict, description="Filter criteria for tasks")


class TaskListResponse(TypedBaseModel):
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


class SpawnRequest(TypedBaseModel):
    """
    Request schema for cube/spawn endpoint.

    Used by: cube/spawn

    Attributes:
        task_id (str): Task ID to spawn
        seed (int | None): Random seed for reproducibility (default: None)
    """

    task_id: str = Field(..., description="Task ID to spawn")
    seed: int | None = Field(default=None, description="Random seed for reproducibility")


class SpawnResponse(TypedBaseModel):
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
    other: dict[str, Any] = Field(default_factory=dict, description="Additional session information")
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


class StatusRequest(TypedBaseModel):
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
    limit: int = Field(default=-1, description="Limit for number od tasks to return. -1 means no limit")
    filter: dict[str, Any] = Field(default_factory=dict, description="Filter criteria for tasks")


class TaskStatus(TypedBaseModel):
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


class StatusResponse(TypedBaseModel):
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


class ShutdownRequest(TypedBaseModel):
    """
    Request schema for cube/shutdown endpoint.

    Used by: cube/shutdown

    Attributes:
        session_id (str | None): Specific session to shutdown (omit for all) (default: None)
    """

    session_id: str | None = Field(default=None, description="Specific session to shutdown (omit for all)")


class ShutdownResponse(TypedBaseModel):
    """
    Response schema for cube/shutdown endpoint.

    Used by: cube/shutdown

    Attributes:
        success (bool): Whether shutdown was successful
        cleaned (list[str]): List of session IDs that were cleaned up
    """

    success: bool = Field(..., description="Whether shutdown was successful")
    cleaned: list[str] = Field(..., description="List of session IDs that were cleaned up")


# =============================================================================
# Task-Level API Schemas
# =============================================================================

# =============================================================================
# tools/list endpoint -- use MCP types
# =============================================================================


# =============================================================================
# tools/call endpoint -- use MCP types
# =============================================================================

# =============================================================================
# resources/list endpoint -- use MCP types
# =============================================================================


# =============================================================================
# resources/read endpoint -- use MCP types
# =============================================================================


# =============================================================================
# cube/evaluation endpoint
# =============================================================================


class EvaluationResponse(TypedBaseModel):
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


class ResetRequest(TypedBaseModel):
    """
    Request schema to reset a task.

    Used by: cube/reset

    Attributes:
        seed (int | None): Random seed for reset (default: None)
    """

    seed: int | None = Field(default=None, description="Random seed for reset")


class ResetResponse(TypedBaseModel):
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


class CloseResponse(TypedBaseModel):
    """
    Response schema from closing a task.

    Used by: cube/close

    Attributes:
        success (bool): Whether close was successful
        profiling (dict[str, Any] | None): Optional profiling data (default: empty dict)
    """

    success: bool = Field(..., description="Whether close was successful")
    profiling: dict[str, Any] | None = Field(default_factory=dict, description="Optional profiling data")


# =============================================================================
# cube/step endpoint (CUBE extension)
# =============================================================================


class StepRequest(TypedBaseModel):
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


class StepResponse(TypedBaseModel):
    """
    Response schema from executing a step.

    Used by: cube/step (combines tool result and evaluation)

    Attributes:
        response (EnvironmentOutput): Environment output after step
    """

    response: EnvironmentOutput = Field(..., description="Environment output after step")
