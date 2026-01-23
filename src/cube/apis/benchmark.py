# =============================================================================
# Benchmark-Level API Schemas
# =============================================================================

from typing import Any
from pydantic import BaseModel, Field


class BenchmarkInfo(BaseModel):
    """Benchmark metadata response."""

    name: str = Field(..., description="Benchmark name")
    version: str = Field(..., description="Benchmark version")
    description: str = Field(..., description="Benchmark description")
    authors: list[str] = Field(default_factory=list, description="List of benchmark author names")
    license: str = Field(..., description="Benchmark license")
    requirements: dict[str, Any] = Field(default_factory=dict, description="Hardware requirements to install and run the benchmark")
    num_tasks: int = Field(..., description="Total number of tasks")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class TaskRequest(BaseModel):
    pass

class TaskInfo(BaseModel):
    """Task metadata."""

    id: str = Field(..., description="Unique task identifier")
    description: str = Field(default="", description="Task description")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional task metadata")


class TaskListResponse(BaseModel):
    """Response for listing tasks."""

    tasks: list[TaskInfo] = Field(..., description="List of tasks")
    total: int = Field(..., description="Total number of tasks available")
    offset: int = Field(default=0, description="Offset used for pagination")
    limit: int = Field(default=10, description="Limit used for pagination")


class SpawnRequest(BaseModel):
    """Request to spawn a task instance."""

    task_id: str = Field(..., description="Task ID to spawn")
    seed: int | None = Field(default=None, description="Random seed for reproducibility")
    tool_config: dict[str, Any] = Field(default_factory=dict, description="Tool configuration")


class SpawnResponse(BaseModel):
    """Response from spawning a task."""

    url: str = Field(..., description="URL endpoint for the spawned task session")
    session_id: str = Field(..., description="Unique session identifier")


class TaskStatus(BaseModel):
    """Status of a running task."""

    session_id: str = Field(..., description="Session identifier")
    task_id: str = Field(..., description="Task identifier")
    status: str = Field(..., description="Task status (running, stopped, error)")
    created_at: datetime = Field(..., description="Session creation timestamp")
    step_count: int = Field(default=0, description="Number of steps executed")


class StatusResponse(BaseModel):
    """Response for status check."""

    tasks: list[TaskStatus] = Field(..., description="List of running task statuses")


class ShutdownRequest(BaseModel):
    """Request to shutdown tasks."""

    session_id: str | None = Field(default=None, description="Specific session to shutdown (omit for all)")


class ShutdownResponse(BaseModel):
    """Response from shutdown."""

    success: bool = Field(..., description="Whether shutdown was successful")
    cleaned: list[str] = Field(..., description="List of session IDs that were cleaned up")

