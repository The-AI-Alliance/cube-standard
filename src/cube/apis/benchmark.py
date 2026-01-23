# =============================================================================
# Benchmark-Level API Schemas
# =============================================================================

import datetime
from typing import Any
from pydantic import BaseModel, Field

# TODO:
# 1)
# copy the input/output json schemas from:
# https://github.com/The-AI-Alliance/cube-standard/blob/1eaeda50f59c31ba728614bb79eb9990017698ed/docs/api/task-level.md
# into README.md
# 2) review README.md examples, and remove stuff if too detailed/verbose
# 3) Align classes below to the examples in the README.md

# =============================================================================
# cube/info -- Nico checked
# =============================================================================
class BenchmarkMetadata(BaseModel):
    """cube/info response schema: all the benchmark information."""

    name: str = Field(..., description="Benchmark name")
    version: str = Field(..., description="Benchmark version")
    description: str = Field(..., description="Benchmark description")
    authors: list[str] = Field(default_factory=list, description="List of benchmark author names")
    license: str = Field(..., description="Benchmark license")
    requirements: dict[str, Any] = Field(default_factory=dict, description="Hardware requirements to install and run the benchmark")
    num_tasks: int = Field(..., description="Total number of tasks")
    other: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    # TODO: discuss adding or removing fields

# =============================================================================
# cube/tasks -- Nico checked
# =============================================================================
class TaskRequest(BaseModel):
    """cube/tasks request schema: parameters for filtering results."""

    task_id: str | None = Field(default=None, description="Unique task identifier. If None, fetches all task")
    offset: int = Field(default=0, description="Offset for pagination")
    limit: int = Field(default=-1, description="Limit for number od tasks to return. -1 means no limit")
    filter: dict[str, Any] = Field(default_factory=dict, description="Filter criteria for tasks")

class TaskMetadata(BaseModel):
    """Task metadata. Part of cube/tasks response schema."""

    id: str = Field(..., description="Unique task identifier")
    seed: int | None = Field(default=None, description="Random seed for the task, if applicable")
    description: str = Field(default="", description="Task description")
    tags: list[str] = Field(default_factory=list, description="List of task tags")
    other: dict[str, Any] = Field(default_factory=dict, description="Additional task metadata")
    # TODO: discuss adding or removing fields such as difficulty, domain, min/max_steps, etc.

class TaskListResponse(BaseModel):
    """cube/tasks response schema: list of tasks with pagination info."""

    tasks: list[TaskMetadata] = Field(..., description="List of tasks")
    total: int = Field(..., description="Total number of tasks available")
    offset: int = Field(default=0, description="Offset used for pagination")
    limit: int = Field(default=-1, description="Limit used for pagination")

# =============================================================================
# cube/spawn -- Nico checked
# =============================================================================

class SpawnRequest(BaseModel):
    """cube/spawn request schema."""

    task_id: str = Field(..., description="Task ID to spawn")
    seed: int | None = Field(default=None, description="Random seed for reproducibility")

class SpawnResponse(BaseModel):
    """cube/spawn response schema."""

    url: str = Field(..., description="URL endpoint for the spawned task session")
    session_id: str = Field(..., description="Unique session identifier")
    other: dict[str, Any] = Field(default_factory=dict, description="Additional session information")
    # TODO: discuss adding fields such as spawned_time, expiration_time, etc.

# =============================================================================
# cube/status -- TODO: Aman check
# =============================================================================
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

# ============================================================================
# cube/shutdown -- TODO: Aman check
# ============================================================================

class ShutdownRequest(BaseModel):
    """Request to shutdown tasks."""

    session_id: str | None = Field(default=None, description="Specific session to shutdown (omit for all)")


class ShutdownResponse(BaseModel):
    """Response from shutdown."""

    success: bool = Field(..., description="Whether shutdown was successful")
    cleaned: list[str] = Field(..., description="List of session IDs that were cleaned up")

