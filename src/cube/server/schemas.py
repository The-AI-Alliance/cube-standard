"""API request and response schemas for CUBE server."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# =============================================================================
# Common Schemas
# =============================================================================

class ErrorDetail(BaseModel):
    """Standard error response."""

    code: str = Field(..., description="Error code (e.g., TASK_NOT_FOUND)")
    message: str = Field(..., description="Human-readable error message")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional error details")


class ErrorResponse(BaseModel):
    """Error response wrapper."""

    error: ErrorDetail


# =============================================================================
# Benchmark-Level API Schemas
# =============================================================================

class BenchmarkInfo(BaseModel):
    """Benchmark metadata response."""

    name: str = Field(..., description="Benchmark name")
    version: str = Field(..., description="Benchmark version")
    description: str = Field(..., description="Benchmark description")
    num_tasks: int = Field(..., description="Total number of tasks")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


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


# =============================================================================
# Task-Level API Schemas (MCP-compatible)
# =============================================================================

class ToolSchema(BaseModel):
    """Schema for a tool/action."""

    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    parameters: dict[str, Any] = Field(default_factory=dict, description="JSON schema for parameters")


class ToolListResponse(BaseModel):
    """Response for listing tools."""

    tools: list[ToolSchema] = Field(..., description="List of available tools")


class ToolCallRequest(BaseModel):
    """Request to call a tool."""

    tool_name: str = Field(..., description="Name of the tool to call")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Tool arguments")


class ContentItem(BaseModel):
    """A piece of content in the response."""

    data: Any = Field(..., description="Content data (can be string, number, dict, etc.)")
    tool_call_id: str | None = Field(default=None, description="ID of the tool call this responds to")
    name: str | None = Field(default=None, description="Optional content name")


class ToolCallResponse(BaseModel):
    """Response from calling a tool."""

    content: list[ContentItem] = Field(..., description="Response content")
    isError: bool = Field(default=False, description="Whether an error occurred")


class ResourceInfo(BaseModel):
    """Resource metadata."""

    uri: str = Field(..., description="Resource URI (e.g., cube://session/abc123/observation)")
    name: str = Field(..., description="Human-readable resource name")
    description: str = Field(default="", description="Resource description")
    mimeType: str = Field(default="application/json", description="MIME type")


class ResourceListResponse(BaseModel):
    """Response for listing resources."""

    resources: list[ResourceInfo] = Field(..., description="List of available resources")


class ResourceReadResponse(BaseModel):
    """Response for reading a resource."""

    content: Any = Field(..., description="Resource content")


class EvaluationResponse(BaseModel):
    """Response for task evaluation."""

    reward: float = Field(..., description="Reward value")
    done: bool = Field(..., description="Whether the task is complete")
    step: int = Field(..., description="Current step number")
    info: dict[str, Any] = Field(default_factory=dict, description="Additional evaluation info")


class ResetRequest(BaseModel):
    """Request to reset a task."""

    seed: int | None = Field(default=None, description="Random seed for reset")


class ObservationData(BaseModel):
    """Observation data structure."""

    contents: list[ContentItem] = Field(..., description="List of content items in observation")


class ResetResponse(BaseModel):
    """Response from resetting a task."""

    observation: ObservationData = Field(..., description="Initial observation after reset")
    info: dict[str, Any] = Field(default_factory=dict, description="Additional reset info")


class CloseResponse(BaseModel):
    """Response from closing a task."""

    success: bool = Field(..., description="Whether close was successful")
    profiling: dict[str, Any] | None = Field(default=None, description="Optional profiling data")


# =============================================================================
# Health Check
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(default="ok", description="Health status")
    version: str = Field(..., description="Server version")
    environment: str = Field(..., description="Runtime environment")
