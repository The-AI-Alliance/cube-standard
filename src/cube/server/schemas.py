"""API request and response schemas for CUBE server."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from cube.core import ActionSchema, Content, Observation


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
# Task-Level API Schemas (MCP-compatible)
# =============================================================================

class ToolListResponse(BaseModel):
    """Response for listing tools."""

    tools: list[ActionSchema] = Field(..., description="List of available tools")


class ToolCallRequest(BaseModel):
    """Request to call a tool."""

    tool_name: str = Field(..., description="Name of the tool to call")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Tool arguments")


class ToolCallResponse(BaseModel):
    """Response from calling a tool."""

    content: list[Content] = Field(..., description="Response content")
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


class ResetRequest(BaseModel):
    """Request to reset a task."""

    seed: int | None = Field(default=None, description="Random seed for reset")


class ResetResponse(BaseModel):
    """Response from resetting a task."""

    observation: Observation = Field(..., description="Initial observation after reset")
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
