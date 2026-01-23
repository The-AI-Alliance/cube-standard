
# =============================================================================
# Task-Level API Schemas (MCP-compatible)
# =============================================================================

from typing import Any
from typing import Any
from pydantic import BaseModel, Field
from cube.core import ActionSchema, Content, Observation

# TODO:
# 1)
# copy the input/output json schemas from:
# https://github.com/The-AI-Alliance/cube-standard/blob/1eaeda50f59c31ba728614bb79eb9990017698ed/docs/api/task-level.md
# into README.md
# 2) review README.md examples, and remove stuff if too detailed/verbose
# 3) Align classes below to the examples in the README.md

# TODO: Aman check

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
