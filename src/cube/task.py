"""
Task Session Management for CUBE.

This module provides the TaskSession class which implements the task-level API
for managing individual task instances. It handles both MCP protocol methods
(tools/*, resources/*) and CUBE extensions (cube/*).
"""

import logging
import uuid
from datetime import datetime

from mcp.types import (
    Resource as MCPResource,
    TextContent as MCPTextContent,
    TextResourceContents,
)

from cube.types import (
    CloseResponse,
    ResetRequest,
    ResetResponse,
    ResourceListResponse,
    ResourceReadResponse,
    StepRequest,
    StepResponse,
    ToolCallRequest,
    ToolCallResponse,
    ToolListResponse,
)
from cube.types import Action, EnvironmentOutput, Observation
from cube.environment import Environment

logger = logging.getLogger(__name__)


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


class TaskSession:
    """
    Manages a single benchmark task session.

    A TaskSession wraps an Environment and implements both the MCP protocol
    (tools/list, tools/call, resources/list, resources/read) and CUBE extensions
    (cube/evaluation, cube/step, cube/reset, cube/close).

    This class is part of the CUBE specification and can be used by any
    server implementation (FastAPI, Flask, gRPC, etc.) or used directly
    for in-process task management.

    Example:
        >>> from cube.session import TaskSession
        >>> from cube.environment import Environment
        >>>
        >>> env = Environment(task, tool)
        >>> session = TaskSession(session_id="abc-123", task_id="task-1", env=env)
        >>>
        >>> # MCP protocol
        >>> tools = session.list_tools()
        >>> result = session.call_tool(ToolCallRequest(tool_name="click", arguments={"x": 10}))
        >>>
        >>> # CUBE extensions
        >>> state = session.evaluate()
        >>> session.reset(ResetRequest(seed=42))
    """

    def __init__(self, session_id: str, task_id: str, env: Environment):
        """
        Initialize a new task session.

        Args:
            session_id: Unique identifier for this session
            task_id: The task ID this session is running
            env: Environment instance wrapping the task and tool
        """
        self.session_id = session_id
        self.task_id = task_id
        self.env = env
        self.step_count = 0
        self.total_reward = 0.0
        self.created_at = datetime.now()
        self.last_state: EnvironmentOutput | None = None
        self.is_closed = False

    # =============================================================================
    # MCP Protocol Methods
    # =============================================================================

    def list_tools(self) -> ToolListResponse:
        """
        List available tools/actions for this task (MCP tools/list).

        Returns:
            ToolListResponse with list of available ActionSchema objects

        Raises:
            TaskClosedException: If the session has been closed
        """
        if self.is_closed:
            raise TaskClosedException(self.session_id)

        actions = self.env.get_actions()
        filtered_actions = self.env.task.filter_actions(actions)

        return ToolListResponse(tools=filtered_actions)

    def call_tool(self, request: ToolCallRequest) -> ToolCallResponse:
        """
        Execute a tool/action (MCP tools/call).

        Args:
            request: ToolCallRequest with tool name and arguments

        Returns:
            ToolCallResponse with content and isError flag

        Raises:
            TaskClosedException: If the session has been closed
        """
        if self.is_closed:
            raise TaskClosedException(self.session_id)

        try:
            action = Action(
                id=str(uuid.uuid4()),
                name=request.name,  # Updated from request.tool_name to request.name (MCP format)
                arguments=request.arguments,
            )

            result = self.env.step(action)

            self.step_count += 1
            self.total_reward += result.reward
            self.last_state = result

            # Convert CUBE Content to MCP TextContent
            mcp_content = [
                MCPTextContent(type="text", text=str(c.data))
                for c in result.obs.contents
            ]

            return ToolCallResponse(content=mcp_content, isError=False)

        except Exception as e:
            logger.exception(f"Tool execution failed: {e}")
            error_content = [
                MCPTextContent(
                    type="text", text=f"Error executing tool {request.name}: {str(e)}"
                )
            ]
            return ToolCallResponse(content=error_content, isError=True)

    def list_resources(self) -> ResourceListResponse:
        """
        List available resources (MCP resources/list).

        Standard resources include:
        - task://description - The task description
        - obs://current - Current observation state

        Returns:
            ResourceListResponse with list of available resources

        Raises:
            TaskClosedException: If the session has been closed
        """
        if self.is_closed:
            raise TaskClosedException(self.session_id)

        resources = [
            MCPResource(
                uri="task://description",
                name="Task Description",
                description="The goal of this task",
                mimeType="text/plain",
            ),
            MCPResource(
                uri="obs://current",
                name="Current Observation",
                description="Current state of the environment",
                mimeType="application/json",
            ),
        ]

        # Allow tasks to provide additional resources
        if hasattr(self.env.task, "get_resources"):
            additional_resources = self.env.task.get_resources()
            resources.extend(additional_resources)

        return ResourceListResponse(resources=resources)

    def read_resource(self, uri: str) -> ResourceReadResponse:
        """
        Read a specific resource by URI (MCP resources/read).

        Standard URIs:
        - task://description - Returns the task description as text
        - obs://current - Returns the current observation as JSON

        Args:
            uri: Resource URI to read

        Returns:
            ResourceReadResponse with resource content

        Raises:
            TaskClosedException: If the session has been closed
            ResourceNotFoundException: If the URI is not found
        """
        if self.is_closed:
            raise TaskClosedException(self.session_id)

        if uri == "task://description":
            description = self.env.task.metadata.description
            return ResourceReadResponse(
                contents=[
                    TextResourceContents(
                        uri=uri, mimeType="text/plain", text=description
                    )
                ]
            )

        if uri == "obs://current":
            if self.last_state is None:
                return ResourceReadResponse(
                    contents=[
                        TextResourceContents(
                            uri=uri,
                            mimeType="text/plain",
                            text="No observation yet. Call reset() first.",
                        )
                    ]
                )

            # Convert observation contents to JSON string
            import json

            obs_data = {
                "contents": [
                    {
                        "data": str(c.data),
                        "tool_call_id": c.tool_call_id,
                        "name": c.name,
                        "type": c.type,
                    }
                    for c in self.last_state.obs.contents
                ]
            }
            return ResourceReadResponse(
                contents=[
                    TextResourceContents(
                        uri=uri, mimeType="application/json", text=json.dumps(obs_data)
                    )
                ]
            )

        # Allow tasks to handle custom resource URIs
        if hasattr(self.env.task, "read_resource"):
            try:
                return self.env.task.read_resource(uri)
            except Exception:
                pass

        raise ResourceNotFoundException(uri)

    # =============================================================================
    # CUBE Extension Methods
    # =============================================================================

    def evaluate(self) -> EnvironmentOutput:
        """
        Get current evaluation state (CUBE cube/evaluation).

        Returns observation, reward, terminated, truncated, and info for the
        current task state.

        Returns:
            EnvironmentOutput with current task evaluation

        Raises:
            TaskClosedException: If the session has been closed
        """
        if self.is_closed:
            raise TaskClosedException(self.session_id)

        if self.last_state is None:
            return EnvironmentOutput(
                obs=Observation.from_text(
                    "No observation yet. Call reset() first."
                ),
                reward=0.0,
                terminated=False,
                truncated=False,
                info={"error": "Task not initialized"},
            )

        # Enrich the info with session-level data
        enriched_info = {
            "step_count": self.step_count,
            "total_reward": self.total_reward,
            **self.last_state.info,
        }

        return EnvironmentOutput(
            obs=self.last_state.obs,
            reward=self.last_state.reward,
            terminated=self.last_state.terminated,
            truncated=self.last_state.truncated,
            step=self.last_state.step,
            info=enriched_info,
        )

    def step(self, request: StepRequest) -> StepResponse:
        """
        Execute a tool and return both tool result and evaluation (CUBE cube/step).

        This is a convenience method that combines tools/call and cube/evaluation
        into a single RPC call to reduce latency.

        Args:
            request: StepRequest with tool name and arguments

        Returns:
            StepResponse with both tool result and evaluation

        Raises:
            TaskClosedException: If the session has been closed
        """
        if self.is_closed:
            raise TaskClosedException(self.session_id)

        tool_request = ToolCallRequest(name=request.name, arguments=request.arguments)
        tool_result = self.call_tool(tool_request)

        evaluation = self.evaluate()

        return StepResponse(tool_result=tool_result, evaluation=evaluation)

    def reset(self, request: ResetRequest) -> ResetResponse:
        """
        Reset the task to its initial state (CUBE cube/reset).

        Args:
            request: ResetRequest with optional seed

        Returns:
            ResetResponse with initial observation and info

        Raises:
            TaskClosedException: If the session has been closed
            ValueError: If reset fails
        """
        if self.is_closed:
            raise TaskClosedException(self.session_id)

        try:
            result = self.env.reset()

            self.step_count = 0
            self.total_reward = 0.0
            self.last_state = result

            return ResetResponse(
                observation=result.obs,
                info={
                    "task_id": self.task_id,
                    "session_id": self.session_id,
                    "seed": request.seed,
                    **result.info,
                },
            )

        except Exception as e:
            logger.exception(f"Reset failed: {e}")
            raise ValueError(f"Failed to reset task: {str(e)}")

    def close(self) -> CloseResponse:
        """
        Cleanup task resources and close the session (CUBE cube/close).

        Returns:
            CloseResponse with success flag and profiling data
        """
        if self.is_closed:
            return CloseResponse(success=True, profiling=None)

        try:
            self.env.close()
            self.is_closed = True

            profiling = {
                "total_steps": self.step_count,
                "total_reward": self.total_reward,
                "duration_seconds": (
                    datetime.now() - self.created_at
                ).total_seconds(),
            }

            logger.info(f"Session {self.session_id} closed successfully")
            return CloseResponse(success=True, profiling=profiling)

        except Exception as e:
            logger.exception(f"Error closing session: {e}")
            return CloseResponse(success=False, profiling=None)


