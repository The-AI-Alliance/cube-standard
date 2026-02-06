"""
Task Session Management for CUBE.

This module provides the Task base class and TaskSession class which implements
the task-level API for managing individual task instances. It handles both MCP
protocol methods (tools/*, resources/*) and CUBE extensions (cube/*).
"""

import json
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from mcp.types import (
    ContentBlock,
    TextResourceContents,
)
from mcp.types import (
    Resource as MCPResource,
)
from mcp.types import (
    TextContent as MCPTextContent,
)
from mcp.types import (
    Tool as MCPTool,
)

from cube.environment import Environment
from cube.types import (
    Action,
    CloseResponse,
    EnvironmentOutput,
    MCPCallToolRequest,
    MCPCallToolResult,
    MCPListResourcesResult,
    MCPListToolsResult,
    MCPReadResourceResult,
    Observation,
    ResetRequest,
    ResetResponse,
    StepRequest,
    StepResponse,
    TaskMetadata,
    TaskStatus,
    TaskStatusEnum,
)

logger = logging.getLogger(__name__)


class Task(ABC):
    """Represents a task that an agent must complete in an environment."""

    metadata: TaskMetadata
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
        >>> session = TaskSession(task_id="task-1", env=env)
        >>>
        >>> # MCP protocol
        >>> tools = await session.list_tools()
        >>> result = session.call_tool(MCPCallToolRequest(tool_name="click", arguments={"x": 10}))
        >>>
        >>> # CUBE extensions
        >>> state = session.evaluate()
        >>> session.reset(ResetRequest(seed=42))
    """

    def __init__(self, task_id: str, env: Environment, mcp_server: Any = None):
        """
        Initialize a new task session.

        Args:
            task_id: The task ID this session is running
            env: Environment instance wrapping the task and tool
            mcp_server: Optional in-memory MCP server for task-specific tools
        """
        self.session_id = str(uuid.uuid4())
        self.task_id = task_id
        self.env = env
        self.mcp_server = mcp_server  # In-memory MCP server reference
        self.step_count = 0
        self.total_reward = 0.0
        self.created_at = datetime.now()
        self.last_state: EnvironmentOutput | None = None
        self.last_updated: datetime | None = None
        self.status: TaskStatusEnum = TaskStatusEnum.stopped

    @property
    def seed(self) -> int | None:
        return self.env.task.seed

    def get_status(self) -> TaskStatus:
        """
        Get the current status of the task session.

        Returns:
            TaskStatus with current session information
        """
        status = TaskStatus(
            session_id=self.session_id,
            task_id=self.task_id,
            status=self.status,
            created_at=self.created_at,
            step_count=self.step_count,
            last_updated=self.last_updated,
            other={
                "total_reward": self.total_reward,
                "last_state": self.last_state,
            },
        )
        return status

    # =============================================================================
    # MCP Protocol Methods
    # =============================================================================

    async def list_tools(self) -> MCPListToolsResult:
        """
        List available tools/actions for this task (MCP tools/list).

        Returns:
            MCPListToolsResult with list of available ActionSchema objects

        Raises:
            TaskClosedException: If the session has been closed

        ==> Source code in .venv/lib/python3.12/site-packages/mcp/client/session.py
        """
        logger.debug(f"[ENTRY] TaskSession.list_tools - {self.session_id} listing tools")
        if self.status == TaskStatusEnum.stopped:
            raise TaskClosedException(self.session_id)

        # Get tools from MCP server if available
        if self.mcp_server:
            # FastMCP server provides async list_tools() method
            tools = await self.mcp_server.list_tools()
            filtered_tools = self.env.task.filter_actions(
                tools
            )  # TODO: filter_actions expects list[Tool], may need to adapt for MCPListToolsResult
            return MCPListToolsResult(tools=filtered_tools)

        # Fallback to environment actions (for backwards compatibility)
        actions = self.env.get_actions()
        filtered_actions = self.env.task.filter_actions(actions)

        return MCPListToolsResult(tools=filtered_actions)

    async def call_tool(self, request: MCPCallToolRequest) -> MCPCallToolResult:
        """
        Execute a tool/action (MCP tools/call).

        Args:
            request: MCPCallToolRequest with tool name and arguments

        Returns:
            MCPCallToolResult with content and isError flag

        Raises:
            TaskClosedException: If the session has been closed

        ==> Source code in .venv/lib/python3.12/site-packages/mcp/client/session.py
        """
        if self.status == TaskStatusEnum.stopped:
            raise TaskClosedException(self.session_id)

        tool_name = request.params.name
        tool_args = request.params.arguments or {}
        try:
            # Call MCP server directly (in-memory, no HTTP)
            if self.mcp_server:
                # FastMCP server provides async call_tool() method
                # Returns Sequence[ContentBlock] or dict
                result = await self.mcp_server.call_tool(tool_name, tool_args)

                # Extract text from result (FastMCP returns ContentBlock sequence or dict)
                if isinstance(result, dict):
                    result_text = str(result)
                elif hasattr(result, "__iter__"):
                    # ContentBlock sequence - extract text from first text block
                    result_text = next((block.text for block in result if hasattr(block, "text")), str(result))
                else:
                    result_text = str(result)
            else:
                # Fallback to environment step (for backwards compatibility)
                action = Action(
                    id=str(uuid.uuid4()),
                    name=tool_name,
                    arguments=tool_args,
                )
                result = self.env.step(action)
                result_text = str(result.obs)

            # Update tracking
            self.step_count += 1
            self.last_updated = datetime.now()

            # Create observation for validation
            obs = Observation.from_text(result_text)

            # Check if task is complete or validate per step
            if self.env.task.validate_per_step or self.env.task.finished():
                reward, info = self.env.task.validate_task(obs)
                self.total_reward += reward
                self.last_state = EnvironmentOutput(
                    obs=obs,
                    reward=reward,
                    terminated=self.env.task.finished(),
                    truncated=False,
                    info=info,
                )

            # Return MCP format
            mcp_content: list[ContentBlock] = [MCPTextContent(type="text", text=result_text)]
            return MCPCallToolResult(content=mcp_content, isError=False)

        except Exception as e:
            logger.exception(f"Tool execution failed: {e}")
            error_content: list[ContentBlock] = [
                MCPTextContent(type="text", text=f"Error executing tool {tool_name}: {str(e)}")
            ]
            return MCPCallToolResult(content=error_content, isError=True)

    def list_resources(self) -> MCPListResourcesResult:
        """
        List available resources (MCP resources/list).

        Standard resources include:
        - task://description - The task description
        - obs://current - Current observation state

        Returns:
            MCPListResourcesResult with list of available resources

        Raises:
            TaskClosedException: If the session has been closed

        ==> Source code in .venv/lib/python3.12/site-packages/mcp/client/session.py
        """
        if self.status == TaskStatusEnum.stopped:
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

        return MCPListResourcesResult(resources=resources)

    def read_resource(self, uri: str) -> MCPReadResourceResult:
        """
        Read a specific resource by URI (MCP resources/read).

        Standard URIs:
        - task://description - Returns the task description as text
        - obs://current - Returns the current observation as JSON

        Args:
            uri: Resource URI to read

        Returns:
            MCPReadResourceResult with resource content

        Raises:
            TaskClosedException: If the session has been closed
            ResourceNotFoundException: If the URI is not found

        ==> Source code in .venv/lib/python3.12/site-packages/mcp/client/session.py
        """
        if self.status == TaskStatusEnum.stopped:
            raise TaskClosedException(self.session_id)

        if uri == "task://description":
            description = self.env.task.metadata.description
            return MCPReadResourceResult(
                contents=[TextResourceContents(uri=uri, mimeType="text/plain", text=description)]  # type: ignore
            )

        if uri == "obs://current":
            if self.last_state is None:
                return MCPReadResourceResult(
                    contents=[
                        TextResourceContents(
                            uri=uri,
                            mimeType="text/plain",
                            text="No observation yet. Call reset() first.",
                        )
                    ]  # type: ignore
                )

            # Convert observation contents to JSON string
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
            return MCPReadResourceResult(
                contents=[TextResourceContents(uri=uri, mimeType="application/json", text=json.dumps(obs_data))]
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
        if self.status == TaskStatusEnum.stopped:
            raise TaskClosedException(self.session_id)

        if self.last_state is None:
            return EnvironmentOutput(
                obs=Observation.from_text("No observation yet. Call reset() first."),
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

    async def step(self, request: StepRequest) -> StepResponse:
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
        if self.status == TaskStatusEnum.stopped:
            raise TaskClosedException(self.session_id)

        tool_request = MCPCallToolRequest(params=request.params)
        await self.call_tool(tool_request)
        evaluation = self.evaluate()
        return StepResponse(response=evaluation)

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
        if self.status == TaskStatusEnum.stopped:
            raise TaskClosedException(self.session_id)

        try:
            result = self.env.reset()
            result.info["task_id"] = self.task_id
            result.info["session_id"] = self.session_id
            result.info["seed"] = request.seed

            self.step_count = 0
            self.total_reward = 0.0
            self.last_state = result
            self.last_updated = datetime.now()

            return ResetResponse(response=result)

        except Exception as e:
            logger.exception(f"Reset failed: {e}")
            raise ValueError(f"Failed to reset task: {str(e)}")

    def close(self) -> CloseResponse:
        """
        Cleanup task resources and close the session (CUBE cube/close).

        Returns:
            CloseResponse with success flag and profiling data
        """
        if self.status == TaskStatusEnum.stopped:
            return CloseResponse(success=True, profiling=None)

        try:
            self.env.close()
            self.status = TaskStatusEnum.stopped
            self.last_updated = datetime.now()

            profiling = {
                "total_steps": self.step_count,
                "total_reward": self.total_reward,
                "duration_seconds": (datetime.now() - self.created_at).total_seconds(),
            }

            logger.info(f"Session {self.session_id} closed successfully")
            return CloseResponse(success=True, profiling=profiling)

        except Exception as e:
            logger.exception(f"Error closing session: {e}")
            return CloseResponse(success=False, profiling=None)
