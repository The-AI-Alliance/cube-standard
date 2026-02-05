"""MCP server factory for CUBE tasks.

This module provides a factory function to create MCP servers for task-specific
tools using ToolConfig. Benchmark contributors define the action space via
ToolConfig, enabling research on tool variability.
"""

from mcp.server.fastmcp import FastMCP

from cube.task import Task
from cube.tool import ToolConfig


def create_task_mcp_server(task: Task, tool_config: ToolConfig) -> FastMCP:
    """Create an MCP server for a task using ToolConfig.

    ToolConfig defines the action space (available tools) for the task.
    This enables research flexibility - researchers can swap tool implementations
    without modifying benchmark code.

    Args:
        task: The Task instance with state as attributes
        tool_config: ToolConfig that defines the MCP tools for this task

    Returns:
        FastMCP server with task tools registered

    Raises:
        ValueError: If tool_config is None or doesn't implement create_mcp_server()

    Example:
        >>> task = ReachTargetTask(task_id="count-to-5", target=5)
        >>> tool_config = CounterToolConfig()
        >>> mcp = create_task_mcp_server(task, tool_config)
    """
    if tool_config is None:
        raise ValueError(
            f"ToolConfig is required to create MCP server for task {task.metadata.id}. "
            "Benchmark must provide a ToolConfig that defines the action space."
        )

    if not hasattr(tool_config, "create_mcp_server"):
        raise ValueError(
            f"ToolConfig must implement create_mcp_server() method. "
            f"Got {type(tool_config).__name__} which doesn't implement this method."
        )

    return tool_config.create_mcp_server(task)
