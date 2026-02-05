"""
Tool configuration for CUBE benchmarks.

ToolConfig allows researchers to swap MCP server implementations for experimentation,
enabling research on different tool sets, implementations, and configurations without
modifying benchmark code.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from mcp.server.fastmcp import FastMCP

from cube.types import TypedBaseModel

# Forward reference to avoid circular import
if TYPE_CHECKING:
    from cube.task import Task


class ToolConfig(TypedBaseModel, ABC):
    """
    Configuration for creating MCP servers with task-specific tools.

    ToolConfig enables research on tool variability by allowing researchers to:
    - Swap out different tool implementations (e.g., Playwright vs Selenium)
    - Provide different tool sets (e.g., basic vs advanced browser tools)
    - Use different MCP server implementations
    - Configure tool behavior (e.g., browser types, shell environments)

    Example:
        >>> class BrowserToolConfig(ToolConfig):
        ...     browser_type: str = "chromium"
        ...     headless: bool = True
        ...
        ...     def create_mcp_server(self, task: WebTask) -> FastMCP:
        ...         mcp = FastMCP(f"Browser: {task.id}")
        ...
        ...         @mcp.tool()
        ...         def navigate(url: str) -> str:
        ...             return task.navigate_with_browser(url, self.browser_type)
        ...
        ...         @mcp.tool()
        ...         def click(selector: str) -> str:
        ...             return task.click_element(selector)
        ...
        ...         return mcp
    """

    @abstractmethod
    def create_mcp_server(self, task: "Task") -> FastMCP:
        """
        Create and configure an MCP server for the given task.

        This method provides full control over MCP server creation:
        - Choose which tools to register
        - Implement tools with different behaviors
        - Configure tool parameters based on research needs

        Args:
            task: The task instance with state and metadata. Task state
                  (e.g., self.counter, self.browser) can be accessed via closure.

        Returns:
            FastMCP server with tools registered

        Example:
            >>> def create_mcp_server(self, task: CounterTask) -> FastMCP:
            ...     mcp = FastMCP(f"Counter: {task.id}")
            ...
            ...     @mcp.tool()
            ...     def increment() -> str:
            ...         task.counter += 1
            ...         return f"Counter is now {task.counter}"
            ...
            ...     if self.enable_decrement:  # Configurable feature
            ...         @mcp.tool()
            ...         def decrement() -> str:
            ...             task.counter -= 1
            ...             return f"Counter is now {task.counter}"
            ...
            ...     return mcp
        """
        raise NotImplementedError("Subclasses must implement create_mcp_server() method.")
