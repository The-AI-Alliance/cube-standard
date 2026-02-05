"""
Example demonstrating ToolConfig for research flexibility.

This example shows how researchers can create custom ToolConfig implementations
to swap tool behavior without modifying benchmark code.
"""

import asyncio

from counter import CounterBenchmark, ReachTargetTask
from mcp.server.fastmcp import FastMCP

from cube.tool import ToolConfig


# Example 1: Basic ToolConfig with configurable features
class CounterToolConfig(ToolConfig):
    """Custom tool configuration for counter benchmark."""

    enable_decrement: bool = False  # example config parameter: allow decrementing?
    enable_reset: bool = False  # example config parameter: allow reset?

    def create_mcp_server(self, task: ReachTargetTask) -> FastMCP:
        """Create MCP server with configurable tools."""
        mcp = FastMCP(f"Counter Task: {task.metadata.id}")

        # Core tools (always available)
        @mcp.tool()
        def increment() -> str:
            """Increment the counter by 1"""
            task.counter += 1
            task.history.append("increment")
            return f"Counter is now {task.counter}"

        @mcp.tool()
        def get_value() -> str:
            """Get the current counter value"""
            return f"Counter value is: {task.counter}"

        # Optional tools based on configuration
        if self.enable_decrement:

            @mcp.tool()
            def decrement() -> str:
                """Decrement the counter by 1"""
                task.counter -= 1
                task.history.append("decrement")
                return f"Counter is now {task.counter}"

        if self.enable_reset:

            @mcp.tool()
            def reset() -> str:
                """Reset counter to 0"""
                task.counter = 0
                task.history.append("reset")
                return "Counter reset to 0"

        return mcp


# Example 2: Advanced ToolConfig with different increment behavior
class DoubleIncrementToolConfig(ToolConfig):
    """Alternative tool implementation that increments by 2."""

    def create_mcp_server(self, task: ReachTargetTask) -> FastMCP:
        """Create MCP server with double increment."""
        mcp = FastMCP(f"Double Counter: {task.metadata.id}")

        @mcp.tool()
        def increment() -> str:
            """Increment the counter by 2 (research variant)"""
            task.counter += 2
            task.history.append("increment")
            return f"Counter is now {task.counter} (incremented by 2)"

        @mcp.tool()
        def get_value() -> str:
            """Get the current counter value"""
            return f"Counter value is: {task.counter}"

        return mcp


async def test_basic_toolconfig():
    """Test basic ToolConfig with decrement enabled."""
    print("\n=== Test 1: Basic ToolConfig with decrement ===")

    # Create benchmark with custom ToolConfig
    benchmark = CounterBenchmark()
    tool_config = CounterToolConfig(enable_decrement=True, enable_reset=False)
    benchmark.setup_benchmark_resources(tool_config=tool_config)

    # Load task
    task = benchmark.load_tasks()[0]

    # Create MCP server using ToolConfig
    from cube.server.mcp_task_server import create_task_mcp_server

    mcp_server = create_task_mcp_server(task, tool_config=tool_config)

    # List tools
    tools = await mcp_server.list_tools()
    tool_names = [tool.name for tool in tools]
    print(f"Available tools: {tool_names}")
    assert "decrement" in tool_names, "Expected 'decrement' tool"
    assert "reset" not in tool_names, "Should not have 'reset' tool"

    # Test increment
    result = await mcp_server.call_tool("increment", {})
    print(f"Increment: {result}")

    # Test decrement (only available with ToolConfig)
    result = await mcp_server.call_tool("decrement", {})
    print(f"Decrement: {result}")
    assert task.counter == 0, f"Expected counter to be 0, got {task.counter}"

    print("✓ Basic ToolConfig test passed!")


async def test_double_increment_toolconfig():
    """Test alternative tool implementation."""
    print("\n=== Test 2: Double Increment ToolConfig ===")

    # Create benchmark with different ToolConfig
    benchmark = CounterBenchmark()
    tool_config = DoubleIncrementToolConfig()
    benchmark.setup_benchmark_resources(tool_config=tool_config)

    # Load task
    task = benchmark.load_tasks()[0]

    # Create MCP server using ToolConfig
    from cube.server.mcp_task_server import create_task_mcp_server

    mcp_server = create_task_mcp_server(task, tool_config=tool_config)

    # Test double increment
    result = await mcp_server.call_tool("increment", {})
    print(f"Increment: {result}")
    assert task.counter == 2, f"Expected counter to be 2, got {task.counter}"

    # Increment again
    result = await mcp_server.call_tool("increment", {})
    print(f"Increment: {result}")
    assert task.counter == 4, f"Expected counter to be 4, got {task.counter}"

    print("✓ Double increment ToolConfig test passed!")


async def test_default_toolconfig():
    """Test default ToolConfig from benchmark."""
    print("\n=== Test 3: Default ToolConfig (from counter.py) ===")

    # Create benchmark - uses default CounterToolConfig from setup_benchmark_resources
    benchmark = CounterBenchmark()
    benchmark.setup_benchmark_resources()  # Sets up default CounterToolConfig

    # Load task
    task = benchmark.load_tasks()[0]

    # Import the default CounterToolConfig from counter.py
    from counter import CounterToolConfig as DefaultCounterToolConfig

    tool_config = DefaultCounterToolConfig()

    # Create MCP server using default ToolConfig
    from cube.server.mcp_task_server import create_task_mcp_server

    mcp_server = create_task_mcp_server(task, tool_config=tool_config)

    # List tools
    tools = await mcp_server.list_tools()
    tool_names = [tool.name for tool in tools]
    print(f"Available tools: {tool_names}")
    assert "increment" in tool_names, "Expected 'increment' tool"
    assert "get_value" in tool_names, "Expected 'get_value' tool"
    assert "decrement" not in tool_names, "Should not have 'decrement' tool (default config)"

    # Test standard increment
    result = await mcp_server.call_tool("increment", {})
    print(f"Increment: {result}")
    assert task.counter == 1, f"Expected counter to be 1, got {task.counter}"

    print("✓ Default ToolConfig test passed!")


async def main():
    """Run all ToolConfig examples."""
    print("=" * 60)
    print("ToolConfig Examples - Research Flexibility Demo")
    print("=" * 60)

    await test_basic_toolconfig()
    await test_double_increment_toolconfig()
    await test_default_toolconfig()

    print("\n" + "=" * 60)
    print("All ToolConfig examples passed!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("- ToolConfig enables swapping tool implementations")
    print("- Researchers can add/remove tools via configuration")
    print("- Researchers can change tool behavior (e.g., increment by 2)")
    print("- Every benchmark must provide a ToolConfig implementation")


if __name__ == "__main__":
    asyncio.run(main())
