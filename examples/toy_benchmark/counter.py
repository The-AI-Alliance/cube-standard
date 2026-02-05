"""Minimal counter benchmark - toy bench"""

from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import Tool as MCPTool

from cube.benchmark import Benchmark
from cube.task import Task
from cube.tool import ToolConfig
from cube.types import (
    BenchmarkMetadata,
    Observation,
    ShutdownRequest,
    ShutdownResponse,
    TaskMetadata,
)


# ToolConfig Implementation
class CounterToolConfig(ToolConfig):
    """Tool configuration for counter benchmark."""

    def create_mcp_server(self, task: Task) -> FastMCP:
        """Create MCP server with counter tools."""
        mcp = FastMCP(f"Counter Task: {task.metadata.id}")

        # Cast to ReachTargetTask for type safety
        assert isinstance(task, ReachTargetTask)
        reach_task = task

        @mcp.tool()
        def increment() -> str:
            """Increment the counter by 1"""
            reach_task.counter += 1
            reach_task.history.append("increment")
            return f"Counter is now {reach_task.counter}"

        @mcp.tool()
        def get_value() -> str:
            """Get the current counter value"""
            return f"Counter value is: {reach_task.counter}"

        return mcp


# 1. Task Implementation
class ReachTargetTask(Task):
    """Task: Increment counter to reach target value."""

    def __init__(self, task_id: str, target: int):
        """Initialize reach target task."""
        self.metadata = TaskMetadata(
            id=task_id,
            description=f"Increment counter to reach value {target}",
            tags=["counter", "simple"],
            difficulty="easy",
            domain="counter",
            max_steps=target + 2,
        )
        # State as Task attributes
        self.counter = 0
        self.target = target
        self.history: list[str] = []

    def setup(self, tool: Any) -> tuple[Observation, dict[str, Any]]:
        """Set up the task."""
        self.counter = 0
        self.history = []
        obs = Observation.from_text(f"Counter starts at 0. Use 'increment' action to reach {self.target}.")
        return obs, {"task_type": "reach_target", "target": self.target}

    def validate_task(self, obs: Observation) -> tuple[float, dict[str, Any]]:
        """Validate if counter reached target."""
        if self.counter == self.target:
            return 1.0, {
                "solved": True,
                "value": self.counter,
                "steps": len(self.history),
            }

        # Partial reward based on progress
        progress = min(1.0, self.counter / self.target) if self.target > 0 else 0.0
        return progress * 0.5, {
            "solved": False,
            "value": self.counter,
            "target": self.target,
            "steps": len(self.history),
        }

    def filter_actions(self, actions: list[MCPTool]) -> list[MCPTool]:
        """Allow all actions."""
        return actions

    def finished(self) -> bool:
        """Check if task is complete."""
        return self.counter == self.target


# 2. Benchmark Implementation
class CounterBenchmark(Benchmark):
    """Minimal benchmark with counter tasks."""

    def __init__(self):
        """Initialize counter benchmark."""
        metadata = BenchmarkMetadata(
            name="toy-counter",
            version="1.0.0",
            description="Simplest possible benchmark - count to target value",
            num_tasks=2,
            tags=["toy", "counter", "minimal"],
        )
        super().__init__(metadata=metadata)

    def setup_benchmark_resources(
        self,
        tool_config: Any = None,
        **kwargs,
    ):
        """Set up the benchmark."""
        # Use provided tool_config or default to CounterToolConfig
        if tool_config is None:
            tool_config = CounterToolConfig()
        return super().setup_benchmark_resources(tool_config=tool_config, **kwargs)

    def load_tasks(self, cache: bool = True):
        """Load counter tasks."""
        if len(self._task_list) > 0 and cache:
            return self._task_list
        self._task_list = [
            ReachTargetTask("count-to-3", target=3),
            ReachTargetTask("count-to-5", target=5),
        ]
        return self._task_list

    def shutdown(self, request: ShutdownRequest) -> ShutdownResponse:
        """Shutdown task sessions."""
        if self._session_manager is not None:
            return self._session_manager.shutdown(request)
        else:
            cleaned = []
            if request.session_id:
                if request.session_id in self._local_sessions:
                    session = self._local_sessions[request.session_id]
                    session.close()
                    del self._local_sessions[request.session_id]
                    cleaned.append(request.session_id)
            else:
                for session_id, session in list(self._local_sessions.items()):
                    session.close()
                    cleaned.append(session_id)
                self._local_sessions = {}
            return ShutdownResponse(success=True, cleaned=cleaned)

    def close(self):
        """Clean up benchmark resources."""
        if hasattr(self, "_local_sessions"):
            for session in self._local_sessions.values():
                try:
                    session.close()
                except Exception:
                    pass
            self._local_sessions = {}


# 3. Test Function
def test_simple_counting():
    """Test the counter benchmark with MCP integration."""
    print("Starting counter benchmark test with MCP integration...")

    # Create benchmark
    benchmark = CounterBenchmark()

    # Load tasks
    tasks = benchmark.load_tasks()
    assert len(tasks) == 2, "Expected 2 tasks"

    task: ReachTargetTask = tasks[0]  # type: ignore
    assert task.metadata.id == "count-to-3"
    assert task.target == 3

    # Test MCP tool registration
    import asyncio

    async def test_mcp_tools():
        # Create MCP server for the task using ToolConfig
        tool_config = CounterToolConfig()
        mcp_server = tool_config.create_mcp_server(task)

        # List tools
        tools = await mcp_server.list_tools()
        tool_names = [tool.name for tool in tools]
        assert "increment" in tool_names, "Expected 'increment' tool"
        assert "get_value" in tool_names, "Expected 'get_value' tool"
        print(f"✓ Found tools: {tool_names}")

        # Test increment tool
        result = await mcp_server.call_tool("increment", {})
        assert "Counter is now 1" in str(result), f"Unexpected result: {result}"
        print(f"✓ Increment result: {result}")

        # Test get_value tool
        result = await mcp_server.call_tool("get_value", {})
        assert "Counter value is: 1" in str(result), f"Unexpected result: {result}"
        print(f"✓ Get value result: {result}")

        # Increment twice more to reach target
        await mcp_server.call_tool("increment", {})
        await mcp_server.call_tool("increment", {})

        # Verify counter reached target
        assert task.counter == 3, f"Expected counter to be 3, got {task.counter}"
        assert task.finished(), "Task should be finished"
        print(f"✓ Counter reached target: {task.counter}")

        # Test validation
        obs = Observation.from_text("Task complete")
        reward, info = task.validate_task(obs)
        assert reward == 1.0, f"Expected reward 1.0, got {reward}"
        assert info["solved"] is True
        print(f"✓ Task validation: reward={reward}, solved={info['solved']}")

    # Run async test
    asyncio.run(test_mcp_tools())

    print("✓ All tests passed! MCP integration working correctly.")


# 7. Main
if __name__ == "__main__":
    test_simple_counting()
