"""Minimal counter benchmark - toy bench"""

from typing import Any, Protocol

from mcp.types import Tool as MCPTool

from cube.benchmark import Benchmark
from cube.task import Task
from cube.tool import AbstractTool, Tool, ToolConfig
from cube.types import (
    BenchmarkMetadata,
    MCPCallToolRequest,
    MCPCallToolRequestParams,
    Observation,
    ResetRequest,
    ShutdownRequest,
    ShutdownResponse,
    SpawnRequest,
    TaskMetadata,
)


# 1. Action Space
class CounterActions(Protocol):
    """Protocol defining counter actions."""

    def increment(self) -> str:
        """Increment counter by 1."""
        ...

    def get_value(self) -> str:
        """Get current counter value."""
        ...


# 2. Tool Implementation
class CounterTool(Tool):
    """A simple counter tool with minimal state."""

    action_space = CounterActions

    def __init__(self):
        """Initialize counter at 0."""
        self.value = 0
        self.history = []

    def reset(self) -> None:
        """Reset counter to 0."""
        self.value = 0
        self.history = []

    def increment(self) -> str:
        """Increment counter by 1."""
        self.value += 1
        self.history.append("increment")
        return "Counter incremented"

    def get_value(self) -> str:
        """Get current counter value."""
        return f"Counter value is: {self.value}"

    def close(self) -> None:
        """Clean up resources."""
        self.history = []


# 3. Tool Config
class CounterToolConfig(ToolConfig):
    """Configuration for counter tool."""

    def make(self) -> AbstractTool:
        """Create and return a CounterTool instance."""
        return CounterTool()


# 4. Task Implementation
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
        self.target = target
        self._tool = None

    def setup(self, tool: AbstractTool) -> tuple[Observation, dict[str, Any]]:
        """Set up the task."""
        self._tool = tool
        obs = Observation.from_text(f"Counter starts at 0. Use 'increment' action to reach {self.target}.")
        return obs, {"task_type": "reach_target", "target": self.target}

    def validate_task(self, obs: Observation) -> tuple[float, dict[str, Any]]:
        """Validate if counter reached target."""
        if self._tool.value == self.target:
            return 1.0, {
                "solved": True,
                "value": self._tool.value,
                "steps": len(self._tool.history),
            }

        # Partial reward based on progress
        progress = min(1.0, self._tool.value / self.target) if self.target > 0 else 0.0
        return progress * 0.5, {
            "solved": False,
            "value": self._tool.value,
            "target": self.target,
            "steps": len(self._tool.history),
        }

    def filter_actions(self, actions: list[MCPTool]) -> list[MCPTool]:
        """Allow all actions."""
        return actions

    def finished(self) -> bool:
        """Check if task is complete."""
        return self._tool.value == self.target


# 5. Benchmark Implementation
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
        tool_config = CounterToolConfig()
        super().__init__(metadata=metadata, tool_config=tool_config)

    def setup(
        self,
        available_ports: list[int],
        tool_config: Any = None,
        server_mode: bool = False,
        server_host: str = "localhost",
        server_port: int = 8000,
    ):
        """Set up the benchmark."""
        if tool_config is not None:
            self.tool_config = tool_config

        return super().setup(
            available_ports=available_ports,
            tool_config=self.tool_config,
            server_mode=server_mode,
            server_host=server_host,
            server_port=server_port,
        )

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


# 6. Test Function
def test_simple_counting():
    """Test the counter benchmark."""
    print("Starting counter benchmark test...")

    # Create and setup benchmark
    benchmark = CounterBenchmark()
    benchmark.setup(available_ports=[9000], server_mode=False)

    # Spawn task "count-to-3"
    spawn_resp = benchmark.spawn(SpawnRequest(task_id="count-to-3"))
    session = spawn_resp.other["session"]

    # Reset task
    session.reset(ResetRequest())

    # Call increment() 3 times
    for i in range(3):
        result = session.call_tool(MCPCallToolRequest(params=MCPCallToolRequestParams(name="increment", arguments={})))
        print(f"Step {i + 1}: {result.content[0].text}")

    # Evaluate - should be solved
    eval_result = session.evaluate()
    print(f"\nEvaluation: reward={eval_result.reward}, info={eval_result.info}")

    assert eval_result.reward == 1.0, f"Expected reward 1.0, got {eval_result.reward}"
    assert eval_result.info["solved"], "Task should be solved"

    # Cleanup
    benchmark.shutdown(ShutdownRequest(session_id=spawn_resp.session_id))
    benchmark.close()

    print("\n✓ Test passed!")


# 7. Main
if __name__ == "__main__":
    test_simple_counting()
