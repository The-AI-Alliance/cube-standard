"""Basic test for benchmark server creation."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from cube.benchmark import Benchmark, BenchmarkMetadata, RuntimeContext
from cube.container import Container, ContainerBackend
from cube.core import Observation
from cube.server import make_benchmark_jsonrpc_app
from cube.task import Task, TaskConfig, TaskMetadata
from cube.tool import Tool, ToolConfig, tool_action


class MinimalTool(Tool):
    """Minimal tool for testing."""

    @tool_action
    def test_action(self) -> str:
        """Test action."""
        return "Test action executed"


class MinimalToolConfig(ToolConfig):
    """Minimal tool config for testing."""

    def make(self, container: Container | None = None) -> Tool:
        """Return minimal tool."""
        return MinimalTool()


class MinimalTask(Task):
    """Minimal task implementation for testing."""

    def reset(self):
        """Reset the task to its initial state."""
        self.tool.reset()
        return Observation.from_text("Test observation"), {}

    def evaluate(self, obs: Observation):
        """Minimal validation."""
        return 0.0, {}


class MinimalTaskConfig(TaskConfig):
    """Minimal task config for testing."""

    def make(
        self,
        runtime_context: RuntimeContext | None = None,
        container_backend: ContainerBackend | None = None,
    ) -> Task:
        """Create minimal task."""
        return MinimalTask(
            metadata=TaskMetadata(id=self.task_id),
            tool_config=self.tool_config or MinimalToolConfig(),
            runtime_context=runtime_context,
            container_backend=container_backend,
        )


class MinimalBenchmark(Benchmark):
    """Minimal benchmark implementation for testing."""

    benchmark_metadata = BenchmarkMetadata(
        name="TestBenchmark",
        version="1.0.0",
        description="A minimal test benchmark",
        num_tasks=2,
    )
    task_metadata = {
        "task-1": TaskMetadata(id="task-1"),
        "task-2": TaskMetadata(id="task-2"),
    }
    task_config_class = MinimalTaskConfig

    def _setup(self) -> None:
        """Minimal setup - no shared infrastructure needed."""
        pass

    def close(self):
        """Minimal close."""
        pass


def test_create_server_app():
    """Test that we can create a benchmark server app and make requests to it."""

    # Create a minimal benchmark
    benchmark = MinimalBenchmark()
    benchmark.setup()

    # Create the FastAPI app (without spawning a server process)
    app = make_benchmark_jsonrpc_app(benchmark)

    # Verify the app was created
    assert isinstance(app, FastAPI)
    assert app.title == "CUBE Benchmark Server - TestBenchmark"

    # Use TestClient to test the endpoints (without actually spawning a server process)
    with TestClient(app) as client:
        # Test cube/info method
        response = client.post("/", json={"jsonrpc": "2.0", "method": "cube/info", "params": {}, "id": 1})
        assert response.status_code == 200
        info = response.json()["result"]
        assert info["name"] == "TestBenchmark"
        assert info["version"] == "1.0.0"
        assert info["num_tasks"] == 2
        print(f"✓ cube/info returned: {info['name']}")

        # Test cube/tasks method
        response = client.post("/", json={"jsonrpc": "2.0", "method": "cube/tasks", "params": {}, "id": 2})
        assert response.status_code == 200
        tasks = response.json()["result"]
        assert len(tasks) == 2
        assert tasks[0]["id"] == "task-1"
        assert tasks[1]["id"] == "task-2"
        print(f"✓ cube/tasks returned {len(tasks)} tasks")

        # Note: Testing cube/spawn (which creates task RPC servers) is out of scope for this test

        # Test cube/shutdown method (this calls benchmark.close())
        response = client.post("/", json={"jsonrpc": "2.0", "method": "cube/shutdown", "params": {}, "id": 3})
        assert response.status_code == 200
        print("✓ cube/shutdown succeeded")

    print("✓ Benchmark server test passed!")
