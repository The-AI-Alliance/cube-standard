"""Basic test for benchmark server creation."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from cube.benchmark import Benchmark, BenchmarkMetadata, RuntimeContext
from cube.containers import ContainerBackend
from cube.core import Observation
from cube.server import make_benchmark_fastapi_app
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

    def make(self) -> Tool:
        """Return minimal tool."""
        return MinimalTool()


class MinimalTask(Task):
    """Minimal task implementation for testing."""

    tool: MinimalTool  # type: ignore[assignment]

    def __init__(self, task_id: str):
        self.metadata = TaskMetadata(id=task_id, description="Test task")

    def setup(self):
        """Minimal setup."""
        self.tool.reset()
        return Observation.from_text("Test observation"), {}

    def evaluate(self, obs: Observation):
        """Minimal validation."""
        return 0.0, {}


class MinimalTaskConfig(TaskConfig):
    """Minimal task config for testing."""

    task_id: str
    tool_config: ToolConfig

    def make(
        self, runtime_context: RuntimeContext | None = None, container_backend: ContainerBackend | None = None
    ) -> Task:
        """Create minimal task."""
        tool = self.tool_config.make()
        task = MinimalTask(task_id=self.task_id)
        task.tool = tool  # type: ignore[assignment]
        task.runtime_context = runtime_context
        return task


class MinimalBenchmark(Benchmark):
    """Minimal benchmark implementation for testing."""

    def __init__(self):
        super().__init__(
            metadata=BenchmarkMetadata(
                name="TestBenchmark",
                version="1.0.0",
                description="A minimal test benchmark",
                num_tasks=2,
            ),
        )

    def setup(self) -> RuntimeContext:
        """Minimal setup."""
        runtime_context = RuntimeContext()
        self._runtime_info = runtime_context
        return runtime_context

    def load_tasks(self, cache: bool = True):
        """Return a list of minimal tasks."""
        if len(self._task_list) > 0 and cache:
            return self._task_list

        self._task_list = [
            MinimalTaskConfig(task_id="task-1", tool_config=MinimalToolConfig()),
            MinimalTaskConfig(task_id="task-2", tool_config=MinimalToolConfig()),
        ]
        return self._task_list

    def close(self):
        """Minimal close."""
        pass


def test_create_server_app():
    """Test that we can create a benchmark server app and make requests to it."""

    # Create a minimal benchmark
    benchmark = MinimalBenchmark()
    benchmark.setup()

    # Create the FastAPI app (without spawning a server process)
    app = make_benchmark_fastapi_app(benchmark)

    # Verify the app was created
    assert isinstance(app, FastAPI)
    assert app.title == "CUBE Benchmark Server - TestBenchmark"

    # Use TestClient to test the endpoints (without actually spawning a server process)
    with TestClient(app) as client:
        # Test /cube/info endpoint
        response = client.get("/cube/info")
        assert response.status_code == 200
        info = response.json()
        assert info["name"] == "TestBenchmark"
        assert info["version"] == "1.0.0"
        assert info["num_tasks"] == 2
        print(f"✓ /cube/info returned: {info['name']}")

        # Test /cube/tasks endpoint
        response = client.get("/cube/tasks")
        assert response.status_code == 200
        tasks = response.json()
        assert len(tasks) == 2
        assert tasks[0]["task_id"] == "task-1"
        assert tasks[1]["task_id"] == "task-2"
        print(f"✓ /cube/tasks returned {len(tasks)} tasks")

        # Note: Testing /cube/spawn (which creates task RPC servers) is out of scope for this test

        # Test /cube/shutdown endpoint (this calls benchmark.close())
        response = client.post("/cube/shutdown")
        assert response.status_code == 200
        print("✓ /cube/shutdown succeeded")

    print("✓ Benchmark server test passed!")
