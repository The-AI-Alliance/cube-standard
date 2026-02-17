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
        self.metadata = TaskMetadata(id=task_id)

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
    seed: int | None = None

    def make(
        self,
        metadata: TaskMetadata,
        runtime_context: RuntimeContext | None = None,
        container_backend: ContainerBackend | None = None,
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

    def _setup(self) -> None:
        """Minimal setup."""
        # Define task metadata
        self.task_list = [
            TaskMetadata(id="task-1", abstract_description="Test task 1"),
            TaskMetadata(id="task-2", abstract_description="Test task 2"),
        ]

        # Set TaskConfig class
        self._task_config_class = MinimalTaskConfig

        # Set default tool config
        self._default_tool_config = MinimalToolConfig()

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
        assert tasks[0]["id"] == "task-1"
        assert tasks[1]["id"] == "task-2"
        print(f"✓ /cube/tasks returned {len(tasks)} tasks")

        # Note: Testing /cube/spawn (which creates task RPC servers) is out of scope for this test

        # Test /cube/shutdown endpoint (this calls benchmark.close())
        response = client.post("/cube/shutdown")
        assert response.status_code == 200
        print("✓ /cube/shutdown succeeded")

    print("✓ Benchmark server test passed!")
