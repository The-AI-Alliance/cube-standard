"""Basic test for benchmark server creation."""

from fastapi import FastAPI

from cube.benchmark import Benchmark
from cube.server import create_benchmark_server_app
from cube.task import Task
from cube.tool import ToolConfig
from cube.types import (
    BenchmarkMetadata,
    Observation,
    ShutdownRequest,
    ShutdownResponse,
    TaskMetadata,
)


def test_create_server_app():
    """Test that we can create a benchmark server app."""

    class MinimalToolConfig(ToolConfig):
        """Minimal tool config for testing."""

        def make(self):
            """Return None as we don't need actual tools for this test."""
            return None

    class MinimalTask(Task):
        """Minimal task implementation for testing."""

        def __init__(self, task_id: str):
            self.metadata = TaskMetadata(id=task_id, description="Test task")

        def setup(self, tool):
            """Minimal setup."""
            self._tool = tool
            return Observation.from_text("Test observation"), {}

        def validate_task(self, obs: Observation):
            """Minimal validation."""
            return 0.0, {}

        def filter_actions(self, actions):
            """Return all actions."""
            return actions

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
                tool_config=MinimalToolConfig(),
            )

        def setup_benchmark_resources(
            self,
            tool_config,
        ):
            """Minimal setup."""
            return super().setup_benchmark_resources(tool_config=tool_config)

        def load_tasks(self, cache: bool = True):
            """Return a list of minimal tasks."""
            return [MinimalTask("task-1"), MinimalTask("task-2")]

        def shutdown(self, request: ShutdownRequest):
            """Minimal shutdown."""
            return ShutdownResponse(success=True, cleaned=[])

        def close(self):
            """Minimal close."""
            pass

    # Create a minimal benchmark
    benchmark = MinimalBenchmark()

    # Create the server app
    app = create_benchmark_server_app(benchmark)

    # Verify it's a FastAPI app
    assert isinstance(app, FastAPI)

    # Verify the app title matches the benchmark name
    assert app.title == "CUBE Benchmark Server - TestBenchmark"
