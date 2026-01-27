from abc import ABC, abstractmethod

from cube.types import (
    BenchmarkMetadata,
    ShutdownRequest,
    ShutdownResponse,
    SpawnRequest,
    SpawnResponse,
    StatusRequest,
    StatusResponse,
    TaskListResponse,
    TaskRequest,
    TypedBaseModel,
)
from cube.core import Task
from cube.environment import EnvConfig
from cube.tool import ToolConfig


class Benchmark(TypedBaseModel, ABC):
    """Represents a benchmark consisting of multiple tasks and an environment."""

    metadata: BenchmarkMetadata  # cube/info returns this
    tool_config: ToolConfig  # set in setup()

    @abstractmethod
    def setup(self, available_ports: list[int], tool_config: ToolConfig) -> None:
        """
        Perform common steps necessary to prepare the environment for all tasks,
        like running web server, launching containers, etc.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Clean up resources after all tasks are done.
        """
        pass

    @abstractmethod
    def load_tasks(self) -> list[Task]:
        """
        Load and return the list of tasks for this benchmark.
        """
        pass

    def info(self) -> BenchmarkMetadata:
        """
        Return benchmark metadata.
        cube/info calls this method.
        """
        return self.metadata

    def list_tasks(self, request: TaskRequest) -> TaskListResponse:
        """
        List tasks with optional filtering, offset, and limit.
        cube/tasks calls this method.
        """
        tasks = self.load_tasks()
        total = len(tasks)

        # Apply filtering
        if request.task_id is not None:
            tasks = [task for task in tasks if task.id == request.task_id]

        # Apply offset and limit
        if request.limit == -1:
            limited_tasks = tasks[request.offset:]
        else:
            limited_tasks = tasks[request.offset : request.offset + request.limit]

        task_metadata_list = [
            task.metadata for task in limited_tasks
        ]

        return TaskListResponse(
            tasks=task_metadata_list,
            total=total,
            offset=request.offset,
            limit=request.limit,
        )

    def env_configs(self) -> list[EnvConfig]:
        """Generate environment configurations for all tasks in the benchmark."""
        tasks = self.load_tasks()
        configs = [EnvConfig(task=task, tool_config=self.tool_config) for task in tasks]
        return configs

    def install(self):
        """
        Optional method to download and prepare any resources required by the benchmark.
        """
        pass

    def uninstall(self):
        """
        Optional method to remove any resources used by the benchmark.
        """
        pass

    @abstractmethod
    def spawn(self, request: SpawnRequest) -> SpawnResponse:
        """
        Spawn a new session for a given task.
        cube/spawn calls this method.
        """
        pass

    @abstractmethod
    def get_task_status(self, request: StatusRequest) -> StatusResponse:
        """
        Get the status of running tasks.
        cube/status calls this method.
        """
        pass

    @abstractmethod
    def shutdown(self, request: ShutdownRequest) -> ShutdownResponse:
        """
        Shutdown a running task session.
        cube/shutdown calls this method.
        """
        pass
