import logging
from abc import ABC, abstractmethod
from typing import Any

from pydantic import PrivateAttr

from cube.task import Task
from cube.types import (
    BenchmarkMetadata,
    SpawnResponse,
    TypedBaseModel,
)
from design.core import TaskConfig

logger = logging.getLogger(__name__)


# TODO: implement actual server spawning and return real endpoint
def make_task_rpc_server(task: Task) -> SpawnResponse:
    """Utility to create a JSON-RPC server for a given task."""
    return SpawnResponse(url="", session_id="")


class RuntimeContext(TypedBaseModel):
    """Shared infrastructure references created during benchmark.setup()."""

    container_id: str | None = None
    vm_address: str | None = None
    ssh_session: Any | None = None
    # ... whatever shared resources the benchmark provisions


class Benchmark(TypedBaseModel, ABC):
    """Represents a benchmark consisting of multiple tasks."""

    metadata: BenchmarkMetadata
    _task_list: list[TaskConfig] = PrivateAttr(default_factory=list)  # cache loaded task configs
    _runtime_info: RuntimeContext | None = PrivateAttr(
        default=None
    )  # track shared runtime resources created in setup()

    @property
    def name(self) -> str:
        return self.metadata.name

    @abstractmethod
    def setup(self) -> RuntimeContext:
        """
        Setup the benchmark and prepare it for spawning tasks.
        It should create all the necessary shared runtime resources and store them in self._runtime_info.
        This is supposed to be implemented by Benchmark *creators*.
        """
        pass

    @abstractmethod
    def load_tasks(self, cache=True) -> list[TaskConfig]:
        """
        Load and return the list of tasks for this benchmark.
        """
        if len(self._task_list) > 0 and cache:
            return self._task_list
        raise NotImplementedError("load_tasks() must be implemented in subclass.")

    def get_task_configs(self, task_id: str | None = None, offset: int = 0, limit: int = -1) -> list[TaskConfig]:
        """
        Util function to get specific tasks with optional filtering, offset, and limit.
        """
        tasks = self.load_tasks()

        # Apply filtering
        if task_id:
            tasks = [task for task in tasks if task.task_id == task_id]

        # Apply offset and limit
        if limit == -1:
            limited_tasks = tasks[offset:]
        else:
            limited_tasks = tasks[offset : offset + limit]

        return limited_tasks

    def get_runtime_info(self) -> RuntimeContext:
        """
        Get the runtime context created during setup().
        This is needed by TaskConfig.make() to create Task instances.
        """
        if self._runtime_info is None:
            raise RuntimeError("Benchmark not set up yet. Call setup() before accessing runtime info.")
        return self._runtime_info

    def spawn(self, task_id: str, seed: int | None) -> SpawnResponse:
        """
        Spawn a new session for a given task.
        cube/spawn calls this method.

        Server mode: Creates subprocess running task server (returns URL)
        Python mode: Creates TaskSession in-process (returns session object)
        """
        task_config = self.get_task_configs(task_id)[0]
        task = task_config.make(self.get_runtime_info())  # type: ignore
        return make_task_rpc_server(task)

    @abstractmethod
    def close(self) -> None:
        """
        Clean up runtime resources that were created during setup().
        """
        pass
