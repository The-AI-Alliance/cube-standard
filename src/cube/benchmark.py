import logging
from abc import ABC, abstractmethod
from typing import Any

from pydantic import Field, PrivateAttr

from cube.containers import ContainerBackend
from cube.core import TypedBaseModel
from cube.server import make_task_rpc_server
from cube.task import TaskConfig

logger = logging.getLogger(__name__)


class RuntimeContext(TypedBaseModel):
    """Shared infrastructure references created during benchmark.setup()."""

    container_id: str | None = None
    vm_address: str | None = None
    ssh_session: Any | None = None
    # ... whatever shared resources the benchmark provisions


class BenchmarkMetadata(TypedBaseModel):
    """
    Metadata describing a benchmark.

    Used by:
    - Benchmark: metadata attribute
    - API endpoint: cube/info

    Attributes:
        name (str): Benchmark name
        version (str): Benchmark version
        description (str): Benchmark description
        authors (list[str]): List of benchmark author names (default: empty list)
        license (str): Benchmark license (default: empty string)
        requirements (dict[str, Any]): Hardware requirements to install and run the benchmark (default: empty dict)
        num_tasks (int): Total number of tasks (default: 0)
        tags (list[str]): Benchmark tags (default: empty list)
        other (dict[str, Any]): Additional metadata (default: empty dict)
    """

    name: str = Field(..., description="Benchmark name")
    version: str = Field(..., description="Benchmark version")
    description: str = Field(..., description="Benchmark description")
    authors: list[str] = Field(default_factory=list, description="List of benchmark author names")
    license: str = Field(default="", description="Benchmark license")
    requirements: dict[str, Any] = Field(
        default_factory=dict, description="Hardware requirements to install and run the benchmark"
    )
    num_tasks: int = Field(default=0, description="Total number of tasks")
    tags: list[str] = Field(default_factory=list, description="Benchmark tags")
    other: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    # TODO: discuss adding fields such as homepage, repository, citation, etc.


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

    def spawn(self, task_id: str, container_backend: ContainerBackend | None = None) -> str:
        """
        Spawn a new RPC server for the specified task on the specified container backend and return its endpoint URL.
        """
        task_config = self.get_task_configs(task_id)[0]
        task = task_config.make(
            runtime_context=self.get_runtime_info(),
            container_backend=container_backend,
        )  # type: ignore
        return make_task_rpc_server(task)

    @abstractmethod
    def close(self) -> None:
        """
        Clean up runtime resources that were created during setup().
        """
        pass
