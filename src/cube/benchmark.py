import logging
from abc import ABC, abstractmethod
from typing import Any, Callable

from pydantic import Field, PrivateAttr

from cube.containers import ContainerBackend
from cube.core import TypedBaseModel
from cube.task import TaskConfig, TaskMetadata
from cube.tool import ToolConfig

logger = logging.getLogger(__name__)


RuntimeContext = dict[str, Any]
"""
Type alias for shared infrastructure references created during benchmark.setup().

example:
    {"container_id": "abc123", "vm_address": "http://12.34.56.78", "ssh_session": session}
"""


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
    # these should be set during setup()
    task_list: list[TaskMetadata] = PrivateAttr(default_factory=list)  # cache loaded task configs
    runtime_context: RuntimeContext = PrivateAttr(
        default_factory=dict
    )  # track shared runtime resources created in setup()
    _default_tool_config: ToolConfig | None = PrivateAttr(
        default=None
    )  # default tool config to be used for tasks that don't specify their own
    _seed_generator: Callable[[], int] | None = PrivateAttr(
        default=None
    )  # optional seed generator for tasks that require random seeds

    @property
    def name(self) -> str:
        return self.metadata.name

    @abstractmethod
    def setup(self) -> None:
        """
        Setup the benchmark and prepare it for spawning tasks.
        This is supposed to be implemented by Benchmark *creators*.
        It **must** (required):
        - define the list of task metadata (self.task_list)
        It should (optional):
        - create any shared infrastructure needed for the tasks and store references in self.runtime_context
        - decide on a default tool config (self._default_tool_config)
        - define a seed generator if needed (self._seed_generator)
        """
        pass

    @abstractmethod
    def _make_task_config(self, task_id, seed: int | None = None, tool_config: ToolConfig | None = None) -> TaskConfig:
        """Concrete subclass creates its specific TaskConfig type"""
        pass

    def create_task_config(
        self, task_id: str, seed: int | None = None, tool_config: ToolConfig | None = None
    ) -> TaskConfig:
        """
        Create a TaskConfig for the specified task_id, using the provided seed and tool_config if given,
        otherwise falling back to defaults defined in the benchmark.

        This is a helper method that calls the abstract _make_task_config() implemented by the concrete subclass.
        """
        # Verify that only task with this ID exists
        metadata = self.get_task_metadata(task_id=task_id)
        assert len(metadata) == 1, f"Expected exactly one task with id {task_id}, but found {len(metadata)}"

        # Use provided params or defaults from setup()
        actual_seed = (
            seed if seed is not None else (self._seed_generator() if self._seed_generator is not None else None)
        )
        actual_tool_config = tool_config or self._default_tool_config

        # Create TaskConfig (calls concrete subclass implementation)
        return self._make_task_config(task_id=task_id, seed=actual_seed, tool_config=actual_tool_config)

    def get_task_metadata(
        self, task_id: list[str] | str | None = None, offset: int = 0, limit: int = -1
    ) -> list[TaskMetadata]:
        """
        Util function to get specific task metadata with optional filtering, offset, and limit.
        """
        if len(self.task_list) == 0:
            raise RuntimeError("Benchmark not set up yet. Call setup() to initialize task metadata.")

        # Apply filtering
        if task_id:
            if isinstance(task_id, str):
                task_id = [task_id]
            tms = [tm for tm in self.task_list if tm.id in task_id]
        else:
            tms = self.task_list

        # Apply offset and limit
        if limit == -1:
            tms = tms[offset:]
        else:
            tms = tms[offset : offset + limit]

        return tms

    def subset_from_glob(self, glob_key: str, glob_pattern: str) -> "Benchmark":
        """
        Create a new Benchmark instance containing only the tasks whose glob_key matches the glob_pattern.
        This is useful for creating smaller benchmarks from a larger one, for example for testing or ablations.
        """
        # TODO
        return self

    def spawn(self, task_id: str, container_backend: ContainerBackend | None = None) -> str:
        """
        Spawn a new RPC server for the specified task on the specified container backend and return its endpoint URL.
        """
        from cube.server import make_task_rpc_server

        tm = self.get_task_metadata(task_id=task_id)[0]
        tc = self.create_task_config(task_id)
        task = tc.make(
            metadata=tm,
            runtime_context=self.runtime_context,
            container_backend=container_backend,
        )  # type: ignore
        _app, _process, url = make_task_rpc_server(task)
        return url

    @abstractmethod
    def close(self) -> None:
        """
        Clean up runtime resources that were created during setup().
        """
        pass
