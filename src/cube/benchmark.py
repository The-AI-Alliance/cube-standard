import fnmatch
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar

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

    # Class-level attributes that must be defined by subclasses
    task_metadata_dict: ClassVar[dict[str, TaskMetadata]]
    task_config_class: ClassVar[type[TaskConfig]]

    # these optional fields should be set during setup()
    runtime_context: RuntimeContext = Field(default_factory=dict)  # track shared runtime resources created in setup()
    _default_tool_config: ToolConfig | None = PrivateAttr(
        default=None
    )  # default tool config to be used for tasks that don't specify their own
    _seed_generator: Callable[[], int] | None = PrivateAttr(
        default=None
    )  # optional seed generator for tasks that require random seeds

    @property
    def name(self) -> str:
        return self.metadata.name

    def __init_subclass__(cls, **kwargs):
        """Ensure concrete subclasses define required class attributes."""
        super().__init_subclass__(**kwargs)
        # Only enforce for concrete classes (not abstract intermediate classes)
        if not getattr(cls, "__abstractmethods__", None):
            # Check task_metadata_dict is defined as a static class variable (not a property or descriptor)
            if "task_metadata_dict" not in cls.__dict__:
                raise TypeError(
                    f"Concrete benchmark class {cls.__name__} must define 'task_metadata_dict' as a class attribute"
                )
            task_meta = cls.__dict__["task_metadata_dict"]
            if not isinstance(task_meta, dict):
                raise TypeError(
                    f"'task_metadata_dict' in {cls.__name__} must be a dict, not {type(task_meta).__name__}"
                )

            # Check task_config_class is defined as a static class variable (not a property or descriptor)
            if "task_config_class" not in cls.__dict__:
                raise TypeError(
                    f"Concrete benchmark class {cls.__name__} must define 'task_config_class' as a class attribute"
                )
            task_cfg = cls.__dict__["task_config_class"]
            if not isinstance(task_cfg, type) or not issubclass(task_cfg, TaskConfig):
                raise TypeError(
                    f"'task_config_class' in {cls.__name__} must be a subclass of TaskConfig, not {task_cfg}"
                )

    @abstractmethod
    def _setup(self) -> None:
        """
        Setup the benchmark and prepare it for spawning tasks.
        This is supposed to be implemented by Benchmark *creators*.
        It should (optional):
        - create any shared infrastructure needed for the tasks and store references in self.runtime_context
        - decide on a default tool config (self._default_tool_config)
        - define a seed generator if needed (self._seed_generator)
        """
        pass

    def setup(self) -> None:
        """
        Public method to setup the benchmark. Calls the internal _setup() implemented by the concrete subclass.
        """
        self._setup()
        if not self.runtime_context:
            logger.warning(
                "Benchmark setup did not define any runtime context. If your tasks require shared infrastructure, please ensure that self.runtime_context is populated."
            )
        if self._default_tool_config is None:
            logger.warning(
                "Benchmark setup did not define a default tool config. You will need to provide a tool config for all calls to create_task_config()."
            )
        if self._seed_generator is None:
            logger.warning(
                "Benchmark setup did not define a seed generator. If your tasks require a random seed, you will need to provide a seed for all calls to create_task_config()."
            )

    def create_task_config(
        self, task_id: str, tool_config: ToolConfig | None = None, seed: int | None = None
    ) -> TaskConfig:
        """
        Create a TaskConfig for the specified task_id, using the provided seed and tool_config if given,
        otherwise falling back to defaults defined in the benchmark.
        """
        # Use provided params or defaults from setup()
        actual_seed = (
            seed if seed is not None else (self._seed_generator() if self._seed_generator is not None else None)
        )
        actual_tool_config = tool_config or self._default_tool_config
        assert actual_tool_config is not None, (
            "No tool config provided and no default tool config defined in benchmark setup."
        )

        # Directly instantiate the TaskConfig class
        return self.task_config_class(
            task_id=task_id,
            tool_config=actual_tool_config,
            seed=actual_seed,
        )

    def subset_from_glob(self, glob_key: str, glob_pattern: str) -> "Benchmark":
        """
        Create a new Benchmark instance containing only the tasks whose glob_key matches the glob_pattern.
        This is useful for creating smaller benchmarks from a larger one, for example for testing or ablations.
        """
        task_subset = [
            tm
            for tm in self.task_metadata_dict.values()
            if hasattr(tm, glob_key) and fnmatch.fnmatch(getattr(tm, glob_key), glob_pattern)
        ]
        if not task_subset:
            raise ValueError(f"No tasks found matching glob pattern '{glob_pattern}' on key '{glob_key}'")
        return self.subset_from_list(tasks=task_subset, benchmark_name_suffix=f"[{glob_key}={glob_pattern}]")

    def subset_from_list(
        self, tasks: list[str] | list[TaskMetadata], benchmark_name_suffix: str = "custom"
    ) -> "Benchmark":
        """Create a new Benchmark instance containing only the tasks whose IDs are in the provided list.

        Args:
            tasks: List of task IDs or TaskMetadata objects to include in the sub-benchmark.
            benchmark_name_suffix: Optional suffix to append to the benchmark name. Defaults to "custom".

        Returns:
            Benchmark: A new benchmark instance containing only the specified tasks.

        Raises:
            ValueError: If the resulting task list is empty or if any specified task doesn't exist.
        """
        existing_task_ids = {tm.id for tm in self.task_metadata_dict.values()}
        if isinstance(tasks, list) and len(tasks) > 0 and isinstance(tasks[0], str):
            task_ids = set(tasks)
            task_subset = [tm for tm in self.task_metadata_dict.values() if tm.id in task_ids]
            invalid_tasks = task_ids - existing_task_ids
        elif isinstance(tasks, list) and len(tasks) > 0 and isinstance(tasks[0], TaskMetadata):
            task_subset: list[TaskMetadata] = tasks  # type: ignore
            invalid_tasks = {tm.id for tm in tasks} - existing_task_ids  # type: ignore
        else:
            raise ValueError("Tasks must be a non-empty list of either task IDs (str) or TaskMetadata objects.")
        if invalid_tasks:
            raise ValueError(f"The following specified tasks do not exist in the benchmark: {invalid_tasks}")
        if not task_subset:
            raise ValueError("The resulting task list cannot be empty.")

        # TODO: figure put how to copy / update class level static variables!
        new_metadata = BenchmarkMetadata(**self.metadata.model_dump())
        new_metadata.name = f"{self.metadata.name}_{benchmark_name_suffix}"
        new_metadata.num_tasks = len(task_subset)
        new_instance = type(self)(
            metadata=new_metadata,
            runtime_context=self.runtime_context,
        )
        # Copy private attributes
        new_instance._default_tool_config = self._default_tool_config
        new_instance._seed_generator = self._seed_generator
        # Note: task_config_class is defined as a property in the concrete benchmark subclass, so we don't need to copy it here
        return new_instance

    def spawn(self, task_id: str, seed: int | None = None, container_backend: ContainerBackend | None = None) -> str:
        """
        Spawn a new RPC server for the specified task on the specified container backend and return its endpoint URL.
        """
        from cube.server import make_task_rpc_server

        # Find the task metadata
        tm = self.task_metadata_dict.get(task_id)
        if tm is None:
            raise ValueError(f"Task '{task_id}' not found in benchmark")

        tc = self.create_task_config(task_id, seed=seed)
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
