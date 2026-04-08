"""
Benchmark management for CUBE.

This module defines the Benchmark base class and BenchmarkMetadata for managing
collections of tasks, including task spawning, subsetting, and metadata loading
from JSON or CSV files.

Abstract classes:
    Benchmark — subclasses must implement:
        _setup() -> None    create shared infrastructure, populate _runtime_context
        close() -> None     tear down resources created in _setup()
    and define the following class variable:
        task_config_class: type[TaskConfig]     used by get_task_configs()
    The following class variables are also required but can be omitted if the
    corresponding metadata files (benchmark_metadata.json/.csv, task_metadata.json/.csv)
    are placed next to the benchmark module file — they will be loaded automatically:
        benchmark_metadata: BenchmarkMetadata
        task_metadata: dict[str, TaskMetadata]
    For benchmarks whose task list is determined at install time (e.g. downloaded datasets),
    define ``task_metadata: ClassVar[dict[str, TaskMetadata]] = {}`` as a placeholder and
    implement ``install()`` to populate and save ``task_metadata.json``. The empty dict
    signals to ``__init_subclass__`` that install-time population is expected.
"""

import copy
import csv
import enum
import fnmatch
import json
import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Generator

from pydantic import ConfigDict, Field, PrivateAttr

from cube import get_cache_dir
from cube.container import ContainerBackend
from cube.core import TypedBaseModel
from cube.seed import AbstractSeedGenerator
from cube.task import TaskConfig, TaskMetadata
from cube.tool import ToolConfig

logger = logging.getLogger(__name__)


class ResetIsolation(str, enum.Enum):
    """Describes the isolation guarantee provided by a benchmark's reset mechanism.

    Declared on BenchmarkMetadata so harness users can reason about safe parallelism
    and reproducibility before running tasks.

    Values:
        SNAPSHOT:     VM reverted to a saved savestate (~5s). Strongest isolation.
        RESTART:      Container/VM stopped and restarted (~30s). No state leakage.
        APP_LEVEL:    Application state reset via scripts; VM stays running (~5s).
                      Risk of OS-level state leakage between tasks on the same VM.
        NEW_INSTANCE: Fresh VM per task (~2-4 min). Strongest guarantee, slowest.
    """

    SNAPSHOT = "snapshot"
    RESTART = "restart"
    APP_LEVEL = "app_level"
    NEW_INSTANCE = "new_instance"


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
    - Benchmark: benchmark_metadata attribute
    - API endpoint: cube/info

    Attributes:
        name (str): Benchmark name
        version (str): Benchmark version
        description (str): Benchmark description
        authors (list[str]): List of benchmark author names (default: empty list)
        license (str): Benchmark license (default: empty string)
        requirements (dict[str, Any]): Environment requirements (hardware, OS, VMs, containers, etc.) to install and run the benchmark (default: empty dict)
        num_tasks (int): Total number of tasks (default: 0)
        tags (list[str]): Benchmark tags (default: empty list)
        extra_info (dict[str, Any]): Additional metadata (default: empty dict)
    """

    name: str = Field(..., description="Benchmark name")
    version: str = Field(..., description="Benchmark version")
    description: str = Field(..., description="Benchmark description")
    authors: list[str] = Field(default_factory=list, description="List of benchmark author names")
    license: str = Field(default="", description="Benchmark license")
    requirements: dict[str, Any] = Field(
        default_factory=dict,
        description="Environment requirements (hardware, OS, VMs, containers, etc.) to install and run the benchmark",
    )
    num_tasks: int = Field(default=0, description="Total number of tasks")
    tags: list[str] = Field(default_factory=list, description="Benchmark tags")
    reset_isolation: ResetIsolation | None = Field(
        default=None,
        description=(
            "Isolation guarantee provided by this benchmark's reset mechanism. "
            "None means unspecified. Set by benchmark authors to let harness users "
            "reason about safe parallelism (e.g. APP_LEVEL + multiple workers on the "
            "same VM is unsafe). See ResetIsolation for possible values."
        ),
    )
    named_subsets: dict[str, tuple[str, str]] = Field(
        default_factory=dict,
        description=(
            "Named subsets of this benchmark, as a mapping from subset name to "
            "(glob_key, glob_pattern) passed to Benchmark.subset_from_glob(). "
            "Lets users obtain a pre-defined filtered view without writing glob expressions manually. "
            "Example: {'lite': ('extra_info', '*\"lite\"*'), 'verified': ('extra_info', '*\"verified\"*')}"
        ),
    )
    extra_info: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    # TODO: discuss adding / removing fields such as homepage, repository, citation, etc.


class Benchmark(TypedBaseModel, ABC):
    """Represents a benchmark consisting of multiple tasks."""

    # Class-level attributes that must be defined by subclasses (not constructor params)
    benchmark_metadata: ClassVar[BenchmarkMetadata]
    task_metadata: ClassVar[dict[str, TaskMetadata]]
    task_config_class: ClassVar[type[TaskConfig]]

    # this optional fields should be set during _setup() by the Benchmark **creator** (not constructor params).
    _runtime_context: RuntimeContext = PrivateAttr(
        default_factory=dict
    )  # track shared runtime resources created in setup()

    # these optional fields should be set by benchmark *users* (constructor params).
    container_backend: ContainerBackend | None = Field(
        default=None
    )  # optional container backend to be used for all tasks in this benchmark
    default_tool_config: ToolConfig | None = Field(
        default=None
    )  # default tool config to be used for tasks that don't specify their own
    seed_generator: AbstractSeedGenerator | None = Field(
        default=None
    )  # optional seed generator for tasks that require random seeds

    model_config = ConfigDict(arbitrary_types_allowed=True)  # allow non-serializable fields like AbstractSeedGenerator

    # ------------------------------------------------------------------
    # File-loading helpers
    #
    # Automatically called in __init_subclass__ if the relevant class variables aren't already defined,
    # looking for files next to the benchmark module file.
    # You can also call these directly at class definition time in your subclass.
    #
    #   class MyBenchmark(Benchmark):
    #       benchmark_metadata = Benchmark.benchmark_metadata_from_json("benchmark.json")
    #       task_metadata = Benchmark.task_metadata_from_csv("tasks.csv")
    #       task_config_class  = MyTaskConfig
    # ------------------------------------------------------------------

    @staticmethod
    def benchmark_metadata_from_json(path: str | Path) -> "BenchmarkMetadata":
        """Load :class:`BenchmarkMetadata` from a JSON file.

        The JSON object keys must match the ``BenchmarkMetadata`` field names.
        """
        with open(path) as f:
            data = json.load(f)
        return BenchmarkMetadata(**data)

    @staticmethod
    def benchmark_metadata_from_csv(path: str | Path) -> "BenchmarkMetadata":
        """Load :class:`BenchmarkMetadata` from a CSV file.

        The file must have a header row followed by exactly one data row.
        Complex fields must be stored as JSON strings:

        * ``authors``      — JSON array,  e.g. ``["Alice","Bob"]``
        * ``tags``         — JSON array,  e.g. ``["toy","counter"]``
        * ``requirements`` — JSON object, e.g. ``{"gpu": false}``
        * ``extra_info``   — JSON object, e.g. ``{}``

        ``num_tasks`` is parsed as an integer (omit the column to use the
        default of ``0``).

        Example CSV::

            name,version,description,num_tasks,tags
            toy-counter,1.0.0,My benchmark,3,"[""toy""]"
        """
        _JSON_FIELDS = ("authors", "tags", "requirements", "extra_info")
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            row = next(reader, None)
            extra = next(reader, None)
        if row is None:
            raise ValueError(f"benchmark_metadata CSV '{path}' has no data rows")
        if extra is not None:
            raise ValueError(f"benchmark_metadata CSV '{path}' must have exactly one data row, found more")
        data: dict[str, Any] = {k: v for k, v in row.items() if v != ""}
        for field in _JSON_FIELDS:
            if field in data:
                data[field] = json.loads(data[field])
        if "num_tasks" in data:
            data["num_tasks"] = int(data["num_tasks"])
        return BenchmarkMetadata(**data)

    @staticmethod
    def task_metadata_from_json(path: str | Path) -> "dict[str, TaskMetadata]":
        """Load ``task_metadata`` from a JSON file.

        The file may contain either:

        * a **list** of task-metadata objects, or
        * a **dict** mapping task IDs to task-metadata objects.

        In both cases the returned dict is keyed by ``TaskMetadata.id``.
        Complex fields (``extra_info``, ``tags``, ``container_config``) are
        parsed natively from JSON.
        """
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = list(data.values())
        else:
            raise ValueError(f"task_metadata JSON must be a list or dict, got {type(data).__name__}")
        tasks = [TaskMetadata(**item) for item in items]
        return {t.id: t for t in tasks}

    @staticmethod
    def task_metadata_from_csv(path: str | Path) -> "dict[str, TaskMetadata]":
        """Load ``task_metadata`` from a CSV file.

        Each row represents one task. The ``id`` column is required; all other
        columns are optional and fall back to their :class:`TaskMetadata`
        defaults when absent or empty.

        Complex fields must be stored as JSON strings in the CSV:

        * ``extra_info``       — JSON object, e.g. ``{"target": 3}``
        * ``tags``             — JSON array,  e.g. ``["easy","counter"]``
        * ``container_config`` — JSON object or ``null``

        ``recommended_max_steps`` is parsed as an integer (or ``None`` if
        the cell is empty).
        """
        _JSON_FIELDS = ("extra_info", "tags", "container_config")
        tasks = []
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                data: dict[str, Any] = {k: v for k, v in row.items() if v != ""}
                for field in _JSON_FIELDS:
                    if field in data:
                        data[field] = json.loads(data[field])
                if "recommended_max_steps" in data:
                    data["recommended_max_steps"] = int(data["recommended_max_steps"])
                tasks.append(TaskMetadata(**data))
        return {t.id: t for t in tasks}

    @property
    def name(self) -> str:
        return self.benchmark_metadata.name

    def __init_subclass__(cls, **kwargs):
        """Hook called automatically by Python whenever a new subclass of Benchmark is defined.

        It validates that every **concrete** subclass (i.e. one that has no remaining abstract
        methods) provides the three required class-level attributes:

        * ``benchmark_metadata`` — a :class:`BenchmarkMetadata` instance describing the benchmark.
        * ``task_metadata`` — a ``dict[str, TaskMetadata]`` mapping task IDs to their metadata.
        * ``task_config_class`` — a subclass of :class:`TaskConfig` used to instantiate tasks.

        For ``benchmark_metadata`` and ``task_metadata``, if the attribute is **not** defined
        directly on the subclass, this hook automatically tries to load it from a file placed
        next to the benchmark module (``benchmark_metadata.json/.csv`` or
        ``task_metadata.json/.csv``).  A ``TypeError`` is raised if neither the attribute nor
        a matching file can be found.

        Abstract intermediate classes (those that still declare ``@abstractmethod`` members) are
        skipped — validation only fires once all abstract methods have been implemented.

        Example::

            # Option 1 — auto-load from files next to the benchmark module
            # (benchmark_metadata.json and task_metadata.csv are found automatically)
            class MyBenchmark(Benchmark):
                task_config_class = MyTaskConfig

                def _setup(self): ...
                def close(self): ...

            # Option 2 — supply metadata explicitly as instances
            class MyBenchmark(Benchmark):
                benchmark_metadata = BenchmarkMetadata(name="my-bench", version="1.0", description="...")
                task_metadata      = {"task-1": TaskMetadata(id="task-1"), ...}
                task_config_class  = MyTaskConfig

                def _setup(self): ...
                def close(self): ...

            # Option 3 — load from custom paths using the provided loader helpers
            class MyBenchmark(Benchmark):
                benchmark_metadata = Benchmark.benchmark_metadata_from_json("/data/bench/meta.json")
                task_metadata      = Benchmark.task_metadata_from_csv("/data/bench/tasks.csv")
                task_config_class  = MyTaskConfig

                def _setup(self): ...
                def close(self): ...
        """
        super().__init_subclass__(**kwargs)
        # Only enforce for concrete classes (not abstract intermediate classes)
        if not getattr(cls, "__abstractmethods__", None):
            # `cls.__module__` is the dotted name of the module where the subclass is defined
            # (e.g. "my_cube.my_benchmark"). We look it up in sys.modules to get the actual
            # module object, then read its __file__ so we know which directory to search for
            # metadata files when the designer hasn't provided them explicitly.
            module_file = getattr(sys.modules.get(cls.__module__), "__file__", None)
            module_dir = Path(module_file).resolve().parent if module_file else None

            # Auto-load benchmark_metadata from a file next to your benchmark file.
            # Triggers when benchmark_metadata is missing or an empty {} placeholder.
            if not cls.__dict__.get("benchmark_metadata"):
                loaded = None
                if module_dir:
                    for fname, loader in [
                        ("benchmark_metadata.json", Benchmark.benchmark_metadata_from_json),
                        ("benchmark_metadata.csv", Benchmark.benchmark_metadata_from_csv),
                    ]:
                        candidate = module_dir / fname
                        if candidate.exists():
                            loaded = loader(candidate)
                            break
                if loaded is None:
                    raise TypeError(
                        f"Concrete benchmark class {cls.__name__} must define 'benchmark_metadata' as a class attribute, "
                        f"or provide a 'benchmark_metadata.json' / 'benchmark_metadata.csv' file next to your benchmark file"
                    )
                cls.benchmark_metadata = loaded

            bench_meta = cls.__dict__["benchmark_metadata"]
            if not isinstance(bench_meta, BenchmarkMetadata):
                raise TypeError(
                    f"'benchmark_metadata' in {cls.__name__} must be a BenchmarkMetadata instance, not {type(bench_meta).__name__}"
                )

            # Auto-load task_metadata from a file next to your benchmark file.
            # Triggers when task_metadata is missing or an empty {} placeholder
            if not cls.__dict__.get("task_metadata"):
                loaded = None
                if module_dir:
                    for fname, loader in [
                        ("task_metadata.json", Benchmark.task_metadata_from_json),
                        ("task_metadata.csv", Benchmark.task_metadata_from_csv),
                    ]:
                        candidate = module_dir / fname
                        if candidate.exists():
                            loaded = loader(candidate)
                            break
                if loaded is not None:
                    cls.task_metadata = loaded
                else:
                    # if no task_metadata file is found, assume install-time population via install().
                    cls.task_metadata = {}
                    logger.warning(
                        f"{cls.__name__}.task_metadata is empty — no task_metadata.json/.csv found "
                        f"next to the benchmark module. Call `{cls.__name__}.install()` to download "
                        f"and cache task metadata before using this benchmark."
                    )

            task_meta = cls.__dict__["task_metadata"]
            if not isinstance(task_meta, dict):
                raise TypeError(f"'task_metadata' in {cls.__name__} must be a dict, not {type(task_meta).__name__}")

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
        It should (optionaly) create any shared infrastructure needed for the tasks and store references in self._runtime_context (default to {})
        """
        pass

    def setup(self) -> None:
        """
        Public method to setup the benchmark. Calls the internal _setup() implemented by the concrete subclass.
        """
        if not self.task_metadata:
            raise RuntimeError(
                f"{type(self).__name__}.task_metadata is empty. "
                f"Run `{type(self).__name__}.install()` first to download and cache task metadata."
            )
        self._setup()
        # One debug line instead of four warnings — optional fields are unset for many minimal benchmarks.
        missing_optional: list[str] = []
        if not self._runtime_context:
            missing_optional.append("_runtime_context")
        if self.container_backend is None:
            missing_optional.append("container_backend")
        if self.default_tool_config is None:
            missing_optional.append("default_tool_config")
        if self.seed_generator is None:
            missing_optional.append("seed_generator")
        if missing_optional:
            logger.debug(
                "%s: optional benchmark fields not set (%s). Normal for simple examples; "
                "define them when tasks need shared infra, containers, or seeds.",
                type(self).__name__,
                ", ".join(missing_optional),
            )

    def get_task_configs(self) -> Generator[TaskConfig, None, None]:
        """Returns the list of TaskConfig objects for this benchmark."""
        for tm in self.task_metadata.values():
            if self.seed_generator is not None:
                for seed in self.seed_generator(tm):
                    yield self.task_config_class(task_id=tm.id, tool_config=self.default_tool_config, seed=seed)
            else:
                yield self.task_config_class(task_id=tm.id, tool_config=self.default_tool_config, seed=None)

    def subset_from_glob(self, glob_key: str, glob_pattern: str) -> "Benchmark":
        """
        Create a new Benchmark instance containing only the tasks whose ``glob_key`` matches ``glob_pattern``.

        This is useful for creating smaller benchmarks from a larger one, for example for testing
        or ablations.

        ``glob_key`` can be any of the top-level :class:`~cube.task.TaskMetadata` fields:

        * ``id``                    — task identifier
        * ``split``                 — ``"train"``, ``"val"``, or ``"test"``
        * ``abstract_description``  — broad description of the task
        * ``recommended_max_steps`` — integer step budget (matched as a string)

        It can also address a key inside the ``extra_info`` dict using dot-notation:

        * ``extra_info.<key>``  e.g. ``extra_info.difficulty`` or ``extra_info.domain``

        ``glob_pattern`` follows standard Unix shell-style wildcards (``*``, ``?``, ``[seq]``).
        Non-string field values are converted to strings before matching.
        """
        if glob_key.startswith("extra_info."):
            extra_key = glob_key[len("extra_info.") :]
            task_subset = [
                tm
                for tm in self.task_metadata.values()
                if extra_key in tm.extra_info and fnmatch.fnmatch(str(tm.extra_info[extra_key]), glob_pattern)
            ]
        else:
            task_subset = [
                tm
                for tm in self.task_metadata.values()
                if hasattr(tm, glob_key) and fnmatch.fnmatch(str(getattr(tm, glob_key)), glob_pattern)
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
        existing_task_ids = {tm.id for tm in self.task_metadata.values()}
        if isinstance(tasks, list) and len(tasks) > 0 and isinstance(tasks[0], str):
            task_ids = set(tasks)
            task_subset = [tm for tm in self.task_metadata.values() if tm.id in task_ids]
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

        # Deep-copy self to preserve the exact concrete class and all instance state (including
        # subclass __init__ args, private attrs, etc.), then override the relevant attributes.
        # ClassVars are not copied by deepcopy (they live on the class), so we deepcopy them
        # explicitly and use object.__setattr__ to set instance-level shadows without triggering
        # Pydantic's ClassVar protection.
        new_instance = copy.deepcopy(self)
        new_bm = copy.deepcopy(type(new_instance).benchmark_metadata)
        new_bm.name = f"{self.benchmark_metadata.name}_{benchmark_name_suffix}"
        new_bm.num_tasks = len(task_subset)
        object.__setattr__(new_instance, "benchmark_metadata", new_bm)
        object.__setattr__(new_instance, "task_metadata", {tm.id: tm for tm in task_subset})
        return new_instance

    @classmethod
    def named_subsets(cls) -> list[str]:
        """Return the names of all pre-defined subsets for this benchmark.

        Callable without instantiation: ``MyBenchmark.named_subsets()``.
        Subsets are defined in ``benchmark_metadata.named_subsets``.
        """
        return list(cls.benchmark_metadata.named_subsets.keys())

    def named_subset(self, name: str) -> "Benchmark":
        """Return a filtered benchmark containing only the tasks in the named subset.

        Equivalent to ``subset_from_glob(*benchmark_metadata.named_subsets[name])``.

        Args:
            name: A key from ``benchmark_metadata.named_subsets``.

        Raises:
            KeyError: If ``name`` is not a known subset.
        """
        if name not in self.benchmark_metadata.named_subsets:
            available = list(self.benchmark_metadata.named_subsets.keys())
            raise KeyError(f"Unknown subset {name!r}. Available: {available}")
        glob_key, glob_pattern = self.benchmark_metadata.named_subsets[name]
        return self.subset_from_glob(glob_key, glob_pattern)

    def spawn(self, task_config: TaskConfig) -> str:
        """
        Spawn a new RPC server for the specified task and return its endpoint URL.

        Args:
            task_config: A TaskConfig produced by get_task_configs().
        """
        from cube.server import make_task_rpc_server

        if task_config.task_id not in self.task_metadata:
            raise ValueError(f"Task '{task_config.task_id}' not found in benchmark")

        task = task_config.make(
            runtime_context=self._runtime_context,
            container_backend=self.container_backend,
        )
        _app, _process, url = make_task_rpc_server(task)
        return url

    @abstractmethod
    def close(self) -> None:
        """
        Clean up runtime resources that were created during setup().
        """
        pass

    @classmethod
    def install(cls) -> None:
        """
        Optional classmethod to install dependencies and cache task metadata for this benchmark.

        Override this in your benchmark to:
          1. Download any required datasets or assets.
          2. Build the full task_metadata dict (all tasks, all splits — no filtering).
          3. Save it as ``task_metadata.json`` next to the benchmark module file.
          4. Update ``cls.task_metadata`` in memory so it takes effect immediately.

        By default does nothing. Can be called without instantiating the benchmark:
        ``MyBenchmark.install()``
        """
        pass

    @classmethod
    def uninstall(cls) -> None:
        """
        Optional classmethod to remove any assets downloaded by install().
        By default does nothing. Can be called without instantiating the benchmark:
        ``MyBenchmark.uninstall()``
        """
        pass

    @classmethod
    def task_execution_cache_dir(cls) -> Path:
        """Return the directory where per-task execution data is cached.

        Per-task execution files are written here by ``install()`` and read
        lazily by ``TaskConfig.make()`` at execution time.  The directory is
        benchmark-specific: ``~/.cube/<benchmark_name>/tasks_execution_info/``.
        """
        return get_cache_dir(cls.benchmark_metadata.name) / "tasks_execution_info"

    @classmethod
    def load_task_execution_info(cls, task_id: str) -> dict[str, Any]:
        """Load per-task execution data for *task_id* from the local cache.

        Raises ``RuntimeError`` if the cache file is missing (i.e. ``install()``
        has not been run yet).  Cube ``TaskConfig.make()`` implementations call
        this to obtain heavy fields (e.g. ``problem_statement``, ``patch``) that
        are excluded from the shipped ``task_metadata.json``.
        """
        cache_file = cls.task_execution_cache_dir() / f"{task_id}.json"
        if not cache_file.exists():
            raise RuntimeError(
                f"No execution data for {task_id!r}. Run `{cls.__name__}.install()` to populate the execution cache."
            )
        return json.loads(cache_file.read_text())
