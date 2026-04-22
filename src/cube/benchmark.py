"""
Benchmark layer for CUBE.

`BenchmarkConfig` is the serializable description of a benchmark: what tasks
exist, what resources they need, and what task-level defaults to apply. It is
pure Pydantic data — safe to serialize, copy, and ship across process
boundaries.

`Benchmark` is the runtime pair: a plain Python class holding live OS state
(container handles, server URLs, open connections). It is produced by
``BenchmarkConfig.make(infra)`` and is never serialized.

The split mirrors ``TaskConfig`` / ``Task``. Subclasses of ``BenchmarkConfig``
declare class-level registries (``benchmark_metadata``, ``task_metadata``,
``task_config_class``, ``benchmark_class``); subclasses of ``Benchmark``
implement ``_setup`` / ``close``.

Typical shape::

    class MyBenchmarkConfig(BenchmarkConfig):
        benchmark_metadata: ClassVar = BenchmarkMetadata(...)
        task_metadata: ClassVar = {...}               # or auto-loaded from file
        task_config_class: ClassVar = MyTaskConfig
        benchmark_class: ClassVar = "MyBenchmark"    # forward ref resolved below

    class MyBenchmark(Benchmark):
        def _setup(self) -> None: ...
        def close(self) -> None: ...

    MyBenchmarkConfig.benchmark_class = MyBenchmark

Subsetting
----------
``subset_from_list`` / ``subset_from_glob`` / ``named_subset`` return a new
``BenchmarkConfig`` with its ``task_ids`` field narrowed. The class-level
``task_metadata`` dict remains the authoritative registry — subsets only select
which ids are emitted by ``get_task_configs``. ``TaskConfig.make`` on a worker
still looks up ``OwnerBenchmarkConfig.task_metadata[self.task_id]`` and works
identically under any subset.

Composition
-----------
See ``cube.benchmark`` later additions for ``CompositeBenchmarkConfig``.
"""

from __future__ import annotations

import csv
import enum
import fnmatch
import json
import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Generator

from pydantic import ConfigDict, Field

from cube import get_cache_dir
from cube.container import ContainerBackend
from cube.core import TypedBaseModel
from cube.resource import InfraConfig, ResourceConfig
from cube.seed import AbstractSeedGenerator
from cube.task import RuntimeContext, Task, TaskConfig, TaskMetadata
from cube.tool import ToolConfig

logger = logging.getLogger(__name__)


class ResetIsolation(str, enum.Enum):
    """Isolation guarantee provided by a benchmark's reset mechanism.

    Declared on ``BenchmarkMetadata`` so harness users can reason about safe
    parallelism and reproducibility before running tasks.

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


class BenchmarkMetadata(TypedBaseModel):
    """Static description of a benchmark — declared once per subclass.

    Used by:
    - ``BenchmarkConfig.benchmark_metadata`` (ClassVar)
    - API endpoint: ``cube/info``
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
    num_tasks: int = Field(default=0, description="Total number of tasks in the *full* benchmark (pre-subset)")
    tags: list[str] = Field(default_factory=list, description="Benchmark tags")
    reset_isolation: ResetIsolation | None = Field(
        default=None,
        description=(
            "Isolation guarantee provided by this benchmark's reset mechanism. "
            "None means unspecified. Set by benchmark authors to let harness users "
            "reason about safe parallelism (e.g. APP_LEVEL + multiple workers on the "
            "same VM is unsafe)."
        ),
    )
    named_subsets: dict[str, tuple[str, str]] = Field(
        default_factory=dict,
        description=(
            "Named subsets of this benchmark, as a mapping from subset name to "
            "(glob_key, glob_pattern) passed to BenchmarkConfig.subset_from_glob(). "
            "Example: {'lite': ('extra_info', '*\"lite\"*')}"
        ),
    )
    extra_info: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class BenchmarkConfig(TypedBaseModel, ABC):
    """Serializable description of a benchmark. Safe to copy, serialize, and ship.

    Subclasses declare four class-level attributes:

    * ``benchmark_metadata: ClassVar[BenchmarkMetadata]`` — static description.
    * ``task_metadata: ClassVar[dict[str, TaskMetadata]]`` — task registry
      (loaded at class definition time from ``task_metadata.json`` / ``.csv``
      next to the module, or declared directly).
    * ``task_config_class: ClassVar[type[TaskConfig]]`` — config emitted by
      ``get_task_configs``.
    * ``benchmark_class: ClassVar[type[Benchmark]]`` — the runtime pair
      instantiated by ``make(infra)``.

    Instance fields carry per-configuration state: the current subset
    (``task_ids``), declared resources, default tool config, seed generator,
    and optional container backend.

    ``make(infra)`` is the single factory that turns a config into a live
    ``Benchmark``: it provisions any declared resources idempotently, then
    constructs and sets up the runtime pair. Users never call ``setup()``
    directly — a ``Benchmark`` returned from ``make`` is born ready.
    """

    # ── Class-level registries (populated by subclasses or __init_subclass__) ──
    benchmark_metadata: ClassVar[BenchmarkMetadata]
    task_metadata: ClassVar[dict[str, TaskMetadata]]
    task_config_class: ClassVar[type[TaskConfig]]
    benchmark_class: ClassVar[type["Benchmark"]]

    # Opt-out marker: set to True on subclasses that populate the ClassVars
    # dynamically (e.g. CompositeBenchmarkConfig) to skip file auto-load and
    # registry validation in ``__init_subclass__``.
    _skip_init_subclass_checks: ClassVar[bool] = False

    # ── Instance fields (user-configurable; safely serializable) ──────────────
    task_ids: list[str] | None = Field(
        default=None,
        description=(
            "Subset selector: if None, ``tasks()`` returns every entry in the class-level "
            "task_metadata. Populated by ``subset_from_list`` / ``subset_from_glob`` / "
            "``named_subset`` to narrow the set without touching the ClassVar."
        ),
    )
    resources: list[ResourceConfig] = Field(
        default_factory=list,
        description=(
            "Declared resource dependencies. ``make(infra)`` calls ``infra.provision(r)`` "
            "for each entry whose ``provision_status`` is not ``ready`` before setup runs."
        ),
    )
    container_backend: ContainerBackend | None = Field(
        default=None,
        description="Optional container backend passed through to every spawned task.",
    )
    default_tool_config: ToolConfig | None = Field(
        default=None,
        description="Default tool configuration for tasks that do not supply their own.",
    )
    seed_generator: AbstractSeedGenerator | None = Field(
        default=None,
        description="Optional seed generator yielding per-task seeds during get_task_configs().",
    )

    # ``AbstractSeedGenerator`` is a Pydantic ``BaseModel`` (not TypedBaseModel);
    # ``ContainerBackend`` is TypedBaseModel but its live handles aren't always
    # JSON-roundtrippable. ``arbitrary_types_allowed`` matches the pre-split shape
    # and is kept for that reason.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ──────────────────────────────────────────────────────────────────────────
    # File-loading helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def benchmark_metadata_from_json(path: str | Path) -> "BenchmarkMetadata":
        """Load ``BenchmarkMetadata`` from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return BenchmarkMetadata.model_validate(data)

    @staticmethod
    def benchmark_metadata_from_csv(path: str | Path) -> "BenchmarkMetadata":
        """Load ``BenchmarkMetadata`` from a single-row CSV.

        Complex fields (``authors``, ``tags``, ``requirements``, ``extra_info``)
        must be stored as JSON-encoded strings.
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
        return BenchmarkMetadata.model_validate(data)

    @staticmethod
    def task_metadata_from_json(path: str | Path) -> dict[str, TaskMetadata]:
        """Load ``task_metadata`` from a JSON file.

        The file may contain either a list of task-metadata objects or a dict
        keyed by task id. In both cases the returned dict is keyed by
        ``TaskMetadata.id``.
        """
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = list(data.values())
        else:
            raise ValueError(f"task_metadata JSON must be a list or dict, got {type(data).__name__}")
        tasks = [TaskMetadata.model_validate(item) for item in items]
        return {t.id: t for t in tasks}

    @staticmethod
    def task_metadata_from_csv(path: str | Path) -> dict[str, TaskMetadata]:
        """Load ``task_metadata`` from a CSV file.

        Each row is one task. ``id`` is required. Complex fields (``extra_info``,
        ``tags``, ``container_config``) must be JSON-encoded strings.
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
                tasks.append(TaskMetadata.model_validate(data))
        return {t.id: t for t in tasks}

    # ──────────────────────────────────────────────────────────────────────────
    # Subclass validation / file auto-load
    # ──────────────────────────────────────────────────────────────────────────

    def __init_subclass__(cls, **kwargs):
        """Validate that every concrete subclass wires the four class-level registries.

        For ``benchmark_metadata`` and ``task_metadata``, attempt auto-load from
        files next to the module if the attribute is not declared on the class
        (including via inheritance). Subclasses that populate the registries
        dynamically (e.g. ``CompositeBenchmarkConfig``) can set
        ``_skip_init_subclass_checks = True`` to opt out entirely.
        """
        super().__init_subclass__(**kwargs)

        # Abstract intermediates still have unimplemented abstract methods — skip.
        if getattr(cls, "__abstractmethods__", None):
            return

        # Dynamic-registry classes opt out via a class-level flag.
        if cls.__dict__.get("_skip_init_subclass_checks"):
            return

        _SENTINEL = object()
        module_file = getattr(sys.modules.get(cls.__module__), "__file__", None)
        module_dir = Path(module_file).resolve().parent if module_file else None

        # ── benchmark_metadata ───────────────────────────────────────────────
        # Check own dict first (explicit declaration); if missing, try file
        # auto-load; otherwise fall back to inherited value from a parent.
        if "benchmark_metadata" not in cls.__dict__:
            loaded = None
            if module_dir:
                for fname, loader in [
                    ("benchmark_metadata.json", BenchmarkConfig.benchmark_metadata_from_json),
                    ("benchmark_metadata.csv", BenchmarkConfig.benchmark_metadata_from_csv),
                ]:
                    candidate = module_dir / fname
                    if candidate.exists():
                        loaded = loader(candidate)
                        break
            if loaded is not None:
                cls.benchmark_metadata = loaded
            elif getattr(cls, "benchmark_metadata", _SENTINEL) is _SENTINEL:
                raise TypeError(
                    f"Concrete benchmark config class {cls.__name__} must define "
                    f"'benchmark_metadata' as a ClassVar or ship a "
                    f"'benchmark_metadata.json' / 'benchmark_metadata.csv' file next to the module."
                )

        bench_meta = getattr(cls, "benchmark_metadata", None)
        if not isinstance(bench_meta, BenchmarkMetadata):
            raise TypeError(
                f"'benchmark_metadata' in {cls.__name__} must be a BenchmarkMetadata instance, "
                f"not {type(bench_meta).__name__}"
            )

        # ── task_metadata ────────────────────────────────────────────────────
        if "task_metadata" not in cls.__dict__:
            loaded = None
            if module_dir:
                for fname, loader in [
                    ("task_metadata.json", BenchmarkConfig.task_metadata_from_json),
                    ("task_metadata.csv", BenchmarkConfig.task_metadata_from_csv),
                ]:
                    candidate = module_dir / fname
                    if candidate.exists():
                        loaded = loader(candidate)
                        break
            if loaded is not None:
                cls.task_metadata = loaded
            elif getattr(cls, "task_metadata", _SENTINEL) is _SENTINEL:
                raise TypeError(
                    f"{cls.__name__} must declare 'task_metadata' as a ClassVar or ship a "
                    f"task_metadata.json / task_metadata.csv file next to the module."
                )

        task_meta = getattr(cls, "task_metadata", None)
        if not isinstance(task_meta, dict):
            raise TypeError(f"'task_metadata' in {cls.__name__} must be a dict, not {type(task_meta).__name__}")

        # ── task_config_class ───────────────────────────────────────────────
        task_cfg = getattr(cls, "task_config_class", _SENTINEL)
        if task_cfg is _SENTINEL:
            raise TypeError(
                f"Concrete benchmark config class {cls.__name__} must define 'task_config_class' as a ClassVar"
            )
        if not isinstance(task_cfg, type) or not issubclass(task_cfg, TaskConfig):
            raise TypeError(f"'task_config_class' in {cls.__name__} must be a subclass of TaskConfig, not {task_cfg!r}")

        # ── benchmark_class ─────────────────────────────────────────────────
        bench_cls = getattr(cls, "benchmark_class", _SENTINEL)
        if bench_cls is _SENTINEL:
            raise TypeError(
                f"Concrete benchmark config class {cls.__name__} must define 'benchmark_class' "
                f"(the runtime Benchmark subclass produced by ``make(infra)``)"
            )
        if not isinstance(bench_cls, type) or not issubclass(bench_cls, Benchmark):
            raise TypeError(f"'benchmark_class' in {cls.__name__} must be a subclass of Benchmark, not {bench_cls!r}")

    # ──────────────────────────────────────────────────────────────────────────
    # Views
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        """Benchmark name from the class-level ``benchmark_metadata``."""
        return type(self).benchmark_metadata.name

    @property
    def num_tasks(self) -> int:
        """Number of tasks in the current (possibly subset) view."""
        return len(self.tasks())

    def tasks(self) -> dict[str, TaskMetadata]:
        """Return the current task view — class-level ``task_metadata`` filtered by ``task_ids``.

        When ``task_ids`` is None, returns the full ClassVar dict. Otherwise
        returns a freshly-built dict containing only the declared ids, in the
        order given by ``task_ids``.
        """
        full = type(self).task_metadata
        if self.task_ids is None:
            return full
        return {tid: full[tid] for tid in self.task_ids}

    def get_task_configs(self) -> Generator[TaskConfig, None, None]:
        """Yield one ``TaskConfig`` per task (expanded by seed_generator if set)."""
        for tm in self.tasks().values():
            if self.seed_generator is not None:
                for seed in self.seed_generator(tm):
                    yield self.task_config_class(
                        task_id=tm.id,
                        tool_config=self.default_tool_config,
                        seed=seed,
                    )
            else:
                yield self.task_config_class(
                    task_id=tm.id,
                    tool_config=self.default_tool_config,
                    seed=None,
                )

    # ──────────────────────────────────────────────────────────────────────────
    # Subsetting (pure data — no deep-copy or private-attr hacks)
    # ──────────────────────────────────────────────────────────────────────────

    def subset_from_list(
        self,
        tasks: list[str] | list[TaskMetadata],
        benchmark_name_suffix: str = "custom",  # noqa: ARG002 — accepted for call-site compat
    ) -> "BenchmarkConfig":
        """Return a new ``BenchmarkConfig`` restricted to the given tasks.

        Accepts either a list of task ids (strings) or a list of
        ``TaskMetadata`` objects. The returned config is the same subclass,
        with ``task_ids`` set and every other field inherited via
        ``model_copy``. No private-attr reset or ClassVar shadowing is
        required — ``TaskConfig.make()`` still resolves metadata via the
        class-level ``task_metadata`` ClassVar.

        ``benchmark_name_suffix`` is retained for call-site compatibility but
        has no effect in the new design; subsets inherit their name from the
        parent class's ``benchmark_metadata``. Use a display-layer convention
        if you need a distinct label.
        """
        current = self.tasks()
        existing_ids = set(current.keys())

        if isinstance(tasks, list) and tasks and isinstance(tasks[0], str):
            task_ids: list[str] = list(tasks)  # preserve caller order; duplicates pruned below
            invalid = set(task_ids) - existing_ids
        elif isinstance(tasks, list) and tasks and isinstance(tasks[0], TaskMetadata):
            task_ids = [tm.id for tm in tasks]  # type: ignore[union-attr]
            invalid = set(task_ids) - existing_ids
        else:
            raise ValueError("tasks must be a non-empty list of task ids (str) or TaskMetadata objects.")

        if invalid:
            raise ValueError(f"The following specified tasks do not exist in the benchmark: {invalid}")

        # Deduplicate while preserving first-occurrence order.
        seen: set[str] = set()
        ordered: list[str] = []
        for tid in task_ids:
            if tid not in seen:
                seen.add(tid)
                ordered.append(tid)

        return self.model_copy(update={"task_ids": ordered})

    def subset_from_glob(self, glob_key: str, glob_pattern: str) -> "BenchmarkConfig":
        """Return a new ``BenchmarkConfig`` containing only tasks whose ``glob_key`` matches ``glob_pattern``.

        ``glob_key`` accepts any top-level ``TaskMetadata`` field (``id``,
        ``split``, ``abstract_description``, ``recommended_max_steps``) or
        ``extra_info.<key>`` via dot-notation. ``glob_pattern`` is a standard
        Unix shell wildcard.
        """
        current = self.tasks()
        if glob_key.startswith("extra_info."):
            extra_key = glob_key[len("extra_info.") :]
            matches = [
                tm
                for tm in current.values()
                if extra_key in tm.extra_info and fnmatch.fnmatch(str(tm.extra_info[extra_key]), glob_pattern)
            ]
        else:
            matches = [
                tm
                for tm in current.values()
                if hasattr(tm, glob_key) and fnmatch.fnmatch(str(getattr(tm, glob_key)), glob_pattern)
            ]
        if not matches:
            raise ValueError(f"No tasks found matching glob pattern '{glob_pattern}' on key '{glob_key}'")
        return self.subset_from_list([tm.id for tm in matches])

    @classmethod
    def named_subsets(cls) -> list[str]:
        """Return the names of all pre-defined subsets for this benchmark."""
        return list(cls.benchmark_metadata.named_subsets.keys())

    def named_subset(self, name: str) -> "BenchmarkConfig":
        """Return a filtered config for a pre-defined named subset.

        Equivalent to ``subset_from_glob(*benchmark_metadata.named_subsets[name])``.
        """
        named = type(self).benchmark_metadata.named_subsets
        if name not in named:
            raise KeyError(f"Unknown subset {name!r}. Available: {list(named.keys())}")
        glob_key, glob_pattern = named[name]
        return self.subset_from_glob(glob_key, glob_pattern)

    # ──────────────────────────────────────────────────────────────────────────
    # Data lifecycle (class-level; shared across all instances of a subclass)
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def install(cls) -> None:
        """Populate the per-task execution cache with heavy data needed at task-run time.

        Override in subclasses that ship minimal ``task_metadata.json`` and
        need to download or compute heavier per-task execution data (e.g.
        SWE-bench problem statements, OSWorld evaluator configs). The default
        is a no-op.

        ``install()`` MUST NOT mutate ``task_metadata`` — that registry is
        populated at class-definition time from a shipped file (or declared
        directly) and is the stable source of truth. Heavy execution data
        belongs in per-task JSON files under ``task_execution_cache_dir()``,
        read back later via ``load_task_execution_info(task_id)``.

        Must be idempotent.
        """

    @classmethod
    def uninstall(cls) -> None:
        """Remove assets installed by ``install()``. Default: no-op."""

    @classmethod
    def cache_dir(cls) -> Path:
        """Directory where this benchmark config may store files.

        Defaults to ``~/.cube/<benchmark_name>/``. Override if a different
        caching strategy is required.
        """
        return get_cache_dir(cls.benchmark_metadata.name)

    @classmethod
    def task_execution_cache_dir(cls) -> Path:
        """Directory where per-task execution data is cached by ``install()``."""
        return cls.cache_dir() / "tasks_execution_info"

    @classmethod
    def load_task_execution_info(cls, task_id: str) -> dict[str, Any]:
        """Read heavy per-task execution data from the cache. Called from ``TaskConfig.make`` on workers.

        Raises ``RuntimeError`` if the cache file is missing — signals that
        ``install()`` has not been run. Callers should not silently swallow
        this.
        """
        cache_file = cls.task_execution_cache_dir() / f"{task_id}.json"
        if not cache_file.exists():
            raise RuntimeError(
                f"No execution data for {task_id!r}. Run `{cls.__name__}.install()` to populate the execution cache."
            )
        return json.loads(cache_file.read_text())

    # ──────────────────────────────────────────────────────────────────────────
    # The factory
    # ──────────────────────────────────────────────────────────────────────────

    def make(self, infra: InfraConfig | None = None) -> "Benchmark":
        """Instantiate the paired ``Benchmark`` and return it ready to spawn tasks.

        Steps:

        1. For every declared resource whose ``provision_status(infra) != 'ready'``,
           call ``infra.provision(resource)``. Idempotent.
        2. Instantiate ``type(self).benchmark_class(config=self)``.
        3. Call ``benchmark.setup()`` so the returned instance is live.

        If ``resources`` is non-empty but ``infra`` is None, provisioning is
        skipped with a debug log. Benchmarks that rely entirely on task-scoped
        (L3) resources may not need infra at ``make`` time — their resources
        are launched per-task by the task itself.
        """
        if self.resources:
            if infra is None:
                logger.debug(
                    "%s.make() called without infra but %d resource(s) are declared; "
                    "skipping provisioning (task-scoped resources will be launched per-task).",
                    type(self).__name__,
                    len(self.resources),
                )
            else:
                for resource in self.resources:
                    status = infra.provision_status(resource)
                    if status == "ready":
                        logger.info("Resource %s already provisioned on %s", resource.name, infra.fingerprint())
                        continue
                    logger.info("Provisioning resource %s on %s...", resource.name, infra.fingerprint())
                    infra.provision(resource)

        benchmark = type(self).benchmark_class(config=self)
        benchmark.setup()
        return benchmark


class Benchmark(ABC):
    """Runtime pair of ``BenchmarkConfig``. Holds live OS state.

    Not serializable. Instantiated only by ``BenchmarkConfig.make(infra)``,
    which calls ``setup()`` before returning — users never construct
    ``Benchmark`` objects directly and never call ``setup()`` manually.

    Subclasses implement ``_setup`` (populate ``self._runtime_context`` with
    shared infrastructure references) and ``close`` (tear down what ``_setup``
    created). Use as a context manager to guarantee cleanup::

        with config.make(infra) as benchmark:
            for tc in benchmark.config.get_task_configs():
                task = benchmark.spawn(tc)
                ...
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config: BenchmarkConfig = config
        self._runtime_context: RuntimeContext = {}

    @abstractmethod
    def _setup(self) -> None:
        """Create shared infrastructure and populate ``self._runtime_context``.

        Implementer hook. Called exactly once by ``setup()`` inside
        ``BenchmarkConfig.make``. Storing live handles directly on ``self``
        (outside ``_runtime_context``) is also allowed — they simply won't be
        visible to tasks via ``runtime_context``.
        """

    @abstractmethod
    def close(self) -> None:
        """Tear down runtime resources created in ``_setup()``."""

    def setup(self) -> None:
        """Public wrapper around ``_setup``. Called automatically by ``BenchmarkConfig.make``.

        Emits a debug line listing optional config fields left unset — useful
        sanity signal for minimal benchmarks.
        """
        self._setup()
        missing: list[str] = []
        if not self._runtime_context:
            missing.append("_runtime_context")
        if self.config.container_backend is None:
            missing.append("container_backend")
        if self.config.default_tool_config is None:
            missing.append("default_tool_config")
        if self.config.seed_generator is None:
            missing.append("seed_generator")
        if missing:
            logger.debug(
                "%s: optional fields not set (%s). Normal for simple benchmarks; "
                "populate in _setup() or on the config when needed.",
                type(self).__name__,
                ", ".join(missing),
            )

    def spawn(self, task_config: TaskConfig) -> Task:
        """Create and return a ``Task`` for the given config.

        Pure creation call — no subprocess, no server, no network. For a
        JSON-RPC server wrapper, use ``cube.server.make_task_jsonrpc_app``
        on the returned task.
        """
        if task_config.task_id not in self.config.tasks():
            raise ValueError(
                f"Task '{task_config.task_id}' not found in benchmark "
                f"{self.config.name!r} (current view has {self.config.num_tasks} tasks)"
            )
        return task_config.make(
            runtime_context=self._runtime_context,
            container_backend=self.config.container_backend,
        )

    # ── Context-manager sugar ─────────────────────────────────────────────────

    def __enter__(self) -> "Benchmark":
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()
