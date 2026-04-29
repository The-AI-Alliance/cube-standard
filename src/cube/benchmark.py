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
``task_metadata`` dict remains the driver-side registry; ``get_task_configs``
stamps each emitted ``TaskConfig`` with its full ``TaskMetadata`` so workers
have everything they need without importing the owning ``BenchmarkConfig``.

Composition
-----------
``CompositeBenchmarkConfig`` holds a list of ``BenchmarkConfig``s and emits
each sub-config's TaskConfigs unchanged in type, with ``task_id`` prefixed by
the sub-benchmark's name and a ``sub_bench_name`` routing tag set.
``CompositeBenchmark.spawn()`` reads ``sub_bench_name`` to route the task to
its origin sub-benchmark's ``_runtime_context``.
"""

from __future__ import annotations

import csv
import enum
import fnmatch
import json
import logging
import sys
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any, ClassVar, Generator, Mapping, Self, Sequence

from pydantic import Field, SerializeAsAny

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
            "Example: {'train_only': ('split', 'train')}"
        ),
    )


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

    # ── Instance fields (user-configurable; safely serializable) ──────────────
    task_ids: list[str] | None = Field(
        default=None,
        description=(
            "Subset selector: if None, ``tasks()`` returns every entry in the class-level "
            "task_metadata. Populated by ``subset_from_list`` / ``subset_from_glob`` / "
            "``named_subset`` to narrow the set without touching the ClassVar."
        ),
    )
    # ``SerializeAsAny`` on every polymorphic field so subclass-specific state
    # survives JSON round-trip. Without it Pydantic dumps only the declared
    # base type's fields — subclass extras are silently stripped and the
    # reloaded value is semantically different from the original. Required for
    # reliable cross-process transport via ``make_benchmark_rpc_server`` and
    # any future Ray / storage round-trip.
    resources: list[SerializeAsAny[ResourceConfig]] = Field(
        default_factory=list,
        description=(
            "Declared resource dependencies. ``make(infra)`` calls ``infra.provision(r)`` "
            "for each entry whose ``provision_status`` is not ``ready`` before setup runs."
        ),
    )
    container_backend: SerializeAsAny[ContainerBackend] | None = Field(
        default=None,
        description="Optional container backend passed through to every spawned task.",
        deprecated=True,
    )
    tool_config: SerializeAsAny[ToolConfig] | None = Field(
        default=None,
        description="Tool config applied uniformly to every task by the default get_task_configs(); override get_task_configs() for per-task variation.",
    )
    seed_generator: SerializeAsAny[AbstractSeedGenerator] | None = Field(
        default=None,
        description="Optional seed generator yielding per-task seeds during get_task_configs().",
    )

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

        Complex fields (``authors``, ``tags``, ``requirements``) must be
        stored as JSON-encoded strings. Subclass-specific fields are
        encoded as their own columns; complex types in subclass fields go
        through JSON-encoded strings just like the built-in complex fields.
        """
        _JSON_FIELDS = ("authors", "tags", "requirements")
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

        Each row is one task. ``id`` is required. Complex built-in fields
        (``container_config``) must be JSON-encoded strings. Subclass-specific
        fields are encoded as their own columns; complex types in subclass
        fields go through JSON-encoded strings as well.
        """
        _JSON_FIELDS = ("container_config",)
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
        """Validate that every concrete subclass wires the class-level registries.

        Auto-loads ``benchmark_metadata`` / ``task_metadata`` from files next
        to the module if neither a ClassVar value nor a descriptor (e.g.
        ``@property``) is declared on the subclass. Subclasses that compute
        these dynamically (e.g. ``CompositeBenchmarkConfig``) just declare
        them as ``@property`` and the validation steps out of the way.

        ``task_config_class`` and ``benchmark_class`` are always required.
        """
        super().__init_subclass__(**kwargs)

        # Abstract intermediates still have unimplemented abstract methods — skip.
        if getattr(cls, "__abstractmethods__", None):
            return

        # Pydantic synthesises a parametrised intermediate class for every
        # ``BenchmarkConfig[FooTaskMetadata]`` expression (e.g. when a user
        # writes ``class FooBenchmarkConfig(BenchmarkConfig[FooTaskMetadata]):``,
        # Pydantic constructs ``BenchmarkConfig[FooTaskMetadata]`` first and
        # then the user class subclasses it). The intermediate has no
        # ClassVar wiring of its own and is not a real concrete subclass —
        # validation should run only on the user class that follows.
        # ``__pydantic_generic_metadata__`` is not yet finalised when
        # ``__init_subclass__`` fires on the intermediate, but Pydantic
        # gives the intermediate a name containing ``[`` (e.g.
        # ``"BenchmarkConfig[FooTaskMetadata]"``); user class names never do.
        if "[" in cls.__name__:
            return

        module_file = getattr(sys.modules.get(cls.__module__), "__file__", None)
        module_dir = Path(module_file).resolve().parent if module_file else None

        def _is_dynamic(name: str) -> bool:
            """True iff *name* is declared as a ``@property`` somewhere in the MRO
            below ``BenchmarkConfig`` — meaning the attribute is computed at access
            time and should not be overwritten with a file-loaded ClassVar value.
            """
            return any(isinstance(klass.__dict__.get(name), property) for klass in cls.__mro__)

        # ── benchmark_metadata ───────────────────────────────────────────────
        if not _is_dynamic("benchmark_metadata"):
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
                elif not hasattr(cls, "benchmark_metadata"):
                    raise TypeError(
                        f"Concrete benchmark config class {cls.__name__} must define "
                        f"'benchmark_metadata' as a ClassVar, declare it as a @property, "
                        f"or ship 'benchmark_metadata.json' / '.csv' next to the module."
                    )

            bench_meta = getattr(cls, "benchmark_metadata", None)
            if not isinstance(bench_meta, BenchmarkMetadata):
                raise TypeError(
                    f"'benchmark_metadata' in {cls.__name__} must be a BenchmarkMetadata instance, "
                    f"not {type(bench_meta).__name__}"
                )

        # ── task_metadata ────────────────────────────────────────────────────
        if not _is_dynamic("task_metadata"):
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
                elif not hasattr(cls, "task_metadata"):
                    raise TypeError(
                        f"{cls.__name__} must declare 'task_metadata' as a ClassVar, declare it as a "
                        f"@property, or ship 'task_metadata.json' / '.csv' next to the module."
                    )

            task_meta = getattr(cls, "task_metadata", None)
            if not isinstance(task_meta, dict):
                raise TypeError(f"'task_metadata' in {cls.__name__} must be a dict, not {type(task_meta).__name__}")

        # ── task_config_class ───────────────────────────────────────────────
        # Always required. Subclasses that override ``get_task_configs`` (and
        # therefore don't use the factory) can set ``task_config_class = TaskConfig``
        # as a placeholder — see ``CompositeBenchmarkConfig``.
        task_cfg = getattr(cls, "task_config_class", None)
        if not (isinstance(task_cfg, type) and issubclass(task_cfg, TaskConfig)):
            raise TypeError(
                f"Concrete benchmark config class {cls.__name__} must define "
                f"'task_config_class' as a ClassVar subclass of TaskConfig, not {task_cfg!r}."
            )

        # ── benchmark_class ─────────────────────────────────────────────────
        bench_cls = getattr(cls, "benchmark_class", None)
        if not (isinstance(bench_cls, type) and issubclass(bench_cls, Benchmark)):
            raise TypeError(
                f"Concrete benchmark config class {cls.__name__} must define "
                f"'benchmark_class' as a ClassVar subclass of Benchmark, not {bench_cls!r}."
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Views
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        """Benchmark name from ``benchmark_metadata``. Resolves via the
        descriptor chain, so composites' @property overrides work here too.
        """
        return self.benchmark_metadata.name

    @property
    def num_tasks(self) -> int:
        """Number of tasks in the current (possibly subset) view."""
        return len(self.tasks())

    def tasks(self) -> Mapping[str, TaskMetadata]:
        """Return the current task view — ``task_metadata`` filtered by ``task_ids``.

        When ``task_ids`` is None, returns the full ``task_metadata`` registry.
        Otherwise returns a freshly-built dict containing only the declared
        ids, in the order given by ``task_ids``. Access via ``self.`` so the
        composite's @property override resolves correctly.

        The return type is ``Mapping`` (read-only) — the runtime value is a
        ``dict`` but the static contract is the wider type so subclasses can
        override with a narrower value type if they need ``TaskMetadata``
        subclass fields visible at use sites.
        """
        full = self.task_metadata
        if self.task_ids is None:
            return full
        return {tid: full[tid] for tid in self.task_ids}

    def get_task_configs(self) -> Generator[TaskConfig, None, None]:
        """Yield one ``TaskConfig`` per task (expanded by seed_generator if set).

        Stamps full ``TaskMetadata`` onto each emitted config so workers never
        need to import the owning ``BenchmarkConfig`` class to resolve metadata.

        Cubes with heavy execution data (problem statements, patches, …)
        populate ``Task.execution_info`` inside ``TaskConfig.make()`` by
        validating ``cls.load_task_execution_info(task_id)`` against a typed
        ``TaskExecutionInfo`` subclass — no override of this method needed.
        """
        for tm in self.tasks().values():
            if self.seed_generator is not None:
                for seed in self.seed_generator(tm):
                    yield self.task_config_class(
                        metadata=tm,
                        tool_config=self.tool_config,
                        seed=seed,
                    )
            else:
                yield self.task_config_class(
                    metadata=tm,
                    tool_config=self.tool_config,
                    seed=None,
                )

    # ──────────────────────────────────────────────────────────────────────────
    # Subsetting (pure data — no deep-copy or private-attr hacks)
    # ──────────────────────────────────────────────────────────────────────────

    def subset_from_list(
        self,
        tasks: Sequence[str] | Sequence[TaskMetadata],
        benchmark_name_suffix: str = "custom",  # noqa: ARG002 — accepted for call-site compat
    ) -> Self:
        """Return a new ``BenchmarkConfig`` restricted to the given tasks.

        Accepts either a sequence of task ids (strings) or a sequence of
        ``TaskMetadata`` objects (or any subclass). The returned config
        is the same subclass, with ``task_ids`` set and every other field
        inherited via ``model_copy``. No private-attr reset or ClassVar
        shadowing is required — ``TaskConfig.make()`` still resolves metadata
        via the class-level ``task_metadata`` ClassVar.

        ``benchmark_name_suffix`` is retained for call-site compatibility but
        has no effect in the new design; subsets inherit their name from the
        parent class's ``benchmark_metadata``. Use a display-layer convention
        if you need a distinct label.
        """
        current = self.tasks()
        existing_ids = set(current.keys())

        if tasks and isinstance(tasks[0], str):
            task_ids: list[str] = list(tasks)  # type: ignore[list-item]
            invalid = set(task_ids) - existing_ids
        elif tasks and isinstance(tasks[0], TaskMetadata):
            task_ids = [tm.id for tm in tasks]  # type: ignore[union-attr]
            invalid = set(task_ids) - existing_ids
        else:
            raise ValueError("tasks must be a non-empty sequence of task ids (str) or TaskMetadata objects.")

        if invalid:
            raise ValueError(f"The following specified tasks do not exist in the benchmark: {invalid}")

        # Deduplicate while preserving first-occurrence order.
        ordered = list(dict.fromkeys(task_ids))
        if not ordered:
            raise ValueError("The resulting task list cannot be empty.")
        return self.model_copy(update={"task_ids": ordered})

    def subset_from_glob(self, glob_key: str, glob_pattern: str) -> Self:
        """Return a new ``BenchmarkConfig`` containing only tasks whose ``glob_key`` matches ``glob_pattern``.

        ``glob_key`` is any top-level field on the (subclassed) ``TaskMetadata``
        — built-ins like ``id`` / ``split`` / ``abstract_description``, or any
        named field declared by a ``TaskMetadata`` subclass.
        ``glob_pattern`` is a standard Unix shell wildcard.
        """
        current = self.tasks()
        matches = [
            tm
            for tm in current.values()
            if hasattr(tm, glob_key) and fnmatch.fnmatch(str(getattr(tm, glob_key)), glob_pattern)
        ]
        if not matches:
            raise ValueError(f"No tasks found matching glob pattern '{glob_pattern}' on key '{glob_key}'")
        return self.subset_from_list(tasks=matches, benchmark_name_suffix=f"[{glob_key}={glob_pattern}]")

    @classmethod
    def named_subsets(cls) -> list[str]:
        """Return the names of all pre-defined subsets for this benchmark."""
        return list(cls.benchmark_metadata.named_subsets.keys())

    def named_subset(self, name: str) -> Self:
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

        Convention: write each task's processed data as a JSON file at
        ``cls.task_config_class.task_execution_cache_dir() / f"{task_id}.json"``
        — that path is the single source of truth, owned by ``TaskConfig``,
        and read back by workers via
        ``TaskConfig.load_task_execution_info(task_id)``.

        ``install()`` MUST NOT mutate ``task_metadata`` — that registry is
        populated at class-definition time from a shipped file (or declared
        directly) and is the stable source of truth.

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


class Benchmark[TBenchConfig: BenchmarkConfig](ABC):
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

    Type parameter ``TBenchConfig`` (bound to ``BenchmarkConfig``) lets
    cubes statically narrow ``self.config`` to a ``BenchmarkConfig`` subclass:

        # Unparametrised — ``self.config`` typed as ``BenchmarkConfig``.
        class FooBenchmark(Benchmark): ...

        # Parametrised — ``self.config`` typed as ``FooBenchmarkConfig``,
        # autocomplete and static checking work for subclass-specific fields.
        class FooBenchmark(Benchmark[FooBenchmarkConfig]): ...
    """

    def __init__(self, config: TBenchConfig) -> None:
        self.config: TBenchConfig = config
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
        if self.config.tool_config is None:
            missing.append("tool_config")
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

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()


# ──────────────────────────────────────────────────────────────────────────────
# Composite benchmarks
#
# Combine multiple BenchmarkConfigs into a single serializable suite.
# Tasks are prefixed with their sub-benchmark's name so the composite exposes a
# unique id per task. Routing back to the source sub-benchmark happens through
# a ``CompositeTaskConfig`` wrapper that carries the sub-benchmark's name and
# the original (un-prefixed) TaskConfig. The wrapper is itself a ``TaskConfig``
# so every generic code path that consumes a TaskConfig continues to work.
# ──────────────────────────────────────────────────────────────────────────────


class CompositeBenchmark(Benchmark["CompositeBenchmarkConfig"]):
    """Runtime pair of ``CompositeBenchmarkConfig``.

    Holds a dict of live sub-benchmarks keyed by sub-benchmark name. Routes
    ``spawn(task_config)`` to the matching sub-benchmark by reading
    ``task_config.sub_bench_name``. Does not own any shared infrastructure of its
    own — each sub-benchmark has its own ``_runtime_context`` and its own
    provisioned resources.
    """

    def __init__(self, config: "CompositeBenchmarkConfig") -> None:
        super().__init__(config)
        self.sub_benchmarks: dict[str, Benchmark] = {}

    def _setup(self) -> None:
        """No-op. Sub-benchmarks are instantiated by CompositeBenchmarkConfig.make
        *before* this class is constructed, and are already live when we get here.
        """

    def close(self) -> None:
        """Close every sub-benchmark; exceptions are logged but not re-raised so
        one failing sub-benchmark does not block teardown of the others.
        """
        for sub_name, sub_bench in self.sub_benchmarks.items():
            try:
                sub_bench.close()
            except Exception:
                logger.exception("Error closing sub-benchmark %r", sub_name)

    def spawn(self, task_config: TaskConfig) -> Task:
        """Route the task to its origin sub-benchmark via ``sub_bench_name``.

        For flat composites ``sub_bench_name`` is just the sub-benchmark's
        name (e.g. ``"bench-a"``).  For nested composites it is a
        ``"/"``-joined path of names from outermost to leaf
        (e.g. ``"inner-suite/bench-a"``); the path is peeled one hop at a
        time so each ``CompositeBenchmark`` in the chain delegates correctly.

        The final hop calls ``task_config.make()`` directly instead of
        ``sub_bench.spawn()`` — the latter would re-validate ``task_id``
        against the sub's ``config.tasks()`` and reject the
        composite-prefixed id.
        """
        if task_config.sub_bench_name is None:
            raise ValueError(
                f"CompositeBenchmark.spawn() expects a TaskConfig with sub_bench_name set "
                f"(emitted by CompositeBenchmarkConfig.get_task_configs()); "
                f"got {type(task_config).__name__} with sub_bench_name=None"
            )
        # Peel the first component of the (possibly multi-hop) routing path.
        parts = task_config.sub_bench_name.split("/", 1)
        next_hop = parts[0]
        remaining_path = parts[1] if len(parts) > 1 else None
        if next_hop not in self.sub_benchmarks:
            raise ValueError(
                f"Unknown sub-benchmark {next_hop!r} "
                f"(from sub_bench_name={task_config.sub_bench_name!r}); "
                f"known: {list(self.sub_benchmarks.keys())}"
            )
        sub_bench = self.sub_benchmarks[next_hop]
        if remaining_path is not None:
            # Nested composite: delegate with the remaining path so the inner
            # CompositeBenchmark can continue routing.
            return sub_bench.spawn(task_config.model_copy(update={"sub_bench_name": remaining_path}))
        # Leaf: call make() directly to bypass sub_bench.spawn() validation.
        return task_config.make(
            runtime_context=sub_bench._runtime_context,
            container_backend=sub_bench.config.container_backend,
        )


class CompositeBenchmarkConfig(BenchmarkConfig):
    """Composition of multiple BenchmarkConfigs into one serializable suite.

    ``sub_bench_configs`` is a list of any ``BenchmarkConfig`` subclasses (including
    other ``CompositeBenchmarkConfig`` instances — composites nest freely).
    The composite exposes merged, prefixed views of ``benchmark_metadata`` and
    ``task_metadata`` derived from its sub_bench_configs; no per-composite metadata
    is stored statically.

    Sub-benchmark names (``sub.benchmark_metadata.name``) must be unique across
    the composite — duplicate names raise ``ValueError`` at construction.

    Typical usage::

        suite = CompositeBenchmarkConfig(
            sub_bench_configs=[
                WorkArenaConfig().named_subset("l1"),
                OSWorldConfig().subset_from_list(["chrome-1", "chrome-2"]),
                ArithmeticConfig(),
            ],
            composite_name="my-multi-suite",
        )
        with suite.make(infra) as bench:
            for tc in suite.get_task_configs():
                task = bench.spawn(tc)
                ...
    """

    # benchmark_metadata: ClassVar[BenchmarkMetadata]
    # task_metadata: ClassVar[dict[str, TaskMetadata]]
    # benchmark_metadata / task_metadata are declared below as @property so
    # they reflect the current sub_bench_configs at access time; BenchmarkConfig's
    # __init_subclass__ detects the descriptors and skips file auto-load.

    task_config_class: ClassVar[type[TaskConfig]] = TaskConfig
    # Composite doesn't emit a single TaskConfig class — each sub_config's
    # TaskConfig subclass is preserved through the clone in get_task_configs().
    # task_config_class is set to the abstract base as a placeholder to satisfy
    # __init_subclass__ validation; it is never instantiated (get_task_configs
    # is overridden to yield sub-configs directly).

    benchmark_class: ClassVar[type[Benchmark]] = CompositeBenchmark

    # SerializeAsAny forces each element to serialize using its runtime type's
    # full field set, so subclass-specific fields (composite_name, sub_bench_configs on
    # nested composites, infra knobs on sub-configs, etc.) survive a JSON round
    # trip. Without it Pydantic dumps only the base BenchmarkConfig fields.
    sub_bench_configs: list[SerializeAsAny[BenchmarkConfig]] = Field(
        ...,
        description=(
            "Ordered list of BenchmarkConfigs to compose. Order is preserved in "
            "``get_task_configs``. Each sub-config's benchmark_metadata.name must "
            "be unique across the list."
        ),
    )
    composite_name: str = Field(
        default="composite",
        description="Display name for this composite; surfaces in benchmark_metadata.name.",
    )
    composite_version: str = Field(
        default="0.0.0", description="Version label for this composite; surfaces in benchmark_metadata.version."
    )
    composite_description: str = Field(
        default="", description="Description for this composite; surfaces in benchmark_metadata.description."
    )

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        duplicates = [name for name, count in Counter(s.name for s in self.sub_bench_configs).items() if count > 1]
        if duplicates:
            raise ValueError(
                f"CompositeBenchmarkConfig requires unique sub-benchmark names; found duplicates: {duplicates}"
            )

    @property
    def benchmark_metadata(self) -> BenchmarkMetadata:  # type: ignore[override]
        """Computed metadata reflecting the composite's current sub_bench_configs."""
        return BenchmarkMetadata(
            name=self.composite_name,
            version=self.composite_version,
            description=self.composite_description,
            num_tasks=sum(sub.num_tasks for sub in self.sub_bench_configs),
            tags=sorted({t for sub in self.sub_bench_configs for t in sub.benchmark_metadata.tags}),
        )

    @property
    def task_metadata(self) -> dict[str, TaskMetadata]:  # type: ignore[override]
        """Merged view with keys prefixed by sub-benchmark name (``"{sub_bench_name}/{task_id}"``)."""
        merged: dict[str, TaskMetadata] = {}
        for sub in self.sub_bench_configs:
            sub_bench_name = sub.name
            for tid, tm in sub.tasks().items():
                merged[f"{sub_bench_name}/{tid}"] = tm
        return merged

    # tasks() is inherited from BenchmarkConfig and reads ``task_metadata`` (our
    # property) — so subsetting via ``task_ids`` on the composite continues to
    # work. Callers should use prefixed ids.

    def get_task_configs(self) -> Generator[TaskConfig, None, None]:  # type: ignore[override]
        """Yield each sub-config's TaskConfigs, cloned with composite-level routing info.

        Each emitted TaskConfig is the sub-config's own subclass (preserving
        its native type, metadata, and any subclass-specific fields) with
        ``sub_bench_name`` set to the sub-benchmark's name. ``task_id`` is
        derived from ``metadata.id`` and ``sub_bench_name`` by ``TaskConfig``
        itself, so the prefix appears automatically. No wrapper type — the
        clone stays a fully-functional TaskConfig whose ``make()`` works
        standalone.

        ``task_ids`` (instance-level subset) filters at the composite level:
        if set, only prefixed ids in the list are emitted.
        """
        allowed_task_ids: set[str] | None = set(self.task_ids) if self.task_ids is not None else None
        for sub_bench_config in self.sub_bench_configs:
            sub_bench_name = sub_bench_config.name
            for task_config in sub_bench_config.get_task_configs():
                prefixed_id = f"{sub_bench_name}/{task_config.task_id}"
                # skip if this task is not in the allowed set
                if allowed_task_ids is not None and prefixed_id not in allowed_task_ids:
                    continue
                # For nested composites the inner composite already set
                # sub_bench_name; prepend the outer name to build the full
                # routing path (e.g. "inner-suite/bench-a").  For flat tasks
                # (sub_bench_name is None) just set the name directly.
                new_sub_bench_name = (
                    f"{sub_bench_name}/{task_config.sub_bench_name}"
                    if task_config.sub_bench_name is not None
                    else sub_bench_name
                )
                yield task_config.model_copy(update={"sub_bench_name": new_sub_bench_name})

    def make(self, infra: InfraConfig | None = None) -> CompositeBenchmark:
        """Instantiate every sub-benchmark and wire them into a CompositeBenchmark.

        Each sub-config's ``make(infra)`` is called in order; on any failure,
        already-built sub-benchmarks are closed before the error propagates.
        """
        composite = CompositeBenchmark(config=self)
        try:
            for sub_bench_config in self.sub_bench_configs:
                composite.sub_benchmarks[sub_bench_config.name] = sub_bench_config.make(infra)
        except Exception:
            for sub_bench in composite.sub_benchmarks.values():
                try:
                    sub_bench.close()
                except Exception:
                    logger.exception("Error closing partially-initialised sub-benchmark")
            raise
        composite.setup()  # no-op for CompositeBenchmark; kept for consistency
        return composite

    # Data lifecycle on composites: instance methods that delegate to every
    # sub-config.  Unlike BenchmarkConfig's classmethods, these require an
    # instance because sub_bench_configs is instance state.
    # Note: `cube install <name>` does not support composite benchmarks —
    # install each sub-benchmark individually instead.

    def install(self) -> None:  # type: ignore[override]
        """Install every sub-config. Idempotent; safe to call multiple times."""
        for sub_bench_config in self.sub_bench_configs:
            sub_bench_config.install()

    def uninstall(self) -> None:  # type: ignore[override]
        """Uninstall every sub-config."""
        for sub_bench_config in self.sub_bench_configs:
            sub_bench_config.uninstall()
