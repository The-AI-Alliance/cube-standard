# Benchmark Layer

**Module:** `cube.benchmark` | **Layer:** 4 (task collection + shared infra)

## Purpose

A `Benchmark` bundles a collection of tasks with shared infrastructure and metadata.
Subclasses declare class-level registries (`benchmark_metadata`, `task_metadata`,
`task_config_class`) and implement `_setup`/`close` for shared resources. The base class
wires everything: loading metadata from files, subsetting by glob, spawning tasks,
install-time metadata caching.

## Public API

### `BenchmarkMetadata` (serializable)
```python
class BenchmarkMetadata(TypedBaseModel):
    name: str                           # required
    version: str                        # required
    description: str                    # required
    authors: list[str] = []
    license: str = ""
    requirements: dict[str, Any] = {}   # hardware/OS/container requirements
    num_tasks: int = 0
    tags: list[str] = []
    reset_isolation: ResetIsolation | None = None  # snapshot/restart/app_level/new_instance
    named_subsets: dict[str, tuple[str, str]] = {} # name → (glob_key, glob_pattern)
    extra_info: dict[str, Any] = {}
```

`reset_isolation` is informational for harness users to reason about parallelism:
- `SNAPSHOT` — VM reverts to savestate (~5s)
- `RESTART` — container/VM stopped and restarted (~30s)
- `APP_LEVEL` — scripts reset app state, VM stays alive (~5s; unsafe with workers on same VM)
- `NEW_INSTANCE` — fresh VM per task (~2–4 min)

### `Benchmark` (abstract, Pydantic)

**Required class-level attributes** (ClassVar, not constructor params):
```python
class MyBenchmark(Benchmark):
    benchmark_metadata: ClassVar[BenchmarkMetadata]
    task_metadata: ClassVar[dict[str, TaskMetadata]]
    task_config_class: ClassVar[type[TaskConfig]]
```

`__init_subclass__` validates every concrete subclass has all three. `benchmark_metadata`
and `task_metadata` are auto-loaded from files next to the module if not declared:
- `benchmark_metadata.{json,csv}` → `benchmark_metadata_from_{json,csv}`
- `task_metadata.{json,csv}` → `task_metadata_from_{json,csv}`

If no `task_metadata` file and no declaration → `TypeError` at class definition time.

**Optional constructor fields:**
```python
resources: list[ResourceConfig] = []        # resource dependencies (L2/L3)
container_backend: ContainerBackend | None  # passed to each Task
tool_config: ToolConfig | None              # applied to every task by the default get_task_configs(); override get_task_configs() for per-task variation
seed_generator: AbstractSeedGenerator | None # yields seeds per TaskMetadata
```

**Abstract methods:**
- `_setup()` — create shared infrastructure, populate `self._runtime_context: RuntimeContext`
- `close()` — tear down what `_setup()` created

**Concrete methods:**

- `setup()` — public. Calls `_setup()`. Logs a debug line listing unset optional fields.
- `get_task_configs()` → `Generator[TaskConfig]` — yields one `task_config_class` per task,
  expanding via `seed_generator` if set
- `spawn(task_config)` → `Task` — calls `task_config.make(runtime_context=..., container_backend=...)`
  after validating `task_config.task_id` is known
- `subset_from_glob(glob_key, pattern)` → `Benchmark` — glob-filtered copy. `glob_key` can be
  any `TaskMetadata` field or `extra_info.<key>` (dot-notation)
- `subset_from_list(tasks, suffix="custom")` → `Benchmark` — filter by task IDs or `TaskMetadata` list
- `named_subsets()` (classmethod) → list of names from `benchmark_metadata.named_subsets`
- `named_subset(name)` → `Benchmark` — resolves a named subset via `subset_from_glob`

**Install-time API (classmethods, no instance required):**
- `install()` — download assets, build `task_metadata`, save to `task_metadata.json`,
  update `cls.task_metadata`. Default: no-op.
- `uninstall()` — remove downloaded assets. Default: no-op.
- `cache_dir()` → `~/.cube/<name>/` (overridable)
- `task_execution_cache_dir()` → `~/.cube/<name>/tasks_execution_info/`
- `load_task_execution_info(task_id)` → dict — reads `{cache_dir}/tasks_execution_info/{task_id}.json`;
  raises `RuntimeError` if missing

**Metadata loaders (staticmethods, also usable at class definition):**
- `benchmark_metadata_from_{json,csv}(path)` → `BenchmarkMetadata`
- `task_metadata_from_{json,csv}(path)` → `dict[str, TaskMetadata]`

CSV complex fields (`authors`, `tags`, `requirements`, `extra_info`, `container_config`)
must be JSON-encoded strings in their cells.

### `ResetIsolation` enum
`SNAPSHOT | RESTART | APP_LEVEL | NEW_INSTANCE`

### `RuntimeContext`
Re-exported from `cube.task`. A `dict[str, Any]` populated by `_setup()`, passed to every Task.

## Invariants

1. Every concrete `Benchmark` subclass declares all three ClassVars (or has matching
   files) — enforced at class definition.
2. `setup()` refuses to run with empty `task_metadata` — forces `install()` first.
3. `subset_from_*()` returns a new instance with fresh private state — caller MUST call
   `.setup()` on the subset before use (PrivateAttrs are reset to defaults).
4. `spawn()` is pure creation — no subprocesses, no servers, no network. Server semantics
   live in the server layer.
5. `install()` must be idempotent and safe to call multiple times.

## Contracts for implementers

**Minimal benchmark:**
```python
class MyBenchmark(Benchmark):
    benchmark_metadata = BenchmarkMetadata(name="mine", version="0.1", description="...")
    task_metadata = {"t1": TaskMetadata(id="t1"), "t2": TaskMetadata(id="t2")}
    task_config_class = MyTaskConfig

    def _setup(self) -> None: pass
    def close(self) -> None: pass
```

**Benchmark with install-time metadata (large datasets):**
```python
class SWEBench(Benchmark):
    task_metadata: ClassVar[dict[str, TaskMetadata]] = {}   # placeholder
    task_config_class = SWEBenchTaskConfig

    @classmethod
    def install(cls) -> None:
        # Download dataset, build task_metadata, save to task_metadata.json,
        # update cls.task_metadata in memory
        ...

    def _setup(self) -> None: ...
    def close(self) -> None: ...
```

**Shared L2 resource (WebArena/WorkArena pattern):**
```python
class MyBench(Benchmark):
    resources: list[ResourceConfig] = [DockerServiceConfig(name="..", scope="benchmark", ...)]

    def _setup(self) -> None:
        # Launch shared server, stash handle/URL in self._runtime_context
        self._runtime_context["server_url"] = "..."
```

## Gotchas

- `task_metadata` is a ClassVar — modifying it on an instance won't work. Use
  `subset_from_list()` to get a filtered copy.
- `subset_from_list()` does a deep copy, but `subset_from_glob` results must still call
  `.setup()` — PrivateAttrs are reset.
- `get_task_configs()` is a generator. If you iterate twice, it yields twice (new configs
  each time) — callers should materialize into a list if reusing.
- File auto-loading uses `sys.modules[cls.__module__].__file__` — in unusual import
  configurations (e.g., module not in `sys.modules`), auto-load silently does nothing.
- `named_subsets` values are `(glob_key, glob_pattern)` tuples. JSON-from-file loads
  them as lists — the `TypedBaseModel` will coerce to tuple.
