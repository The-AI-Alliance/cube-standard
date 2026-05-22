# Benchmark Layer

**Module:** `cube.benchmark` | **Layer:** 4 (task collection + shared infra)

## Purpose

A benchmark is described by a **serializable `BenchmarkConfig`** (pure Pydantic
data) and brought to life by a **runtime `Benchmark`** (plain Python class
holding OS state). The split mirrors `TaskConfig` / `Task`. Authors subclass
both; `BenchmarkConfig.make(infra)` provisions resources and constructs the
runtime pair.

```
BenchmarkConfig   →  make(infra)  →  Benchmark
  serializable                       not serializable
  subset_from_list()                 _setup() / close()
  named_subset()                     spawn()
  install()                          _runtime_context
```

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
    num_tasks: int = 0                  # full-benchmark size (pre-subset)
    tags: list[str] = []
    reset_isolation: ResetIsolation | None = None  # snapshot/restart/app_level/new_instance
    named_subsets: dict[str, tuple[str, str]] = {} # name → (glob_key, glob_pattern)
    benchmark_hint_prompt: str | None = None       # optional benchmark-wide prompt hint
```

Cube authors needing additional benchmark-level fields subclass
`BenchmarkMetadata` with named typed fields. The base class accepts only
the framework-defined fields above.

`benchmark_hint_prompt` is a concise, generic hint a harness may prepend
to the agent's prompt for any task in this benchmark — e.g. a short
workflow summary or a clarification of conventions shared across tasks.
Keep it short and generic: a generalist agent should remain competitive
without it; the field exists so benchmarks with unusual conventions can
opt into a level playing field for evaluations that report numbers with
the hint applied. The framework only declares the slot — actual prompt
assembly happens in the harness.

`reset_isolation` is informational for harness users to reason about parallelism:
- `SNAPSHOT` — VM reverts to savestate (~5s)
- `RESTART` — container/VM stopped and restarted (~30s)
- `APP_LEVEL` — scripts reset app state, VM stays alive (~5s; unsafe with workers on same VM)
- `NEW_INSTANCE` — fresh VM per task (~2–4 min)

### `BenchmarkConfig` (abstract, Pydantic — serializable)

`BenchmarkConfig` is generic over the task-metadata type:

```python
class BenchmarkConfig[TTMetadata: TaskMetadata](TypedBaseModel, ABC): ...
```

The narrowing is method-level only — `task_metadata: ClassVar` stays at
the base type (PEP 526 forbids type parameters inside `ClassVar`, and
`dict` is invariant), and `tasks()` returns the covariant `Mapping[str,
TTMetadata]` view at every read site. Subclassing is opt-in; existing
unparametrised subclasses keep working unchanged.

**Required class-level attributes** (ClassVar):
```python
class MyBenchmarkConfig(BenchmarkConfig):
    benchmark_metadata: ClassVar[BenchmarkMetadata]
    task_metadata: ClassVar[dict[str, TaskMetadata]]
    task_config_class: ClassVar[type[TaskConfig]]
    benchmark_class: ClassVar[type[Benchmark]]
```

`__init_subclass__` validates every concrete subclass has all four. The first
two are auto-loaded from files next to the module if not declared:
- `benchmark_metadata.{json,csv}` → `benchmark_metadata_from_{json,csv}`
- `task_metadata.{json,csv}` → `task_metadata_from_{json,csv}`

The task registry stored in `task_metadata` is populated at class-definition
time — `install()` MUST NOT mutate it.

**Instance fields (Pydantic, serializable):**
```python
task_ids: list[str] | None = None          # None = all; populated by subset_from_*
resources: list[ResourceConfig] = []       # resource dependencies (L2/L3)
tool_config: ToolConfig | None             # applied to every task by the default get_task_configs(); override get_task_configs() for per-task variation
seed_generator: AbstractSeedGenerator | None # yields seeds per TaskMetadata
add_task_clarification: bool = False       # surface TaskMetadata.task_clarification to the agent
```

`add_task_clarification` opts a run into the per-task clarifications
declared on `TaskMetadata.task_clarification` (see
[task/spec.md](../task/spec.md)). Default `False` so a run reproduces the
original benchmark wording untouched and stays comparable with baselines
that ignored these clarifications. Harnesses interpret the flag during
prompt assembly; the framework only carries the intent.

Per-task container needs are declared via `TaskMetadata.container_config`
(`ContainerConfig`) and provisioned through the injected `InfraConfig` — see
[resource/spec.md](../resource/spec.md) and
[container/spec.md](../container/spec.md). The legacy `container_backend` field
has been removed.

`seed_generator` accepts any `AbstractSeedGenerator` subclass. Note that
`AbstractSeedGenerator` is itself a `TypedBaseModel` (changed from a plain
`BaseModel` in the BenchmarkConfig split): cube authors with custom seed
generators must ensure their subclasses inherit through `TypedBaseModel` and
populate the `_type` discriminator, otherwise deserialization on workers will
fail with a `_type`-not-found error.

Subsets are represented entirely by `task_ids` — no ClassVar shadowing or
private-attr hacks. `model_copy(update={"task_ids": [...]})` is all that
happens.

**Concrete methods:**

- `tasks() -> Mapping[str, TTMetadata]` — class-level `task_metadata` filtered by
  `task_ids` (full dict when `task_ids is None`). Returns a read-only
  `Mapping` so subclasses parametrised as `BenchmarkConfig[FooTaskMetadata]`
  get a covariantly-narrowed view at every read site; the runtime value is
  still a `dict`.
- `num_tasks` (property) — `len(self.tasks())`; differs from
  `benchmark_metadata.num_tasks` for subsets.
- `name` (property) — `self.benchmark_metadata.name`.
- `get_task_configs()` → `Generator[TaskConfig]` — yields one
  `task_config_class` per task in `tasks()`, expanding via `seed_generator` if
  set.
- `subset_from_list(tasks, benchmark_name_suffix="custom")` → `Self` (same
  subclass, new instance) with `task_ids` populated. Accepts ids or `TaskMetadata` objects.
  Duplicates deduped (first-wins order).
- `subset_from_glob(glob_key, pattern)` → `Self` (same subclass, new instance). `glob_key`
  is any top-level field on the (subclassed) `TaskMetadata` — built-ins like
  `id` / `split` / `abstract_description`, or any named field declared by a
  `TaskMetadata` subclass.
- `named_subsets()` (classmethod) → list of names from
  `benchmark_metadata.named_subsets`.
- `named_subset(name)` → `Self` (same subclass, new instance) via `subset_from_glob(*...)`.

  All three subsetting methods return `Self`, so chained calls and
  subclass-typed assignments preserve the concrete type without
  `# type: ignore`.

**Class-level data lifecycle (classmethods):**

- `install()` — populate the per-task execution cache with heavy data
  needed at task-run time (SWE-bench problem statements, OSWorld
  evaluator configs). Default: no-op. Must be idempotent. MUST NOT
  mutate `task_metadata`. By convention writes one JSON per task to
  `cls.task_config_class.task_execution_cache_dir() / f"{task_id}.json"` —
  the path has a single owner (`TaskConfig`), and the same cache dir is
  read back on workers via `self.load_task_execution_info()` inside
  `TaskConfig.make()`.
- `uninstall()` — remove assets installed by `install()`. Default: no-op.
- `cache_dir()` → `~/.cube/<benchmark_metadata.name>/` (overridable).

The per-task execution cache helpers (`task_execution_cache_dir()`,
`load_task_execution_info()`, `verify_installed()`) live on
`TaskConfig` — see [task/spec.md](../task/spec.md) for the
worker-side surface. `BenchmarkConfig.__init_subclass__` back-stamps
`cls.cache_dir()` onto `task_config_class._benchmark_cache_dir` so
`task_execution_cache_dir()` lives directly under `cache_dir()`, and
`install()` writes via `cls.task_config_class.task_execution_cache_dir()`
so the path has a single definition.

**Metadata loaders (staticmethods, also usable at class definition):**
- `benchmark_metadata_from_{json,csv}(path)` → `BenchmarkMetadata`
- `task_metadata_from_{json,csv}(path)` → `dict[str, TaskMetadata]`

CSV complex fields (`authors`, `tags`, `requirements`, `container_config`)
must be JSON-encoded strings in their cells. Subclass-specific fields are
encoded as their own columns; complex types in subclass fields go through
JSON-encoded strings as well.

**The factory:**
- `make(infra: InfraConfig | None = None) -> Benchmark` — for every
  resource whose `infra.provision_status(resource) != "ready"`, call
  `infra.provision(resource)` (idempotent), then instantiate
  `type(self).benchmark_class(config=self, infra=infra)`, call
  `benchmark.setup()`, and return the live `Benchmark`. `infra` is
  forwarded to the runtime constructor so subclasses can reach it via
  `self._infra` from `_setup()`. When `infra` is None and `resources` is
  non-empty, provisioning is skipped with a debug log — benchmarks that use
  only task-scoped (L3) resources launched per-task can legitimately pass
  `infra=None` at `make` time.

### `Benchmark` (abstract, plain Python class — not serializable)

Runtime pair produced by `BenchmarkConfig.make(infra)`. Holds only live OS
state. Not Pydantic — no fields, no `arbitrary_types_allowed`, nothing to
round-trip.

```python
class Benchmark[TBenchConfig: BenchmarkConfig](ABC):
    def __init__(self, config: TBenchConfig, infra: InfraConfig | None = None) -> None:
        self.config: TBenchConfig = config
        self._infra: InfraConfig | None = infra
        self._runtime_context: RuntimeContext = {}
```

`infra` is forwarded by `BenchmarkConfig.make(infra)` and stashed as
`self._infra` so cubes that publish it into `runtime_context["infra"]` for
per-task container launches can do so from `_setup()` without overriding
`__init__` or `make`.

**Abstract methods:**
- `_setup()` — create shared infrastructure, populate `self._runtime_context`.
  Cubes that thread per-task infra publish it here (e.g.
  `self._runtime_context["infra"] = self._infra`).
- `close()` — tear down what `_setup()` created.

**Concrete methods:**
- `setup()` — public wrapper. Owns the "harness startup" hook: if `self._infra`
  is set, calls `self._infra.cleanup_stale()` to reclaim TTL-expired resources
  from prior crashed runs *before* dispatching to `_setup()`. Cube authors do
  not call `cleanup_stale()` themselves — the base handles it for every
  benchmark. `cleanup_stale` failures are logged at WARNING and never block
  setup. Concurrent benchmarks in the same resource group are unaffected
  because only resources with `cube:expires_at` in the past are deleted.
  Then calls `_setup()`. Emits a debug log listing unset optional config
  fields. Called exactly once by `make()`.
- `spawn(task_config)` — validate `task_config.task_id` against
  `self.config.tasks()`, then call
  `task_config.make(runtime_context=self._runtime_context)`.
- `__enter__` / `__exit__` — context-manager wrappers. Use
  `with config.make(infra) as bench:` to guarantee cleanup.

### `ResetIsolation` enum
`SNAPSHOT | RESTART | APP_LEVEL | NEW_INSTANCE`

### `RuntimeContext`
Re-exported from `cube.task`. A `dict[str, Any]` populated by `_setup()`,
passed to every Task spawned from that benchmark.

## Invariants

1. Every concrete `BenchmarkConfig` subclass declares all four ClassVars (or
   has matching files for metadata) — enforced at class definition.
2. `BenchmarkConfig` instances never change class-level `task_metadata`.
   Subsets narrow via `task_ids` only.
3. A `Benchmark` returned from `make()` is always in a ready state:
   `setup()` has been called, resources have been provisioned if needed.
   Users never call `setup()` directly.
4. `BenchmarkConfig.install()` is idempotent, safe to call multiple times,
   and MUST NOT mutate `task_metadata`.
5. `spawn()` is pure creation — no subprocesses, no servers, no network.
   Server semantics live in the server layer.
6. `TaskConfig` is self-contained on workers. `metadata` (and any
   `execution_info` populated inside `make()`) travel with the config;
   workers never import the owning `BenchmarkConfig`.

## Contracts for implementers

**Minimal benchmark (Config + Benchmark pair):**
```python
class MyBenchmark(Benchmark):
    def _setup(self) -> None: pass
    def close(self) -> None: pass

class MyBenchmarkConfig(BenchmarkConfig):
    benchmark_metadata = BenchmarkMetadata(name="mine", version="0.1", description="...")
    task_metadata = {"t1": TaskMetadata(id="t1"), "t2": TaskMetadata(id="t2")}
    task_config_class = MyTaskConfig
    benchmark_class = MyBenchmark
```

**Benchmark with install-time data (large datasets):**
```python
class SWEBenchConfig(BenchmarkConfig):
    # task_metadata is auto-loaded from task_metadata.json next to this module
    task_config_class = SWEBenchTaskConfig
    benchmark_class = SWEBench

    @classmethod
    def install(cls) -> None:
        # Download problem statements, patches, tests — write per-task JSON to
        # cls.task_config_class.task_execution_cache_dir(). Idempotent.
        # task_metadata untouched. Read back on workers inside
        # TaskConfig.make() via self.load_task_execution_info() and validated
        # against a typed TaskExecutionInfo subclass surfaced as
        # Task.execution_info.
        ...
```

**Shared L2 resource (WebArena/WorkArena pattern):**
```python
class MyBenchConfig(BenchmarkConfig):
    resources: list[ResourceConfig] = [DockerServiceConfig(name="..", scope="benchmark", ...)]
    # make(infra) will provision this resource before setup.

class MyBench(Benchmark):
    def _setup(self) -> None:
        # Shared server was provisioned by make(); stash handle/URL in runtime_context.
        self._runtime_context["server_url"] = "..."
```

**Typical usage:**
```python
config = MyBenchmarkConfig().named_subset("l1").subset_from_glob("difficulty", "easy")
config.install()                    # one-time: populate task-exec cache
with config.make(infra) as bench:   # resources provisioned + setup() run
    for tc in config.get_task_configs():
        task = bench.spawn(tc)
        try:
            obs, info = task.reset()
            ...
        finally:
            task.close()
```

## Composition

### `CompositeBenchmarkConfig` (in `cube.benchmark`)
Combines multiple `BenchmarkConfig`s into one serializable suite. Any
sub-config may itself be another `CompositeBenchmarkConfig` — composites nest
freely.

```python
class CompositeBenchmarkConfig(BenchmarkConfig):
    # benchmark_metadata and task_metadata are @property (computed from
    # sub_bench_configs at access time).  BenchmarkConfig.__init_subclass__
    # detects property descriptors in the MRO via _is_dynamic() and skips
    # file auto-load for those names — no ClassVar guard flag needed.
    benchmark_class: ClassVar = CompositeBenchmark

    sub_bench_configs: list[SerializeAsAny[BenchmarkConfig]]
    composite_name: str = "composite"
    composite_version: str = "0.0.0"
    composite_description: str = ""
```

- `benchmark_metadata` and `task_metadata` are exposed as **@property** that
  compute from `sub_bench_configs` at access time. `task_metadata` keys are prefixed
  by the sub-benchmark's name (`"{sub.name}/{task_id}"`), guaranteeing
  uniqueness across the composite even when two sub-benchmarks share a task id.
- Construction raises `ValueError` if two sub-configs share a
  `benchmark_metadata.name`.
- `get_task_configs()` emits each sub-config's TaskConfigs **unchanged in
  type** — the clone preserves the sub's native `TaskConfig` subclass,
  including its embedded `metadata`. Only `sub_bench_name` is updated: set
  to the sub's name for flat composites, or prepended as
  `"{outer_name}/{inner_sub_bench_name}"` for nested composites so the full
  routing path is preserved. `task_id` reflects the updated `sub_bench_name`
  automatically (it is a derived `@property`). No wrapper class.
- `task_ids` (instance-level subset) filters at the prefixed level.
- `make(infra)` calls `sub.make(infra)` for every sub_config in order. On any
  failure, already-built sub-benchmarks are closed before the error
  propagates. Returns a `CompositeBenchmark` holding
  `sub_benchmarks: dict[str, Benchmark]`.
- `install()` and `uninstall()` are **instance methods** (not classmethods like
  the base) that delegate to every `sub.install()` / `sub.uninstall()` —
  because the list of sub-configs is instance state. `cube install <name>` is
  not supported for composite benchmarks; install each sub-benchmark
  individually instead.

### `CompositeBenchmark`
Runtime pair. Holds `sub_benchmarks: dict[str, Benchmark]`. `spawn(task_config)`
routes via `task_config.sub_bench_name`, which is a `"/"`-joined path for
nested composites (e.g. `"inner-suite/bench-a"`). Each level peels the first
component, looks it up in `sub_benchmarks`, and either delegates to the
inner `CompositeBenchmark.spawn()` (nested case) or calls
`task_config.make(runtime_context=sub_bench._runtime_context)`
directly at the leaf — bypassing the leaf's own `spawn()` validation, which
would reject the composite-prefixed `task_id`. A TaskConfig with
`sub_bench_name=None` or an unknown first component raises `ValueError`.
`close()` closes every sub-benchmark; exceptions are logged but not
re-raised so one failing sub-benchmark does not block teardown.

### Usage
```python
suite = CompositeBenchmarkConfig(
    sub_bench_configs=[
        WorkArenaConfig().named_subset("l1"),
        OSWorldConfig().subset_from_list(["chrome-1", "chrome-2"]),
        ArithmeticConfig(),
    ],
    composite_name="multi-suite",
)
suite.install()                     # delegates to each sub_config.install()
with suite.make(infra) as bench:
    for tc in suite.get_task_configs():
        task = bench.spawn(tc)
        ...
```

## Gotchas

- `task_metadata` is a ClassVar — modifying it on an instance won't work. Use
  `subset_from_list()` to get a filtered view via `task_ids`.
- `get_task_configs()` is a generator. Iterate twice and it yields twice (new
  configs each time). Callers should materialize into a list if reusing.
- File auto-loading uses `sys.modules[cls.__module__].__file__` — in unusual
  import configurations (e.g., module not in `sys.modules`), auto-load
  silently does nothing. Declare ClassVars explicitly in those cases.
- `named_subsets` values are `(glob_key, glob_pattern)` tuples. JSON-from-file
  loads them as lists — the `TypedBaseModel` will coerce to tuple.
- `install()` never populates `task_metadata`. That registry is declared at
  class-definition time (directly or via file auto-load). `install()` writes
  heavy execution-time data to the per-task cache under
  `cls.task_config_class.task_execution_cache_dir()`, read back on workers
  inside `TaskConfig.make()` via `self.load_task_execution_info()` and
  surfaced as `Task.execution_info` (a typed `TaskExecutionInfo` subclass).
