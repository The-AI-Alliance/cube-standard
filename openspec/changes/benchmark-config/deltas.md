# Deltas — BenchmarkConfig / Benchmark split + CompositeBenchmark

**Targets:** `openspec/specs/benchmark/spec.md`, `openspec/specs/task/spec.md`, `openspec/specs/testing/spec.md`, `openspec/specs/server/spec.md`, `openspec/specs/resource/spec.md`

Applied as each commit lands.

## REMOVED — `Benchmark` as a single serializable class
**Spec:** benchmark

- `Benchmark(TypedBaseModel, ABC)` with ClassVar `benchmark_metadata` /
  `task_metadata` / `task_config_class` and Pydantic fields `resources` /
  `container_backend` / `default_tool_config` / `seed_generator` — removed.
- `Benchmark.subset_from_list`, `subset_from_glob`, `named_subset`,
  `named_subsets` — removed (moved to `BenchmarkConfig`).
- `Benchmark.setup()` / `_setup()` — `setup()` becomes an implementation detail
  of `BenchmarkConfig.make()`; never called by users.
- `Benchmark.install()` / `uninstall()` / `cache_dir()` /
  `task_execution_cache_dir()` / `load_task_execution_info()` — moved to
  `BenchmarkConfig`.
- `Benchmark.spawn(task_config)` — moved to the new `Benchmark` runtime class.
- `Benchmark.benchmark_metadata_from_{json,csv}` /
  `task_metadata_from_{json,csv}` — moved to `BenchmarkConfig` (still invoked by
  `__init_subclass__`).

## ADDED — `BenchmarkConfig` (serializable)
**Spec:** benchmark

Pure Pydantic model (`TypedBaseModel, ABC`). Source of truth for benchmark
identity and task registry.

**ClassVars (populated at class definition time, unchanged loading semantics):**
- `benchmark_metadata: BenchmarkMetadata`
- `task_metadata: dict[str, TaskMetadata]`
- `task_config_class: type[TaskConfig]`
- `benchmark_class: type[Benchmark]` — new; names the runtime pair.

**Instance fields:**
- `task_ids: list[str] | None = None` — new. `None` means all tasks in the
  ClassVar; subsetting sets it to a filtered list.
- `resources: list[ResourceConfig] = []`
- `container_backend: ContainerBackend | None = None`
- `default_tool_config: ToolConfig | None = None`
- `seed_generator: AbstractSeedGenerator | None = None`

**Pure-data methods:**
- `tasks() -> dict[str, TaskMetadata]` — returns ClassVar filtered by
  `task_ids`.
- `get_task_configs() -> Generator[TaskConfig]`
- `subset_from_list(tasks, suffix="custom") -> Self` — returns
  `self.model_copy(update={"task_ids": [...], "benchmark_metadata": new_meta})`.
  No `__pydantic_private__` reset, no `object.__setattr__`.
- `subset_from_glob(key, pattern) -> Self`
- `named_subset(name) -> Self`
- `named_subsets() -> list[str]` (classmethod)

**Class-level data lifecycle (unchanged contracts, new home):**
- `install()` — populates `task_execution_cache_dir()` with per-task heavy data.
  Never mutates `task_metadata`.
- `uninstall()`
- `cache_dir() -> Path`
- `task_execution_cache_dir() -> Path`
- `load_task_execution_info(task_id) -> dict`

**The factory:**
- `make(infra: InfraConfig | None = None) -> Benchmark`
  - For each resource with `provision_status() != "ready"`, call
    `infra.provision(resource)`. Idempotent.
  - Instantiate `cls.benchmark_class(config=self)`.
  - Call `benchmark.setup()` so the returned benchmark is ready.
  - Return live `Benchmark`.

## ADDED — `Benchmark` as runtime pair (plain class)
**Spec:** benchmark

Plain Python abstract class (not Pydantic). Holds only live OS state.

**Constructor:** `Benchmark(config: BenchmarkConfig)` stores the back-reference
and initialises `_runtime_context: RuntimeContext = {}`.

**Abstract methods:**
- `_setup() -> None` — implementer hook.
- `close() -> None` — implementer hook.

**Concrete methods:**
- `setup() -> None` — public wrapper around `_setup()`, retains today's debug
  logging for unset optional fields. Called exactly once by `make()`.
- `spawn(task_config: TaskConfig) -> Task` — validates `task_config.task_id`
  against `self.config.tasks()` and calls
  `task_config.make(runtime_context=self._runtime_context, container_backend=self.config.container_backend)`.
- `__enter__` / `__exit__` — context-manager wrappers calling `close()` on exit.

**Invariant:** `Benchmark` instances never exist in an uninitialized state —
`make()` always calls `setup()` before returning.

## ADDED — `CompositeBenchmarkConfig`, `CompositeBenchmark`, `CompositeTaskConfig`
**Spec:** benchmark

`CompositeBenchmarkConfig(BenchmarkConfig)` holds `sub_configs: list[BenchmarkConfig]`:
- Merged `task_metadata` prefixes keys with the sub-benchmark name
  (`"{sub.benchmark_metadata.name}/{task_id}"`).
- Duplicate sub-benchmark names raise at construction.
- `get_task_configs()` yields `CompositeTaskConfig` wrappers.
- `make(infra)` calls `sub.make(infra)` for each sub_config and returns a
  `CompositeBenchmark` holding `sub_benchmarks: dict[str, Benchmark]`.
- Subset and named-subset methods are delegated / disabled at composite level
  (use `subset_from_list` on individual sub_configs before composition).

`CompositeTaskConfig(TaskConfig)`:
- Fields: `sub_name: str`, `inner: TaskConfig`.
- `task_id` defaults to `"{sub_name}/{inner.task_id}"` (prefixed; unique across
  the composite).
- `make(runtime_context, container_backend)` delegates to
  `inner.make(runtime_context, container_backend)`.

`CompositeBenchmark(Benchmark)`:
- `spawn(CompositeTaskConfig)` routes to `sub_benchmarks[sub_name].spawn(inner)`.
- `close()` closes every sub-benchmark.

## MODIFIED — `TaskConfig.make()` documentation
**Spec:** task

No signature change. Docstring updated to note that `task_metadata` is accessed
via the class-level `BenchmarkConfig.task_metadata` ClassVar (not the instance),
which is the reason subsetting via `task_ids` works on workers after
deserialization.

## MODIFIED — Debug-flow infra injection (closes #96)
**Spec:** testing

- Module protocol: `get_debug_benchmark(infra: InfraConfig | None = None) -> BenchmarkConfig`.
- `run_debug_suite(benchmark_name, module, *, max_steps, workers, infra=None)`
  calls `module.get_debug_benchmark(infra)` then `config.make(infra)` under
  try/finally.
- `assert_debug_tasks_reward_one(module, *, infra=None, max_steps)` — adds
  `infra` kwarg.
- `check_reset_reproducibility(module, *, infra=None)` — adds `infra` kwarg.

## MODIFIED — `make_benchmark_rpc_server` subprocess model
**Spec:** server

- Signature changes from `make_benchmark_rpc_server(benchmark, ...)` to
  `make_benchmark_rpc_server(config: BenchmarkConfig, *, infra: InfraConfig | None = None, ...)`.
- Subprocess target deserializes the config, calls `config.make(infra)`, and
  serves. Real process isolation — no shared-memory thread fallback.
- Resolves the long-standing `# TODO: once BenchmarkConfig lands` in `server.py`
  (cite lines 80–86, 398–402).

## MODIFIED — Resource provisioning trigger
**Spec:** resource

Add one line noting that `BenchmarkConfig.make(infra)` is the framework-level
trigger for `infra.provision(resource)` (previously performed by benchmark
authors inside `_setup()`, now handled by the factory).

## MODIFIED — CLI entry-point group
**Spec:** cli

- `cube.benchmarks` entry-point group now advertises `BenchmarkConfig`
  subclasses (not `Benchmark` subclasses).
- `cube list`, `cube test` surface `BenchmarkConfig` classes; `cube test`
  accepts `--infra-config` to inject an `InfraConfig` into the debug flow.

---

See [proposal.md](proposal.md) for rationale and the migration plan.
