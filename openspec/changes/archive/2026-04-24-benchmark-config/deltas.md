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
- `container_backend: ContainerBackend | None = None` — kept on
  `BenchmarkConfig` for migration but marked `Field(deprecated=True)`.
  Slated for removal once in-tree benchmarks declare container needs via
  `resources`.
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

## ADDED — `CompositeBenchmarkConfig`, `CompositeBenchmark`
**Spec:** benchmark

`CompositeBenchmarkConfig(BenchmarkConfig)` holds `sub_bench_configs: list[BenchmarkConfig]`:
- Merged `task_metadata` prefixes keys with the sub-benchmark name
  (`"{sub.benchmark_metadata.name}/{task_id}"`).
- Duplicate sub-benchmark names raise at construction.
- `get_task_configs()` emits each sub-config's TaskConfigs unchanged in
  type (preserving subclass-specific fields and embedded `metadata`) with
  only `task_id` (prefixed) and `sub_bench_name` (tag) updated. No wrapper
  type.
- `make(infra)` calls `sub.make(infra)` for each sub_config and returns a
  `CompositeBenchmark` holding `sub_benchmarks: dict[str, Benchmark]`.

`CompositeBenchmark(Benchmark)`:
- `spawn(tc)` reads `tc.sub_bench_name` and routes by calling
  `tc.make(runtime_context=sub_bench._runtime_context, ...)` directly.
  Rejects configs with `sub_bench_name=None` or unknown names.
- `close()` closes every sub-benchmark.

## MODIFIED — `TaskConfig` shape
**Spec:** task

`TaskConfig` now carries metadata directly:
- New field: `metadata: TaskMetadata` — stamped onto each emitted config by
  `BenchmarkConfig.get_task_configs()`. `make()` uses `self.metadata`
  directly; workers never import the owning BenchmarkConfig for lookups.
- New field: `sub_bench_name: str | None = None` — routing hint set by
  `CompositeBenchmarkConfig.get_task_configs()`. Standalone benchmarks
  leave it None.

This is the single most important invariant change in the layer: the
serialization boundary is now self-describing. Benchmarks with heavy
install-time data override `get_task_configs()` to merge
`load_task_execution_info(task_id)` into `metadata.extra_info` at emit
time (on the driver), so workers never touch disk.

## MODIFIED — Debug-flow infra injection (closes #96)
**Spec:** testing

- Module protocol: `get_debug_benchmark() -> BenchmarkConfig` (zero args; infra
  is supplied separately to the suite, not threaded through the factory).
- `run_debug_suite(benchmark_name, module, *, max_steps, workers, infra=None)`
  calls `module.get_debug_benchmark()` then `config.make(infra)` under
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

## MODIFIED — `AbstractSeedGenerator` is a `TypedBaseModel`
**Spec:** benchmark

`AbstractSeedGenerator` was changed from a plain Pydantic `BaseModel` to a
`TypedBaseModel`. This is a **silent breaking change** for cubes that
declare a custom seed generator: subclasses authored against the old base
will deserialize on workers without the `_type` discriminator and fail
with a `_type`-not-found error. Migration: re-import nothing — the symbol
is the same — but ensure any custom subclass survives a JSON round-trip
through `TypedBaseModel.model_validate(json.loads(json.dumps(seed_gen.model_dump(mode="json"))))`.

---

See [proposal.md](proposal.md) for rationale and the migration plan.
