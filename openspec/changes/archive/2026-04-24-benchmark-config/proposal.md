# BenchmarkConfig split and CompositeBenchmark

**Status:** Accepted — landing incrementally on `feat/benchmark-config`
**Date:** 2026-04-22
**Scope:** `cube.benchmark`, `cube.task` (docs only), `cube.testing`, `cube.server`, `cube.cli`, `cube.introspect`, `cube._template`
**Related:** cube-standard#111, cube-standard#96, cube-harness RFC `docs/rfc-benchmark-config-and-scaling.md`

---

## Problem

`Benchmark` today mixes serializable configuration (metadata, resources, default
tool config) with runtime state (`_runtime_context`, container handles created in
`_setup()`). Three concrete pains fall out of that:

1. **`subset_from_list` is fragile.** It uses `model_copy(deep=True)` +
   `__pydantic_private__` reset + `object.__setattr__` to shadow ClassVars, because
   the live instance holds resources that cannot be deep-copied safely.
2. **A configured benchmark cannot be serialized.** A WorkArena + MiniWob + OSWorld
   suite cannot be persisted to JSON, shared, or dispatched across Ray workers.
3. **`cube.testing` has no way to inject an `InfraConfig`.** `get_debug_benchmark()`
   is zero-argument, forcing benchmark packages to carry ad-hoc env-var parsing
   (OSWorld today).

## Solution

Split `Benchmark` into two layers, mirroring the existing `TaskConfig` / `Task`
pattern:

```
BenchmarkConfig   →  make(infra)  →  Benchmark
  serializable                       not serializable
  subset_from_list()                 _setup() / close()
  named_subset()                     spawn()
  install()                          _runtime_context
```

`BenchmarkConfig` is a pure Pydantic model. `Benchmark` is a plain Python class
holding live handles only. `config.make(infra)` provisions required resources
(idempotent), calls `setup()`, and returns a ready `Benchmark` — there is no
state where a `Benchmark` exists uninitialized.

### Subsets without hacks

`BenchmarkConfig` keeps `task_metadata: ClassVar[dict[str, TaskMetadata]]` as the
authoritative registry (populated at class definition time from files or direct
declaration — `install()` never touches it). Subsets are tracked with a new
instance field `task_ids: list[str] | None = None` (None = all tasks).
`subset_from_list` becomes `self.model_copy(update={"task_ids": [...]})` — no
deep copy, no private-attr reset, no setattr hacks. `TaskConfig.make()` keeps
using `OwnerBenchmarkConfig.task_metadata[self.task_id]` (class-level lookup)
and works identically under subsetting, since subsets only narrow which ids are
emitted — every emitted id is still a valid key in the ClassVar.

### Composition

`CompositeBenchmarkConfig(sub_bench_configs=[...])` holds a list of
`BenchmarkConfig`s. Its merged `task_metadata` prefixes keys with the
sub-benchmark name (`"workarena-cube/<id>"`). Emitted `TaskConfig`s are wrapped
in `CompositeTaskConfig(sub_name, inner)` — the wrapper is itself a `TaskConfig`
whose `make()` delegates to `inner.make()`, so every generic path that takes a
`TaskConfig` still works. `CompositeBenchmark.spawn(wrapper)` routes by
`sub_name`. Duplicate sub-benchmark names raise at construction.

Because `CompositeBenchmarkConfig` is itself a `BenchmarkConfig`, composites
compose (composite-of-composite works out of the box) and the whole suite is
serializable.

### Infra injection

`make(infra: InfraConfig | None = None)` is the single point where a config meets
an infra. Infra is never stored on a `BenchmarkConfig`. `cube.testing` and
`cube.cli` pass `infra` through to `make()`, closing #96. Benchmarks that do not
need infra (arithmetic, miniwob) accept `infra=None` and ignore it.

### `Benchmark` is a plain class

`Benchmark` holds only live OS state. Making it non-Pydantic removes
`arbitrary_types_allowed`, `model_post_init`, and the temptation to serialize it.
It stores a back-reference to its `config` plus `_runtime_context`, exposes
`_setup()` / `close()` / `spawn()`, and acts as a context manager
(`__enter__`/`__exit__`).

## Migration

**Cube-standard (this PR):** complete split, `CompositeBenchmarkConfig`, debug
flow update (`get_debug_benchmark(infra)`), server subprocess via
`make_benchmark_rpc_server(config)`, entry-point update, template update.

**Per-cube (parallel follow-up PRs in cube-harness):** each cube renames
`XxxBenchmark` → `XxxBenchmarkConfig`, adds `XxxBenchmark(benchmark_class)`,
moves `_setup`/`close` to the runtime class, updates `debug.py` to take `infra`.

**Deferred:** `BenchmarkPool` stays a TODO. Once per-cube migration completes,
it can be added as a thin wrapper (see RFC §3).

## Out of scope

- `BenchmarkPool` and Ray Actor load-balancing.
- Backwards-compat shims: per the user instruction, cubes will migrate in
  parallel. No deprecation period for the pre-split API.

See [deltas.md](deltas.md) for the spec changes this introduces.
