# Deltas — SWE cube BenchmarkConfig migration

**Targets:** `openspec/specs/testing/spec.md`, `openspec/specs/benchmark/spec.md`

Applied when the cube-harness migration PR lands and the pattern is established.

---

## MODIFIED — `testing/spec.md`: `get_debug_benchmark()` must be a pure factory

The testing spec states that `get_debug_benchmark()` returns a `BenchmarkConfig` and that
the harness calls `config.install()` / `config.make()`. The **must-not** side of this
contract was implicit. Make it explicit:

> **MUST NOT** call `install()`, `setup()`, or any other lifecycle method inside
> `get_debug_benchmark()`. The function is a pure factory — it constructs and returns a
> (possibly subsetted) `BenchmarkConfig`. Side effects here break harnesses that call
> `install()` and `make()` themselves (double-install, or calling `setup()` on a
> `BenchmarkConfig` which has no such method).

---

## MODIFIED — `benchmark/spec.md`: add migration guide subsection

Add a "Migrating from pre-split cubes" subsection under the `BenchmarkConfig` section:

> ### Migrating a cube from the pre-split pattern
>
> Older cubes declared a single class that subclassed `Benchmark` and held both ClassVars
> (metadata registries) and user-configurable instance fields (`infra`, `oracle_mode`, …).
> The current spec requires a split into two classes. The mechanical steps:
>
> 1. **Rename** the existing class to `FooBenchmarkConfig(BenchmarkConfig[FooTaskMetadata])`.
>    Move all ClassVars, user fields, `install()`, `uninstall()`, and `get_task_configs()` here.
>    Add `benchmark_class: ClassVar[type[Benchmark]] = FooBenchmark`.
>
> 2. **Create** `FooBenchmark(Benchmark["FooBenchmarkConfig"])` with only `_setup()` and `close()`.
>    Access user fields via `self.config.<field>` (e.g. `self.config.infra`).
>
> 3. **Update `get_task_configs()`** to stamp `metadata=tm` on each emitted `TaskConfig`
>    (not `task_id=tm.id`). Workers deserialise the config and call `make()` without
>    importing `BenchmarkConfig`; stamping is what makes this possible.
>
> 4. **Update `task.py`**: replace `BenchmarkClass.task_metadata[self.task_id]` with
>    `self.metadata` (already stamped). Remove the benchmark import and the deprecated
>    `container_backend` parameter from `make()`.
>
> 5. **Update `debug.py`**: `get_debug_benchmark()` returns a `BenchmarkConfig`;
>    it must not call `install()` or `setup()`.
>
> 6. **Update `pyproject.toml`**: entry point must name `FooBenchmarkConfig`, not `FooBenchmark`.
>
> 7. **Update `__init__.py`**: export `FooBenchmarkConfig`.
