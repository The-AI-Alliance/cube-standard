# SWE cube migration: BenchmarkConfig split + generic types + debug contract

**Status:** Proposed
**Date:** 2026-04-30
**Scope:** cube-harness — `swebench-verified-cube`, `swebench-live-cube`, `terminalbench-cube`
**Targets:** cube-harness PR (branch `feat/benchmark-config`); builds on cube-standard `nico_fix` (PR #124)
**Related:** archived `2026-04-24-benchmark-config`, archived `2026-04-27-typed-task-execution-info`, `task-config-generic` (nico_fix PR #124)

---

## Context

Three changes landed in cube-standard that together define the new cube contract:

| Change | What it introduced |
|--------|-------------------|
| PR #118 — `feat/benchmark-config` | `BenchmarkConfig` / `Benchmark` split; `BenchmarkConfig` is the serialisable registry, entry-point target, and owner of `install()`, `get_task_configs()`. `Benchmark` is pure runtime. |
| PR #121 — `feat/typed-task-execution-info` | Typed `TaskExecutionInfo` slot replaces the `extra_info: dict` bag; `task_id` promoted to `TaskConfig.metadata` field (stamped by `get_task_configs()`). |
| PR #124 — `nico_fix` (open) | `Task`, `TaskConfig`, `BenchmarkConfig`, `Benchmark` made generic via PEP 695 class-scoped type parameters. Cubes opt in with `class FooTaskConfig(TaskConfig[FooTaskMetadata]):` etc.; removes all `# type: ignore` re-annotation overrides. |

The three SWE cubes in cube-harness pre-date all three of these changes and violate several
invariants of the current spec. This proposal documents the complete migration.

---

## Problems

All three cubes share the same set of violations:

### 1. `Benchmark` class holds ClassVars and user fields (should be `BenchmarkConfig`)

```python
# current (wrong)
class SWEBenchVerifiedBenchmark(Benchmark):
    benchmark_metadata: ClassVar[BenchmarkMetadata] = ...
    task_metadata: ClassVar[dict[str, SWEBenchVerifiedTaskMetadata]]
    task_config_class: ClassVar[type[TaskConfig]] = SWEBenchVerifiedTaskConfig
    include_hints: bool = False
    oracle_mode: bool = False
    infra: InfraConfig = Field(default_factory=LocalInfraConfig)
```

The spec requires ClassVars and serialisable user fields on `BenchmarkConfig`, not `Benchmark`.
`Benchmark` must be pure runtime state.

### 2. `install()`, `get_task_configs()` on `Benchmark` (should be `BenchmarkConfig`)

Same class hosts lifecycle methods that the spec assigns to `BenchmarkConfig`.

### 3. `get_task_configs()` emits `task_id=` instead of `metadata=`

```python
# current (wrong)
yield SWEBenchVerifiedTaskConfig(task_id=tm.id, ...)

# required
yield SWEBenchVerifiedTaskConfig(metadata=tm, ...)
```

`metadata` must be stamped at driver time so workers can deserialise configs and call
`make()` without importing `BenchmarkConfig`.

### 4. `task.py` circular import via `BenchmarkClass.task_metadata`

`TaskConfig.make()` imports the benchmark class to look up task metadata:

```python
from swebench_verified_cube.benchmark import SWEBenchVerifiedBenchmark
metadata = SWEBenchVerifiedBenchmark.task_metadata[self.task_id]
```

This is the exact anti-pattern the `metadata` stamping was introduced to eliminate.
After metadata stamping, `metadata = self.metadata` and the import disappears.

### 5. Entry point names `Benchmark` class (must name `BenchmarkConfig`)

```toml
# current (wrong)
swebench-verified-cube = "swebench_verified_cube.benchmark:SWEBenchVerifiedBenchmark"

# required
swebench-verified-cube = "swebench_verified_cube.benchmark:SWEBenchVerifiedBenchmarkConfig"
```

### 6. `debug.py` calls `install()` and `setup()` inside `get_debug_benchmark()`

```python
# current (wrong)
def get_debug_benchmark(infra=None) -> Benchmark:
    bench = SWEBenchVerifiedBenchmark(infra=..., oracle_mode=True)
    bench.install()   # violates: harness owns install()
    bench.setup()     # violates: harness owns setup(); also impossible after migration
                      # since BenchmarkConfig has no .setup()
    return bench.subset_from_list(...)
```

The testing spec is clear: `get_debug_benchmark()` returns a `BenchmarkConfig`; the harness
calls `config.install()` then `config.make(infra)`. The function must be a pure factory.

### 7. `debug_harness.py` calls `bench.setup()` after `get_debug_benchmark()`

```python
# current (wrong) — integration-tests/cube_integration_tests/debug_harness.py
bench = cube_debug_module.get_debug_benchmark(infra=infra)
bench.setup()   # will fail: BenchmarkConfig has no .setup()
```

This custom harness bypasses `run_debug_suite` and must be updated alongside the cube
`debug.py` files.

### 8. `task.py` keeps deprecated `container_backend` parameter

`swebench-live` and `terminalbench` still pass `container_backend` to the `Task` constructor.
`swebench-verified` still declares it on `make()` without using it. All three should remove it.

### 9. No unit tests

None of the three cubes has a `tests/` directory.

---

## Migration pattern

This section documents the canonical per-file migration recipe. Each cube applies it
with name substitutions only.

### `benchmark.py` — split into Config + Benchmark

```python
# ── Runtime pair — only _setup() and close() ──────────────────────────────

class SWEBenchVerifiedBenchmark(Benchmark["SWEBenchVerifiedBenchmarkConfig"]):

    def _setup(self) -> None:
        self.config.infra.cleanup_stale()
        self._runtime_context["infra"] = self.config.infra

    def close(self) -> None:
        pass


# ── Serialisable registry — everything else ────────────────────────────────

class SWEBenchVerifiedBenchmarkConfig(BenchmarkConfig[SWEBenchVerifiedTaskMetadata]):

    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(...)
    task_metadata: ClassVar[dict[str, SWEBenchVerifiedTaskMetadata]]
    task_config_class: ClassVar[type[TaskConfig]] = SWEBenchVerifiedTaskConfig
    benchmark_class: ClassVar[type[Benchmark]] = SWEBenchVerifiedBenchmark

    # User-configurable fields (serialisable)
    include_hints: bool = False
    oracle_mode: bool = False
    infra: InfraConfig = Field(default_factory=LocalInfraConfig)

    @classmethod
    def install(cls) -> None: ...

    @classmethod
    def uninstall(cls) -> None: ...

    def get_task_configs(self) -> Generator[TaskConfig, None, None]:
        for tm in self.tasks().values():
            yield SWEBenchVerifiedTaskConfig(
                metadata=tm,                       # ← stamp metadata
                tool_config=self.tool_config,
                seed=None,
                include_hints=self.include_hints,
                oracle_mode=self.oracle_mode,
            )
```

Note: `infra` is now accessed via `self.config.infra` inside `_setup()` because `Benchmark`
carries only a `config` back-reference — it has no fields of its own.

### `task.py` — drop circular import and deprecated param

```python
class SWEBenchVerifiedTaskConfig(TaskConfig[SWEBenchVerifiedTaskMetadata]):

    include_hints: bool = False
    oracle_mode: bool = False

    def make(self, runtime_context: RuntimeContext | None = None) -> SWEBenchVerifiedTask:
        # container_backend param gone — infra-only path
        if runtime_context is None or "infra" not in runtime_context:
            raise ValueError("SWEBenchVerifiedTaskConfig.make() requires runtime_context['infra']")

        # metadata already stamped — no benchmark import needed
        exec_info = self.load_task_execution_info(self.task_id)
        metadata = self.metadata.model_copy(
            update={"extra_info": {**exec_info,
                                   "include_hints": self.include_hints,
                                   "oracle_mode": self.oracle_mode}}
        )
        return SWEBenchVerifiedTask(
            metadata=metadata,
            tool_config=self.tool_config or SWEBenchToolConfig(),
            runtime_context=runtime_context,
        )
```

### `debug.py` — pure factory, no side effects

```python
def get_debug_benchmark(infra: InfraConfig | None = None) -> SWEBenchVerifiedBenchmarkConfig:
    return SWEBenchVerifiedBenchmarkConfig(
        infra=infra or LocalInfraConfig(),
        oracle_mode=True,
    ).subset_from_list(list(_TASK_ACTIONS))
```

No `install()`. No `setup()`. The harness owns both.

### `debug_harness.py` — call `install()` + `make()` instead of `setup()`

```python
# integration-tests/cube_integration_tests/debug_harness.py

config = cube_debug_module.get_debug_benchmark(infra=infra)
config.install()
benchmark = config.make()
task_configs = [tc for tc in config.get_task_configs() if tc.task_id == task_id]
tc = task_configs[0]
task = tc.make(runtime_context=benchmark._runtime_context)
```

### `pyproject.toml` — entry point

```toml
swebench-verified-cube = "swebench_verified_cube.benchmark:SWEBenchVerifiedBenchmarkConfig"
```

### `__init__.py` — export Config class

```python
from swebench_verified_cube.benchmark import SWEBenchVerifiedBenchmark, SWEBenchVerifiedBenchmarkConfig
```

---

## Per-cube differences

| Aspect | swebench-verified | swebench-live | terminalbench |
|--------|:-----------------:|:-------------:|:-------------:|
| Generic param | `BenchmarkConfig[SWEBenchVerifiedTaskMetadata]` | `BenchmarkConfig[SWEBenchLiveTaskMetadata]` | `BenchmarkConfig[TerminalBenchTaskMetadata]` |
| User fields on Config | `include_hints`, `oracle_mode`, `infra` | `include_hints`, `oracle_mode`, `infra` | `oracle_mode`, `infra` |
| `container_backend` in `make()` | declared but unused — remove signature | fallback path + passed to Task — remove both | fallback path + passed to Task — remove both |
| `ContainerBackend` import in `task.py` | not imported | imported, becomes unused — remove | imported, becomes unused — remove |

---

## Files changed

Per cube (×3):

| File | Change |
|------|--------|
| `benchmark.py` | Split into `BenchmarkConfig` + `Benchmark`; use generic types |
| `task.py` | Drop `container_backend`; remove circular import; use `self.metadata` |
| `debug.py` | Pure factory; drop `install()` and `setup()` calls |
| `__init__.py` | Add `BenchmarkConfig` export |
| `pyproject.toml` | Update entry point to `BenchmarkConfig` class |
| `tests/test_benchmark.py` (new) | Unit tests — no Docker required |

Shared:

| File | Change |
|------|--------|
| `recipes/swe_agent_recipe.py` | Update 3 instantiation sites to `BenchmarkConfig` |
| `integration-tests/cube_integration_tests/debug_harness.py` | Replace `bench.setup()` with `config.install()` + `config.make()` |

---

## Unit test plan (new `tests/` per cube)

All tests are Docker-free and network-free. Each cube gets the same five tests:

| Test | What it validates |
|------|-------------------|
| `test_config_roundtrip` | `model_dump()` → `model_validate()` round-trips cleanly |
| `test_task_metadata_loaded` | ClassVar populated at import; correct task count |
| `test_get_task_configs_stamps_metadata` | Every emitted `TaskConfig.metadata.id` matches iteration order |
| `test_subset_from_list` | Scopes config to exactly the requested task IDs |
| `test_debug_benchmark_type` | `get_debug_benchmark()` returns a `BenchmarkConfig` instance |

---

## Commit plan (cube-harness PR)

```
feat(swebench-verified): migrate to BenchmarkConfig split + generic types
feat(swebench-live): migrate to BenchmarkConfig split + generic types
feat(terminalbench): migrate to BenchmarkConfig split + generic types
test: add unit tests for swebench-verified, swebench-live, terminalbench
fix(recipe): update swe_agent_recipe to BenchmarkConfig classes
fix(debug-harness): call install()+make() instead of setup()
```

See [deltas.md](deltas.md) for the two spec clarifications this surfaces.
