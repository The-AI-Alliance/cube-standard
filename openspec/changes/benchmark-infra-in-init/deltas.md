# Deltas — Thread `infra` through `Benchmark.__init__`

## MODIFIED — `openspec/specs/benchmark/spec.md`: `Benchmark` constructor

**Before**

```python
class Benchmark(ABC):
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config: BenchmarkConfig = config
        self._runtime_context: RuntimeContext = {}
```

**After**

```python
class Benchmark(ABC):
    def __init__(self, config: BenchmarkConfig, infra: InfraConfig | None = None) -> None:
        self.config: BenchmarkConfig = config
        self._infra: InfraConfig | None = infra
        self._runtime_context: RuntimeContext = {}
```

`infra` is forwarded by `BenchmarkConfig.make(infra)` and stashed as
`self._infra` so cubes that publish it into `runtime_context["infra"]` for
per-task container launches can do so from `_setup()` without overriding
`__init__` or `make`.

## MODIFIED — `openspec/specs/benchmark/spec.md`: `BenchmarkConfig.make`

The factory now constructs the runtime as
`type(self).benchmark_class(config=self, infra=infra)` instead of
`type(self).benchmark_class(config=self)`. Resource-provisioning behaviour and
the `infra is None` debug-log branch are unchanged.

## ADDED — `openspec/specs/benchmark/spec.md`: `_setup()` clarification

`_setup()` is the per-cube hook for publishing `self._infra` (or a derived
handle) into `self._runtime_context`. The base does **not** auto-publish
`runtime_context["infra"]` — each cube decides what shape to expose, since
some cubes publish the bare `InfraConfig` and others publish a launched
service handle.

## Migration impact

`BenchmarkConfig.make` now calls `benchmark_class(config=self, infra=infra)`.
Three subclass shapes exist; all need a follow-up cube-harness PR:

**1. Breaking — `__init__(self, config)` without an `infra` kwarg.**
First `make()` call raises `TypeError: __init__() got an unexpected keyword
argument 'infra'`. Fix: add `infra: InfraConfig | None = None` to the
signature and forward to `super().__init__()`.

- `cube-harness/cubes/miniwob` — must update before this PR can be consumed.
- `cube.benchmark.CompositeBenchmark` — updated in this change.

**2. Redundant — `__init__(self, config, infra=None)` + custom `make(infra)`
that re-implements the base provisioning loop.**
Now compatible with the base; the override is dead boilerplate that should be
deleted. Cubes that relied on `infra or LocalInfraConfig()` defaulting in
their `make()` override either need callers to pass `LocalInfraConfig()`
explicitly or keep a thin `make()` shim that resolves the default and calls
`super().make(infra=resolved)`.

- `cube-harness/cubes/osworld-cube`
- `cube-harness/cubes/swebench-live-cube`
- `cube-harness/cubes/swebench-verified-cube`
- `cube-harness/cubes/terminalbench-cube`
- `cube-harness/cubes/webarena-verified`
