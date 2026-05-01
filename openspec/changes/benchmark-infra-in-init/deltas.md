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
