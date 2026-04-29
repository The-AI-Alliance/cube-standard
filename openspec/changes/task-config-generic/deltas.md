# Deltas — `TaskConfig`, `BenchmarkConfig`, `Benchmark` generic

**Targets:** `openspec/specs/task/spec.md`, `openspec/specs/benchmark/spec.md`

Applied when the change lands.

## Naming convention

TypeVars use long-form prefixes to avoid collisions across layers:

- `TTMetadata` — `TaskMetadata`-bound (declared in `cube.task`,
  re-imported by `cube.benchmark`). The double-T avoids collision with
  a future `TBenchMetadata`.
- `TBenchConfig` — `BenchmarkConfig`-bound (declared in
  `cube.benchmark`). The `Bench` qualifier avoids collision with a
  future `TaskConfig` / `ToolConfig` TypeVar.

New TypeVars introduced by future changes should follow the same pattern.

## ADDED — `TTMetadata` TypeVar

**Spec:** task

New module-level export in `cube.task`:

```python
TTMetadata = TypeVar("TTMetadata", bound=TaskMetadata)
```

Bound to `TaskMetadata` so any parametrisation must subclass it. Exported so
cube authors and downstream layers can reuse a single symbol — `cube.benchmark`
imports and reuses it for `BenchmarkConfig[TTMetadata]`.

## MODIFIED — `TaskConfig` is generic

**Spec:** task

Signature changes from:

```python
class TaskConfig(TypedBaseModel, ABC):
    metadata: SerializeAsAny[TaskMetadata]
    ...
```

to:

```python
class TaskConfig(TypedBaseModel, ABC, Generic[TTMetadata]):
    metadata: SerializeAsAny[TTMetadata]
    ...
```

`metadata` is an instance field (not a `ClassVar`) and Pydantic generic models
substitute the TypeVar at class-finalisation time, so this narrows soundly
under parametrisation — no PEP 526 / variance issues.

**Subclassing forms:**

- *Unparametrised* (existing cubes, no migration required):

  ```python
  class FooTaskConfig(TaskConfig): ...
  ```

  `metadata` is statically typed as `TaskMetadata` — same as before.

- *Parametrised* (opt-in for narrower typing):

  ```python
  class FooTaskConfig(TaskConfig[FooTaskMetadata]): ...
  ```

  `metadata` is statically typed as `FooTaskMetadata`. No re-annotation of
  `metadata` on the subclass is needed; doing so is forbidden because
  Pydantic-invariant container semantics make it unsound (a holder of the
  unparametrised parent could legally assign a plain `TaskMetadata`).

**Runtime behaviour is unchanged** for both forms.

## MODIFIED — `BenchmarkConfig` is generic; `tasks()` returns `Mapping[str, TTMetadata]`

**Spec:** benchmark

`cube.benchmark` imports `TTMetadata` from `cube.task` and uses it on
`BenchmarkConfig`. The narrowing is method-level only, not field-level:

```python
class BenchmarkConfig(TypedBaseModel, ABC, Generic[TTMetadata]):
    benchmark_metadata: ClassVar[BenchmarkMetadata]
    task_metadata: ClassVar[dict[str, TaskMetadata]]   # NOT narrowed (see below)
    task_config_class: ClassVar[type[TaskConfig]]
    benchmark_class: ClassVar[type["Benchmark"]]
    ...

    def tasks(self) -> Mapping[str, TTMetadata]: ...    # narrowed (covariant)
```

**Why `tasks()` returns `Mapping` instead of `dict`.** `Mapping` is the
read-only ABC and is *covariant* in its value type, so subclasses
parametrised as `BenchmarkConfig[FooTaskMetadata]` get a properly-narrowed
`Mapping[str, FooTaskMetadata]` view at every read site. `dict` is
invariant and could not be narrowed soundly. The runtime return value is
still a `dict`; only the static contract changes.

**Why `task_metadata: ClassVar` is *not* narrowed.** Two independent reasons:

1. PEP 526 (and pyright/mypy in strict mode) forbid TypeVars inside
   `ClassVar` — a `ClassVar` is shared across all generic specialisations,
   but TypeVars vary per specialisation, so combining them is incoherent.
2. Even setting PEP 526 aside, `dict` is invariant in its value type, so
   `dict[str, FooTaskMetadata]` is not a subtype of `dict[str,
   TaskMetadata]` — narrowing the registry directly would be unsound for
   the same reason the original `task_metadata: ClassVar[dict[str,
   FooTaskMetadata]]` overrides were unsound.

Direct reads of `cfg.task_metadata` therefore see `dict[str, TaskMetadata]`
regardless of parametrisation. Reads should go through `cfg.tasks()` to get
the narrowed view.

**Subclassing forms:**

- *Unparametrised* (existing cubes, no migration required):

  ```python
  class FooBenchmarkConfig(BenchmarkConfig): ...
  ```

  `cfg.tasks()` returns `Mapping[str, TaskMetadata]` — same as today
  (the previous `dict[str, TaskMetadata]` was a stricter return type that
  no caller depended on).

- *Parametrised* (opt-in for narrower typing):

  ```python
  class FooBenchmarkConfig(BenchmarkConfig[FooTaskMetadata]): ...
  ```

  `cfg.tasks()` returns `Mapping[str, FooTaskMetadata]`. Iteration,
  membership tests, and value access give autocomplete on
  `FooTaskMetadata` fields. No re-annotation on the subclass is needed.

**`task_config_class` / `benchmark_class` are not separately parametrised.**
`type[Sub]` is covariantly assignable to `type[Base]`, so existing
declarations like `task_config_class: ClassVar = WorkArenaTaskConfig` against
the parent's `ClassVar[type[TaskConfig]]` are already sound without an
override.

**`CompositeBenchmarkConfig`** subclasses `BenchmarkConfig[TaskMetadata]`
(the heterogeneous case): a composite's merged `task_metadata` mixes
metadata types across sub-benchmarks, so the upper bound is the only
correct narrowing.

**Runtime behaviour is unchanged.** `__init_subclass__` validation continues
to enforce ClassVar wiring on user classes; Pydantic's parametrised
intermediates (`BenchmarkConfig[FooTaskMetadata]`, synthesised internally
when a user writes `class FooBenchmarkConfig(BenchmarkConfig[FooTaskMetadata]):`)
are skipped by an early-return guard keyed off the intermediate's name
(which contains `[`, never present in user class names). File loaders
(`task_metadata_from_json`, `task_metadata_from_csv`) continue to return
`dict[str, TaskMetadata]`; runtime dispatch via the `_type` discriminator
already produces the right subclass instances.

## MODIFIED — `subset_from_list` accepts `Sequence` instead of `list`

**Spec:** benchmark

Signature changes from:

```python
def subset_from_list(
    self,
    tasks: list[str] | list[TaskMetadata],
    ...
) -> Self: ...
```

to:

```python
def subset_from_list(
    self,
    tasks: Sequence[str] | Sequence[TaskMetadata],
    ...
) -> Self: ...
```

`Sequence` is covariant in its element type, so callers parametrised as
`BenchmarkConfig[FooTaskMetadata]` can pass a `list[FooTaskMetadata]` (e.g.
the result of iterating `cfg.tasks().values()` and filtering) without a
cast. The body of `subset_from_list` only iterates and `isinstance`-checks
the input — never mutates it — so the widening is sound.

This is a non-breaking change: every `list[…]` accepted before is still
accepted (lists are sequences).

## ADDED — `TBenchConfig` TypeVar

**Spec:** benchmark

New module-level export in `cube.benchmark`:

```python
TBenchConfig = TypeVar("TBenchConfig", bound=BenchmarkConfig)
```

Bound to `BenchmarkConfig` so any parametrisation must subclass it. Used as
the parameter for `Benchmark[TBenchConfig]`.

## MODIFIED — `Benchmark` is generic

**Spec:** benchmark

Signature changes from:

```python
class Benchmark(ABC):
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config: BenchmarkConfig = config
        self._runtime_context: RuntimeContext = {}
```

to:

```python
class Benchmark(ABC, Generic[TBenchConfig]):
    def __init__(self, config: TBenchConfig) -> None:
        self.config: TBenchConfig = config
        self._runtime_context: RuntimeContext = {}
```

`Benchmark` is a plain ABC, not a Pydantic model, so the parametrisation
uses standard `typing.Generic` — no Pydantic generic-intermediate handling
required.

**Subclassing forms:**

- *Unparametrised* (existing cubes, no migration required):

  ```python
  class FooBenchmark(Benchmark): ...
  ```

  `self.config` is statically typed as `BenchmarkConfig` — same as before.

- *Parametrised* (opt-in for narrower typing):

  ```python
  class FooBenchmark(Benchmark[FooBenchmarkConfig]): ...
  ```

  `self.config` is statically typed as `FooBenchmarkConfig`. No
  re-annotation is needed; the previous pattern of
  `config: FooBenchmarkConfig  # type: ignore` (or
  `self.config: FooBenchmarkConfig = config  # type: ignore[assignment]`
  inside `__init__`) becomes unnecessary and should be deleted.

**`CompositeBenchmark`** subclasses `Benchmark["CompositeBenchmarkConfig"]`
(forward-ref string, since `CompositeBenchmarkConfig` is defined later in
the same module). The previous redundant `self.config: ... = config  #
type: ignore[assignment]` re-annotation is dropped — parametrisation gives
the right static type without it.

**`benchmark_class: ClassVar[type[Benchmark]]` on `BenchmarkConfig` is *not*
parametrised over a separate TypeVar.** `type[Sub]` is covariantly assignable
to `type[Base]`, so existing declarations like
`benchmark_class: ClassVar = WorkArenaBenchmark` against the parent's
`ClassVar[type[Benchmark]]` are already sound without an override.

**Runtime behaviour is unchanged.** `BenchmarkConfig.make()` still calls
`type(self).benchmark_class(config=self)` exactly as before.

## MODIFIED — Gotcha: do not re-annotate narrowed types on subclasses

**Spec:** task, benchmark

Add to the existing "Gotchas" sections:

> Cubes that need narrower static types use the parametrised forms
> `class FooTaskConfig(TaskConfig[FooTaskMetadata]):`,
> `class FooBenchmarkConfig(BenchmarkConfig[FooTaskMetadata]):`, and
> `class FooBenchmark(Benchmark[FooBenchmarkConfig]):` rather
> than re-annotating fields on the subclass. Re-annotations like
> `metadata: SerializeAsAny[FooTaskMetadata]`,
> `task_metadata: ClassVar[dict[str, FooTaskMetadata]]`, or
> `config: FooBenchmarkConfig` are unsound under
> invariant-field / invariant-container / invariant-attribute semantics
> and type checkers reject them; the parametrised forms express the
> intent correctly without an override. For `BenchmarkConfig`,
> field-level narrowing is replaced by read-site narrowing via
> `cfg.tasks() -> Mapping[str, TTMetadata]` (covariant); the underlying
> `task_metadata: ClassVar` registry stays typed as the base
> `dict[str, TaskMetadata]`.
