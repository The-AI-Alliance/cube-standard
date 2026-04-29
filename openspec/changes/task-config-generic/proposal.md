# Make `Task`, `TaskConfig`, `BenchmarkConfig`, and `Benchmark` generic over their narrowable types

**Status:** Proposed
**Date:** 2026-04-29
**Scope:** `cube.task`, `cube.benchmark`
**Targets:** `nico_fix` branch in cube-standard PR #124; unblocks per-cube migrations in cube-harness PRs #322 / #323 / #324
**Related:** none active. Builds on archived `2026-04-24-benchmark-config` and `2026-04-27-typed-task-execution-info`.

---

## Problem

Today, `TaskConfig.metadata` is annotated as `SerializeAsAny[TaskMetadata]` and
`BenchmarkConfig.task_metadata` as `ClassVar[dict[str, TaskMetadata]]`. Cubes
that subclass `TaskMetadata` with extra typed fields (e.g.
`WorkArenaTaskMetadata`) want their `TaskConfig.make()` body and their
benchmark's `tasks()` view to surface the narrower subclass with autocomplete
and static type checking. The historical workaround was to re-declare the
field on the subclass with a narrowed type:

```python
class WorkArenaTaskConfig(TaskConfig):
    metadata: SerializeAsAny[WorkArenaTaskMetadata]  # type: ignore

class WorkArenaBenchmarkConfig(BenchmarkConfig):
    task_metadata: ClassVar[dict[str, WorkArenaTaskMetadata]]  # type: ignore
```

These overrides are **unsound under invariant container semantics**:

- `metadata: SerializeAsAny[Sub]` — a holder of `TaskConfig` could legally
  assign a plain `TaskMetadata`, breaking the subclass's narrowed contract.
- `task_metadata: dict[str, Sub]` — same story; `dict` is invariant in its
  value type.

Type checkers correctly reject both, and authors silenced them with
`# type: ignore`. The codebase accumulated several such overrides; a
recent code-review on PR #124 flagged the pattern as the central typing
smell of the project.

## Solution

Four generic-class promotions using PEP 695 class-scoped type parameters
(Python ≥3.12), each addressing the same shape at a different layer:

```python
# cube.task
class Task[TTMetadata: TaskMetadata](TypedBaseModel, ABC):
    metadata: SerializeAsAny[TTMetadata]
    ...

class TaskConfig[TTMetadata: TaskMetadata](ABC, TypedBaseModel):
    metadata: SerializeAsAny[TTMetadata] = Field(...)
    ...

# cube.benchmark
class BenchmarkConfig[TTMetadata: TaskMetadata](TypedBaseModel, ABC):
    task_metadata: ClassVar[dict[str, TaskMetadata]]   # NOT narrowed (see below)
    ...

    def tasks(self) -> Mapping[str, TTMetadata]: ...   # narrowed (covariant)

class Benchmark[TBenchConfig: BenchmarkConfig](ABC):
    def __init__(self, config: TBenchConfig) -> None:
        self.config: TBenchConfig = config
        ...
```

Cubes opt in to the narrowed form at each layer independently:

```python
class WorkArenaTask(Task[WorkArenaTaskMetadata]):
    # self.metadata is statically WorkArenaTaskMetadata — no cast, no override.
    def reset(self) -> tuple[Observation, dict]: ...
    def evaluate(self, obs=None) -> tuple[float, dict]: ...

class WorkArenaTaskConfig(TaskConfig[WorkArenaTaskMetadata]):
    def make(self, runtime_context=None, container_backend=None) -> WorkArenaTask:
        # self.metadata is statically WorkArenaTaskMetadata — no cast, no override.
        ...

class WorkArenaBenchmarkConfig(BenchmarkConfig[WorkArenaTaskMetadata]):
    # cfg.tasks() returns Mapping[str, WorkArenaTaskMetadata].
    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(...)
    task_config_class: ClassVar = WorkArenaTaskConfig
    benchmark_class: ClassVar = WorkArenaBenchmark

class WorkArenaBenchmark(Benchmark[WorkArenaBenchmarkConfig]):
    # self.config is statically WorkArenaBenchmarkConfig — no override.
    def _setup(self) -> None: ...
    def close(self) -> None: ...
```

**Naming convention.** Type parameters use a long-form prefix to avoid
collisions across layers as more generic types are introduced:

- `TTMetadata` — TaskMetadata-bound (not just `TMetadata`, which could
  collide with a future `TBenchMetadata`).
- `TBenchConfig` — BenchmarkConfig-bound (not just `TConfig`, which
  could collide with future `TaskConfig` / `ToolConfig` type parameters).

`task_config_class` and `benchmark_class` are *not* themselves
parametrised over separate type parameters. `type[Sub]` is covariantly assignable
to `type[Base]`, so subclasses can already assign their own narrower types
(e.g. `task_config_class: ClassVar = WorkArenaTaskConfig` against a parent
declared as `ClassVar[type[TaskConfig]]`) without an override or
`# type: ignore`. Adding type parameters there would be cosmetic.

`Benchmark` is a plain ABC (not a Pydantic model), so the parametrisation
is simpler — no Pydantic generic-intermediate handling needed.

## Backwards compatibility

**No runtime breakage.** Existing cubes written as
`class FooTaskConfig(TaskConfig):` and
`class FooBenchmarkConfig(BenchmarkConfig):` keep working unchanged:

- Pydantic supports unparametrised generic-model subclasses; `metadata` and
  `task_metadata` resolve to the type parameter's `bound` (`TaskMetadata`),
  identical to today.
- `__init_subclass__` validation in `BenchmarkConfig` is `isinstance(...)` /
  `issubclass(...)`-based — generic-agnostic.
- `SerializeAsAny[TTMetadata]` round-trips polymorphic subclass instances the
  same way `SerializeAsAny[TaskMetadata]` does, because Pydantic resolves
  the type parameter at class-finalisation time.
- `task_metadata_from_json` / `task_metadata_from_csv` continue to return
  `dict[str, TaskMetadata]` at the static type level; runtime dispatch via
  `TypedBaseModel`'s `_type` discriminator already produces the right
  subclass instances. Composite benchmarks (which mix metadata types
  across sub-benchmarks) parametrise as `BenchmarkConfig[TaskMetadata]`.

**Optional migration** for cubes that want narrower types: parametrise
each layer's subclass declaration:

- `class FooTask(Task):` → `class FooTask(Task[FooTaskMetadata]):`
- `class FooTaskConfig(TaskConfig):` → `class FooTaskConfig(TaskConfig[FooTaskMetadata]):`
- `class FooBenchmarkConfig(BenchmarkConfig):` → `class FooBenchmarkConfig(BenchmarkConfig[FooTaskMetadata]):`
- `class FooBenchmark(Benchmark):` → `class FooBenchmark(Benchmark[FooBenchmarkConfig]):`

Remove any associated re-annotations
(`metadata: SerializeAsAny[FooTaskMetadata]` on `FooTask` or `FooTaskConfig`,
`task_metadata: ClassVar[dict[str, FooTaskMetadata]]`,
`config: FooBenchmarkConfig`) along with their `# type: ignore`
comments. Each cube migrates independently in its own PR.

## Non-goals

- Forcing existing cubes to adopt the parametrised form. The
  unparametrised form remains a first-class citizen.
- Adding separate type parameters for `TaskConfig` / `Benchmark` subclass narrowing
  on `task_config_class` / `benchmark_class`. These already work via
  covariant `type[…]` semantics.
- Changing the registry shape (`task_metadata`, `task_config_class`,
  `benchmark_class`), the file-loader contract, or composite routing.

## Migration

**This PR (cube-standard `nico_fix`):**

- `cube.task`: change `class Task(TypedBaseModel, ABC)` →
  `class Task[TTMetadata: TaskMetadata](TypedBaseModel, ABC)`;
  change `metadata: SerializeAsAny[TaskMetadata]` →
  `metadata: SerializeAsAny[TTMetadata]`.
- `cube.task`: change `class TaskConfig(ABC, TypedBaseModel)` →
  `class TaskConfig[TTMetadata: TaskMetadata](ABC, TypedBaseModel)`;
  change `metadata: SerializeAsAny[TaskMetadata]` →
  `metadata: SerializeAsAny[TTMetadata]`.
- `cube.benchmark`: change `class BenchmarkConfig(TypedBaseModel, ABC)` →
  `class BenchmarkConfig[TTMetadata: TaskMetadata](TypedBaseModel, ABC)`;
  keep `task_metadata: ClassVar[dict[str, TaskMetadata]]` unchanged
  (cannot reference type parameters under PEP 526; would also be unsound
  under invariant `dict`); narrow the return type of `tasks()` to
  `Mapping[str, TTMetadata]` (covariant). Widen `subset_from_list`'s
  `tasks` parameter from `list[…]` to `Sequence[…]` (covariant) so
  parametrised subclasses can pass narrowed lists. Add an
  `__init_subclass__` early-return guard to skip Pydantic-synthesised
  parametrised intermediates (identifiable by `[` in the class name).
- `cube.benchmark`: change `class Benchmark(ABC)` →
  `class Benchmark[TBenchConfig: BenchmarkConfig](ABC)`; type
  `__init__(self, config: TBenchConfig)` and
  `self.config: TBenchConfig`.
- `CompositeBenchmarkConfig` subclasses `BenchmarkConfig[TaskMetadata]`
  (heterogeneous case). `CompositeBenchmark` subclasses
  `Benchmark["CompositeBenchmarkConfig"]` (forward-ref string for the
  later-defined class) and drops its `self.config: ... = config  # type: ignore[assignment]`
  re-annotation.
- Update `TaskConfig` / `BenchmarkConfig` / `Benchmark` docstrings with
  one-paragraph notes on the parametrised form.
- Update `openspec/specs/task/spec.md` and
  `openspec/specs/benchmark/spec.md` to reflect the new signatures.
- No example or test changes required — examples and tests use the
  unparametrised form, which still works.

**Follow-up PRs in cube-harness (out of scope here):**

- Migrate `workarena-cube`, `osworld-cube`, etc. that have the unsound
  re-annotations on `metadata`, `task_metadata`, or `config`. Each
  cube's migration is up to three lines: parametrise the `TaskConfig`,
  `BenchmarkConfig`, and `Benchmark` subclasses, delete the overrides.

## Out of scope

- Touching the registry shape on `BenchmarkConfig`.
- Any change to `TaskMetadata` or `BenchmarkMetadata` themselves.

See [deltas.md](deltas.md) for the spec changes.
