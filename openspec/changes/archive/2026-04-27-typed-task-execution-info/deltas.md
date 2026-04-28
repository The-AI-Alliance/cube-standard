# Deltas — Typed `TaskExecutionInfo` slot + remove `extra_info` bags

**Targets:** `openspec/specs/task/spec.md`, `openspec/specs/benchmark/spec.md`, `openspec/specs/cli/spec.md`

Applied as each commit lands.

## ADDED — `TaskExecutionInfo`
**Spec:** task

New base class:

```python
class TaskExecutionInfo(TypedBaseModel):
    """Heavy, lazy per-task execution data. Cube authors subclass with typed
    fields (problem statements, patches, archives, evaluator scripts, etc.).
    Default is the empty base — cubes with no heavy data leave the slot None.
    """
```

Subclassed by cubes that need heavy per-task data. Polymorphic via the
`TypedBaseModel` `_type` discriminator, like every other config layer.

## ADDED — `Task.execution_info`
**Spec:** task

New optional field on `Task`:

```python
class Task:
    metadata: TaskMetadata
    execution_info: TaskExecutionInfo | None = None
```

**Population sites (cube author chooses):**
- Inside `TaskConfig.make()` — most common; the worker reads from
  `cls.load_task_execution_info(task_id)`, validates against the typed
  subclass, and passes to the `Task` constructor.
- Inside `Task.model_post_init` — for cubes that prefer hydration during Task
  construction.
- Inside `Task.reset()` — for cubes that defer hydration to first reset.

The framework does not call into `execution_info` itself. Subclasses access
their typed fields via `self.execution_info.<field>` instead of
`self.metadata.extra_info["<key>"]`.

## REMOVED — `TaskMetadata.extra_info`
**Spec:** task

`TaskMetadata.extra_info: dict[str, Any]` is removed.

Cube authors needing extra per-task fields subclass `TaskMetadata` with typed
named fields. Polymorphism is preserved through the existing `_type`
discriminator on `TypedBaseModel`.

## REMOVED — `BenchmarkMetadata.extra_info`
**Spec:** benchmark

`BenchmarkMetadata.extra_info: dict[str, Any]` is removed.

Same rationale: cube authors subclass `BenchmarkMetadata` with typed fields
when they need extras. Most cubes will not need to.

## MODIFIED — `BenchmarkConfig.subset_from_glob`
**Spec:** benchmark

The `"extra_info.<key>"` dot-notation special case is removed. `glob_key`
accepts only top-level fields of the (subclassed) `TaskMetadata` instance.

Migration: cubes that previously called `subset_from_glob("extra_info.difficulty",
"hard")` change to `subset_from_glob("difficulty", "hard")` after promoting
`difficulty` to a typed field on their `TaskMetadata` subclass.

## MODIFIED — CSV loaders
**Spec:** benchmark

`task_metadata_from_csv` and `benchmark_metadata_from_csv` lose their special
JSON-decoded `extra_info` column handling.

Subclass-specific fields are encoded as their own columns. Complex types still
go through JSON-encoded strings, but listed explicitly per subclass (loaders
are unchanged for non-`extra_info` JSON fields like `tags`, `container_config`).

## MOVED — `task_execution_cache_dir()` / `load_task_execution_info()`
**From:** `BenchmarkConfig` (classmethods)
**To:** `TaskConfig` (classmethods)
**Spec:** task (added), benchmark (removed)

Single source of truth on `TaskConfig` — workers read via
`type(self).task_execution_cache_dir()` with no back-import to BenchmarkConfig.

```python
class TaskConfig:
    @classmethod
    def task_execution_cache_dir(cls) -> Path:
        """Default: ~/.cube/<top-level-package-name>/tasks_execution_info/.
        Override to customize the path (e.g. to match a custom benchmark name
        that differs from the package name)."""
        return get_cache_dir(cls.__module__.split(".")[0]) / "tasks_execution_info"

    @classmethod
    def load_task_execution_info(cls, task_id: str) -> dict[str, Any]:
        """Read processed per-task data written by BenchmarkConfig.install().
        Raises RuntimeError with a clear remediation message if the file is
        missing (signals install() has not run on this worker)."""
```

`BenchmarkConfig.install()` writes via `cls.task_config_class.task_execution_cache_dir()`
— exactly one definition of the path, no drift possible.

**Path key change:** the cache directory is now keyed by the Python **package** name
(`cls.__module__.split(".")[0]`, underscores), not by `benchmark_metadata.name` (which
may use hyphens). A benchmark named `"osworld-cube"` in its metadata but living in the
`osworld_cube` package caches at `~/.cube/osworld_cube/` under the new formula.
Override `task_execution_cache_dir()` on the subclass if the two differ and a specific
path is required.

`BenchmarkConfig.task_execution_cache_dir()` and
`BenchmarkConfig.load_task_execution_info()` are deleted.

## ADDED — `TaskConfig.verify_installed()`
**Spec:** task

```python
class TaskConfig:
    @classmethod
    def verify_installed(cls) -> None:
        """Optional fail-fast check that data this task relies on is locally
        available on this worker. Default: no-op. Cube authors override with a
        check appropriate to their cache (e.g. `not list(cls.task_execution_cache_dir().iterdir())`,
        or `HF_HOME / 'datasets' / '...'.exists()`).

        Convention: TaskConfig.make() calls type(self).verify_installed() at
        the top, so a misconfigured worker fails fast with an actionable error
        instead of timing out on a surprise download."""
```

The check lives on `TaskConfig` (worker-side) so workers do not need to import
the owning `BenchmarkConfig` to verify their environment — preserving the
PR #118 worker-side decoupling.

## ADDED — `cube install <bench>` CLI subcommand
**Spec:** cli

```
cube install <benchmark-id>
```

Discovers the `BenchmarkConfig` class via the `cube.benchmarks` entry-point
group (same mechanism `cube test` uses) and invokes its `install()` classmethod.

Operators wire this into Dockerfiles (`RUN cube install <bench>`), init
containers, or run it on shared volumes. Without the subcommand, the
documented fallback is `python -c 'from <pkg> import <Cls>; <Cls>.install()'`.

## MODIFIED — `_template/` scaffold
**Spec:** task / benchmark (referenced by template tests)

The new-cube template gains:
- A `TaskExecutionInfo` subclass stub (commented out by default — most cubes
  start with no heavy data).
- A `verify_installed()` override stub on the `TaskConfig` subclass.
- Updated `make()` body showing the typed-hydration pattern.
- No `extra_info` field on the `TaskMetadata` subclass (no more `dict[str,
  Any]` smuggling channel).

## MODIFIED — `.claude/skills/{new-cube,review-cube}`
**Non-spec, but in-repo:**

- `new-cube` interview / scaffold instructions reflect the typed
  `TaskExecutionInfo` slot, the deleted `extra_info` bag, the relocated cache
  helpers, and the optional `verify_installed()` override. Adds two worked
  examples (HF-dataset cube, git-repo cube) showing the install/make split.
- `review-cube` checks gain rules:
  - Cube `TaskMetadata` subclass MUST NOT define a field named `extra_info`
    (and the base no longer has one — type-checked).
  - Cube `TaskMetadata` subclass fields SHOULD be lightweight (heuristic
    size threshold per task — flagged, not blocked).
  - Heavy per-task data MUST live on a `TaskExecutionInfo` subclass, not on
    `TaskMetadata`.

---

See [proposal.md](proposal.md) for rationale and the operator deployment model.
