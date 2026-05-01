# Align `TaskConfig.task_execution_cache_dir()` with `BenchmarkConfig.cache_dir()`

**Status:** Accepted
**Date:** 2026-05-01
**Scope:** `cube.task`, `cube.benchmark`
**Targets:** cube-standard PR #135 (closes #126)

## Problem

`TaskConfig.task_execution_cache_dir()` keys on the top-level Python package
name (`cls.__module__.split(".")[0]`), while `BenchmarkConfig.cache_dir()` keys
on `benchmark_metadata.name`. Whenever the two differ — which is every cube,
since Python identifiers can't contain hyphens (e.g. `osworld_cube` vs
`osworld-cube`) — the per-task execution cache lands at a different path than
the benchmark's own cache directory unless the cube author manually overrides
`task_execution_cache_dir()`.

`load_task_execution_info(cls, task_id)` and `verify_installed(cls)` were
classmethods, even though both have a natural per-instance form (`self.task_id`
is already on `TaskConfig`).

## Solution

`BenchmarkConfig.__init_subclass__` back-stamps `cls.cache_dir()` onto
`task_config_class._benchmark_cache_dir` (a `ClassVar[Path | None]`).
`TaskConfig.task_execution_cache_dir()` reads
`cls.__dict__.get("_benchmark_cache_dir")` (directly-set, not MRO-inherited)
and falls back to `~/.cube/<top-level-package-name>/` when no stamp is
attached.

Why store the resolved `Path` (not just the name): cubes that override
`cache_dir()` to point somewhere other than `~/.cube/<benchmark-name>/`
(e.g. OSWorld co-locates the cache with VM data) get the override applied
to the per-task cache automatically — no need for a parallel
`task_execution_cache_dir()` override.

Why `ClassVar` (not a serialized field): the value is identical for every
task in a benchmark, so it doesn't belong on per-instance config payloads.
ClassVars aren't Pydantic fields and aren't serialized; the worker
re-populates them when it imports the cube package.

Why `cls.__dict__.get` (not attribute lookup): a `TaskConfig` subclass with
no owning `BenchmarkConfig` (test scaffolds, derived experimentation
classes) must not silently inherit the parent's stamp through the MRO.

`__init_subclass__` raises `TypeError` if two `BenchmarkConfig` subclasses
point at the same `task_config_class`. Each benchmark must own its own
`TaskConfig` subclass — sharing would silently overwrite the stamp.

`load_task_execution_info` and `verify_installed` become instance methods
that use `self.task_id` directly. `task_execution_cache_dir` stays a
classmethod since `BenchmarkConfig.install()` calls it without a task
instance.

## Backwards compatibility

Cubes that override `task_execution_cache_dir()` keep working unchanged
(e.g. OSWorld co-locates the cache with VM data — the override returns
its custom path regardless of the back-stamp).

Cubes that previously called `cls.load_task_execution_info(task_id)` or
`type(self).verify_installed()` need a one-line migration to
`self.load_task_execution_info()` / `self.verify_installed()`. Cube-harness
migration lands in a separate PR.

Existing on-disk caches at the old `~/.cube/<package>/` path are not
auto-migrated. Operators run `cube install <bench>` once on each worker
to populate the new path.

## Migration

**This PR (cube-standard):**

- `cube.task`: add `_benchmark_cache_dir: ClassVar[Path | None] = None` on
  `TaskConfig`. Change `task_execution_cache_dir()` default to use it (with
  package-name fallback). Convert `load_task_execution_info` and
  `verify_installed` to instance methods.
- `cube.benchmark`: in `BenchmarkConfig.__init_subclass__`, stamp
  `cls.cache_dir()` onto `task_config_class._benchmark_cache_dir` after the
  class-level validations pass. Skip when `task_config_class` is abstract
  (composite placeholder) or `benchmark_metadata` is dynamic (`@property`).
  Raise `TypeError` on shared-class collision.
- Realign `openspec/specs/{task,benchmark}/spec.md`, the `_template/`
  scaffold, and the `new-cube` skill (`SKILL.md`, `pitfalls.md`,
  `todo-checklist.md`) with the new shape.
- Refactor existing tests where multiple `BenchmarkConfig` subclasses
  shared one `TaskConfig` subclass (composite + benchmark tests) — each
  pair now declares its own `TaskConfig` subclass.

**Follow-up PRs in cube-harness (out of scope here):**

- Migrate call sites in `osworld-cube`, `swebench-live-cube`,
  `swebench-verified-cube`, `terminalbench-cube`:
  `type(self).verify_installed()` → `self.verify_installed()` and
  `type(self).load_task_execution_info(self.task_id)` →
  `self.load_task_execution_info()`.

See [deltas.md](deltas.md) for the spec changes.
