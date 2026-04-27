# Typed `TaskExecutionInfo` slot + remove `extra_info` bags

**Status:** Proposed
**Date:** 2026-04-25
**Scope:** `cube.task`, `cube.benchmark`, `cube.cli`, `_template/`, `.claude/skills/{new-cube,review-cube}`
**Targets:** stacks on top of `feat/benchmark-config` (PR #118)
**Related:** PR #118 follow-up; explicit "open design questions" item from #118 description

---

## Problem

`TaskMetadata.extra_info: dict[str, Any]` is a stringly-typed bag that has drifted
into being the smuggling channel for **heavy per-task execution data** (problem
statements, patches, test commands, archives). Three concrete issues:

1. **Untyped access.** Cubes read `metadata.extra_info["instruction"]` with no
   autocomplete, no schema validation, and easy typo failures at runtime.
2. **Wrong lifetime in the wrong slot.** `TaskMetadata` is meant to be lean and
   eager-loaded — it ships in the wheel and powers `cube list`, registry
   listings, and filtering. The current pattern of merging
   `load_task_execution_info()` into `extra_info` at `get_task_configs()` emit
   time works, but it's a workaround: the type system says "this is metadata,"
   the data says "this is heavy execution payload." Reviewers can't tell by
   looking which kind they're dealing with.
3. **No enforced boundary.** Nothing stops a cube author from putting heavy
   data eagerly into `task_metadata.json` (bloating `cube list` output and
   registry payloads). `review-cube` has no rule to flag it.

The same critique applies to `BenchmarkMetadata.extra_info` — same anti-pattern,
just at benchmark scope. Same fix.

## Non-goals (explicitly rejected after discussion)

We considered, and rejected, building any of the following into cube-standard:

- A pluggable `DataStore` / `BlobRef` abstraction over HF / S3 / git.
- A framework-managed manifest that ships content-addressed references in
  `TaskConfig`.
- Any cross-machine data-distribution coordination.

These belong to mature ecosystem tools (HuggingFace Datasets, git, Docker
images, shared volumes). The framework's job is the **protocol** (typed slot,
optional install hook); distribution is the **operator's** problem, solved with
existing infrastructure. See PR description for the longer reasoning.

## Solution

A small, principled refactor with five moving parts.

### 1. Typed `execution_info` slot on `Task`

```python
class TaskExecutionInfo(TypedBaseModel):
    """Heavy, lazy, per-task execution data. Cube authors subclass with typed
    fields. Default is empty for cubes with no heavy data."""

class Task:
    metadata: TaskMetadata                          # eager, ships in wheel
    execution_info: TaskExecutionInfo | None = None # lazy, populated on the worker
```

`execution_info` is populated either inside `TaskConfig.make()` (most common)
or inside `Task.model_post_init` / `Task.reset()`. The cube author owns
hydration — the framework provides the typed slot, nothing more.

### 2. Delete `TaskMetadata.extra_info` and `BenchmarkMetadata.extra_info`

Cube authors who need extra fields subclass `TaskMetadata` / `BenchmarkMetadata`
with named, typed fields. This is consistent with how `TaskConfig` and
`BenchmarkConfig` already work (`TypedBaseModel` polymorphism via `_type`).

Consequences:
- `subset_from_glob` loses its `"extra_info.<key>"` dot-notation special case.
  It becomes plain field access on the typed subclass.
- `task_metadata_from_csv` / `benchmark_metadata_from_csv` drop their special
  `extra_info` JSON-decoding column. CSVs encode each subclass field as its own
  column.

### 3. `TaskConfig` owns the per-task execution cache directory

Move `task_execution_cache_dir()` and `load_task_execution_info()` from
`BenchmarkConfig` to `TaskConfig`. Single source of truth, no consistency
check needed, and crucially — no worker → BenchmarkConfig back-import (which
would re-introduce exactly the coupling PR #118 just removed).

```python
class TaskConfig:
    @classmethod
    def task_execution_cache_dir(cls) -> Path:
        """Default: ~/.cube/<top-level-package-name>/tasks_execution_info/.
        Override to customize (e.g. to match a custom benchmark name)."""
        return get_cache_dir(cls.__module__.split(".")[0]) / "tasks_execution_info"

    @classmethod
    def load_task_execution_info(cls, task_id: str) -> dict[str, Any]:
        """Read processed per-task data from the cache. Raises with a clear
        remediation message if the file is missing."""

    @classmethod
    def verify_installed(cls) -> None:
        """Optional fail-fast check. Default: no-op. Cube authors override
        (typically a one-liner verifying task_execution_cache_dir() is non-empty,
        or that an HF dataset / git clone is locally present)."""
```

`BenchmarkConfig.install()` is unchanged in contract but writes via
`cls.task_config_class.task_execution_cache_dir()` so there is exactly one
definition of the path. `BenchmarkConfig.task_execution_cache_dir()` and
`BenchmarkConfig.load_task_execution_info()` are deleted.

### 4. `verify_installed()` convention on the worker

`TaskConfig.make()` calls `type(self).verify_installed()` at the top by
convention. Workers fail fast with an actionable error instead of timing out
on a surprise download.

### 5. `cube install <bench>` CLI subcommand

Small UX addition mirroring `cube test <bench>`: discovers the `BenchmarkConfig`
class via the `cube.benchmarks` entry-point group, calls its `install()`
classmethod. Operators wire this into Dockerfiles, init containers, or run it
manually on shared volumes. Without it, operators fall back to
`python -c 'from <pkg> import <Cls>; <Cls>.install()'` — workable but less
discoverable.

## Operator deployment model (non-normative, documented in spec)

`install()` runs **once per worker environment**, not once per experiment. The
three standard strategies:

1. **Bake into the worker image** — add `RUN cube install <bench>` to the
   Dockerfile. Every container starts with the cache warm. This is the default
   pattern for production agent fleets.
2. **Shared volume** — mount NFS / EFS / S3FS on all workers. Run `install()`
   once on any node that mounts the volume.
3. **Worker bootstrap** — each worker runs `cube install <bench>` as part of
   its startup. N parallel cold downloads on first run; warm thereafter.
   Acceptable for ≤dozens of workers, painful at thousands — push to (1).

The framework provides the hook and `verify_installed()` for fail-fast. It does
not coordinate downloads, manage shared caches, or move bytes between hosts.

## Migration

**This PR (cube-standard `feat/typed-task-execution-info`):**
- Add `TaskExecutionInfo`, `Task.execution_info` slot.
- Delete `TaskMetadata.extra_info` / `BenchmarkMetadata.extra_info`.
- Move `task_execution_cache_dir()` / `load_task_execution_info()` to
  `TaskConfig`; delete from `BenchmarkConfig`.
- Add `TaskConfig.verify_installed()` classmethod.
- Add `cube install <bench>` CLI subcommand.
- Update `subset_from_glob`, CSV loaders, `_template/`, OpenSpec specs, and
  `.claude/skills/{new-cube,review-cube}`.
- Update counter-cube + toy examples (the only in-repo cubes that touch
  `extra_info` today).

**Follow-up PRs in cube-harness (out of scope here):**
- Migrate `osworld-cube`, `swebench-live-cube`, `swebench-verified-cube`,
  `terminalbench-cube` to typed `TaskExecutionInfo` and the relocated cache
  helpers. The contract is mechanical: drop the `model_copy(update={"extra_info":
  ...})` dance, replace `metadata.extra_info["x"]` with `execution_info.x`,
  point `install()` at `cls.task_config_class.task_execution_cache_dir()`.

## Out of scope

- Any data-distribution layer (rejected — see Non-goals).
- Backwards-compat shim for `extra_info`. Per the project's no-shims policy,
  cubes migrate in parallel.
- `BenchmarkPool` (still deferred from PR #118).

See [deltas.md](deltas.md) for the spec changes.
