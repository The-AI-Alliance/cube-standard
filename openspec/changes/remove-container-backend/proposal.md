# Remove the legacy `ContainerBackend` provisioning path

**Status:** Draft
**Date:** May 2026
**Depends on:** `deprecate-container-backend` (PR #115), `feat/daytona-infra-config` (PR #116)
**Tracks:** GitHub issue #94 (Track 1)

---

## Background

`deltas.md` of `resource-convergence` already anticipated this removal — its
`Task.model_post_init` delta (line ~54) notes the container-backend path is
"preserved for back-compat with the legacy `cube.backends.*` stubs, but is
scheduled for removal in a follow-up". This change is that follow-up.

`deprecate-container-backend` (PR #115) flagged the parameter
(`Field(deprecated=True)`) and the `DaytonaInfraConfig` work (PR #116)
landed the InfraConfig path the migration converges on. With all in-tree
benchmarks now declaring container needs via `TaskMetadata.container_config`
and provisioning through the injected `InfraConfig`, the legacy factory has
no remaining callers and can be deleted.

---

## What changes

- **Remove `ContainerBackend`** from `cube.container`. The serializable
  `TypedBaseModel` factory (`launch()` / `health_check()`) and its invariants
  (blocking launch, health-check semantics) are gone.
- **Delete the `cube.backends` package** entirely (`local`, `modal`,
  `daytona`, `toolkit`). The local Docker driver `LocalContainer` is
  relocated to a new module — `from cube.local_container import
  LocalContainer`. The Daytona/Toolkit/Modal container drivers continue to
  live in their `cube_infra_*` packages (unchanged).
- **Drop `Task.container_backend`** (constructor field) and
  **`BenchmarkConfig.container_backend`** (instance field, forwarded by
  `spawn()`).
- **Remove the `container_backend` parameter** from `TaskConfig.make()` —
  the signature is now `make(self, runtime_context: RuntimeContext | None =
  None) -> Task`. `Benchmark.spawn()` and the composite leaf-spawn call
  `task_config.make(runtime_context=...)` only.
- **`ContainerConfig` is retained** — it is the live
  `TaskMetadata.container_config` type consumed by the InfraConfig path (via
  `cube.task_infra.launch_task_container`). It is not deprecated and is
  reframed in `container/spec.md` as the task-side container *requirement*.

Container provisioning is now exclusively via `InfraConfig`: a `Task`
launches its container in `model_post_init` iff `metadata.container_config is
not None and runtime_context is not None and "infra" in runtime_context`;
`launch_task_container` provisions through the injected `InfraConfig`.
`LocalInfraConfig` is the default/local infra.

---

## Impact

- **Breaking** for any caller passing `container_backend` to `Task`,
  `TaskConfig.make()`, or `BenchmarkConfig`. Migration: declare container
  needs via `TaskMetadata.container_config` and publish the infra into
  `runtime_context["infra"]` from `Benchmark._setup()`.
- Imports of `cube.backends.*` break — use `cube.local_container.LocalContainer`
  for the local Docker driver; cloud drivers move to `cube_infra_*` packages.
- A paired **cube-harness** PR updates the downstream callers
  (`episode.py`, `experiment.py`) that previously threaded
  `container_backend`, so both land together.

---

## Sequencing

1. `deprecate-container-backend` (PR #115) — soft-deprecation. Done.
2. `feat/daytona-infra-config` (PR #116) — InfraConfig convergence. Done.
3. This change — hard removal, paired with the cube-harness caller update.
