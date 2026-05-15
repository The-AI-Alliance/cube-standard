# Deltas — Remove the legacy `ContainerBackend` provisioning path

**Targets:** `openspec/specs/container/spec.md`, `openspec/specs/task/spec.md`, `openspec/specs/benchmark/spec.md`, `openspec/specs/server/spec.md`

Follow-up to the `Task.model_post_init` delta in
`openspec/changes/resource-convergence/deltas.md` (line ~54), which scheduled
the container-backend path for removal. Applied when this change lands.

---

## REMOVED — `ContainerBackend`
**Spec:** container

The serializable, harness-owned `ContainerBackend(TypedBaseModel, ABC)`
factory is removed: the `launch()` / `health_check()` surface, the
"Contracts for implementers (new backend)" section, the
"Concrete implementations live under `cube/backends/`" note, and the
invariant that `launch()` blocks until ready / runs the health check.

`ContainerConfig`, `Container` (now a `ResourceHandle`), `ExecResult`,
`ContainerStatus`, the container exceptions, `port_from_url`, and
`relocate_if_readonly` are retained. Provisioning is owned exclusively by
`InfraConfig` — see `../resource/spec.md`. `container/spec.md` now documents
only the task-side container *requirement* (`ContainerConfig`) and the live
handle abstraction (`Container`).

## REMOVED — `cube.backends` package
**Spec:** container

The `cube.backends` package (`local`, `modal`, `daytona`, `toolkit`) is
deleted. The local Docker driver `LocalContainer` is relocated to
`cube.local_container` (`from cube.local_container import LocalContainer`).
Daytona/Toolkit/Modal container drivers live in their `cube_infra_*`
packages, unchanged.

## REMOVED — `Task.container_backend`
**Spec:** task

The `container_backend: ContainerBackend | None = None` field on `class
Task` is removed. `model_post_init` no longer has a container-backend launch
mode: if `metadata.container_config` is set and `runtime_context["infra"]` is
present, the container is provisioned via the injected `InfraConfig`
(`cube.task_infra.launch_task_container`), then `_build_tool()` runs.

## REMOVED — `TaskConfig.make(container_backend=...)` parameter
**Spec:** task

`TaskConfig.make()` loses the `container_backend` parameter. The signature is
now `make(self, runtime_context: RuntimeContext | None = None) -> Task`.

## REMOVED — `BenchmarkConfig.container_backend`
**Spec:** benchmark

The deprecated `container_backend` instance field is removed.
`Benchmark.spawn()` and the `CompositeBenchmark` leaf-spawn call
`task_config.make(runtime_context=...)` only. The stale
`arbitrary_types_allowed` gotcha (justified by `ContainerBackend` holding
non-roundtrippable handles) is dropped — `BenchmarkConfig` no longer sets it.

## MODIFIED — server invariant on cross-process container provisioning
**Spec:** server

The invariant "`container_backend` is NOT forwarded to task subprocesses" is
replaced: container provisioning crosses the subprocess boundary via
`runtime_context` (the `InfraConfig` is published into
`_runtime_context["infra"]`, JSON-serialized, and rehydrated by `_type` on
the worker before `task_config.make(runtime_context=...)`).
