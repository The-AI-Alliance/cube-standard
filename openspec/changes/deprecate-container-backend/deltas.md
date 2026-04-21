# Deltas — Deprecate ContainerBackend

**Targets:** `openspec/specs/container/spec.md`, `openspec/specs/task/spec.md`,
`openspec/specs/benchmark/spec.md`, `openspec/specs/resource/spec.md`

---

## DEPRECATED — `ContainerBackend`, `Container`, and concrete backends
**Spec:** container

`ContainerBackend`, `Container`, `ContainerConfig`, `ExecResult`, `ContainerStatus`,
`ContainerError`, `ContainerLaunchError`, `HealthCheckError`, and `ContainerExecError`
remain available but emit `DeprecationWarning` at import of `cube.container` and at
construction of any concrete backend in `cube.backends.{local,daytona,modal,toolkit}`.
Removal follows in a separate change once in-tree cubes have migrated.

New cubes MUST NOT use these types. Per-task container workloads use
`DockerServiceConfig(scope="task")` served by an `InfraConfig` subclass. Per-benchmark
shared stacks use `DockerServiceConfig(scope="benchmark")`.

The `container/spec.md` file itself is retained for the deprecation window and
annotated with a banner pointing to `resource/spec.md`.

---

## DEPRECATED — `TaskMetadata.container_config`
**Spec:** task

The `container_config: ContainerConfig | None` field on `TaskMetadata` is deprecated.
Tasks that need a container read `runtime_context["infra"]` and construct a per-task
`DockerServiceConfig` in `TaskConfig.make()`.

The `model_post_init` auto-launch branch (triggered when both `container_backend` and
`metadata.container_config` are set) emits `DeprecationWarning`.

---

## DEPRECATED — `container_backend` parameter
**Spec:** benchmark, task

The `container_backend: ContainerBackend | None` field on `Benchmark` and the
`container_backend` parameter on `TaskConfig.make()` and `Task.__init__` emit
`DeprecationWarning` when passed a non-`None` value. Benchmarks declare
infrastructure via `infra: InfraConfig` and `resources: list[ResourceConfig]` instead.

---

## ADDED — `runtime_context["infra"]` convention
**Spec:** task, benchmark

When a benchmark owns an `InfraConfig`, `Benchmark._setup()` SHOULD stash it under
the `"infra"` key in `self._runtime_context`. `TaskConfig.make()` MAY read
`runtime_context["infra"]` to launch per-task resources (L3).

This is a convention, not a base-class behavior: the base class does not set the key
automatically. `RuntimeContext` remains `dict[str, Any]`.

---

## REMOVED — `cube.resource.DockerImageConfig`
**Spec:** resource

`DockerImageConfig` was declared as a `ResourceConfig` subclass but no `InfraConfig`
ever implemented it — `LocalInfraConfig`, `AWSInfraConfig`, and `AzureInfraConfig`
all dispatch only on `VMResourceConfig` and `DockerServiceConfig`. The type and its
`cube/__init__.py` export are removed. The single test case in
`tests/test_resource_lifecycle.py` that instantiated it is deleted.

References to "DockerImageConfig" in the `resource.py:232` docstring and in
`docs/design-docker-service-provisioning.md` are updated to describe the
`DockerServiceConfig(scope="task")` path.

---

## MODIFIED — `DockerServiceConfig` documented for L3
**Spec:** resource

The `ResourceConfig` subclass list in `resource/spec.md` is updated: the
`DockerImageConfig` bullet is removed; the `DockerServiceConfig` bullet notes that
single-image per-task workloads (SWE-bench, MLE-bench, CTF) use
`DockerServiceConfig(scope="task", docker_images=[image], services={"main": port}, launch_script=...)`.

New invariant in `resource/spec.md`:

> `scope` is informational metadata read by the benchmark and harness; it does not
> change `InfraConfig` dispatch. An infra that serves `DockerServiceConfig` serves it
> at any scope.

---

See [proposal.md](./proposal.md) for rationale and migration guide.
