# Deprecate ContainerBackend and migrate SWE/terminal cubes to resource/infra

**Status:** Draft
**Date:** April 2026
**Scope:** `cube.container`, `cube.backends`, `cube.task.TaskMetadata.container_config`,
`Benchmark.container_backend`, `TaskConfig.make()` signature

---

## Problem

Two parallel abstractions exist for per-task container provisioning:

1. **Legacy** — `ContainerBackend` + `cube.backends/{local,daytona,modal,toolkit}.py`. A
   single-container runtime attached to `Benchmark.container_backend` and auto-launched
   in `Task.model_post_init` from `TaskMetadata.container_config`.
2. **Unified** — `ResourceConfig` + `InfraConfig` + `ResourceHandle`. L1/L2/L3 lifetime
   model, cloud-agnostic, with `provision()` / `launch()` / `cleanup_stale()` tagged
   by `run_id` for crash recovery.

The unified pattern covers everything the legacy pattern does and more: TTL-based GC,
cross-process cleanup via `run_id`, multi-container stacks, pre-baked volumes,
polymorphic serialization of the infra config. `DockerServiceConfig(scope="task")` is
semantically identical to what `ContainerBackend.launch(ContainerConfig)` does, minus
the lifecycle affordances.

Having both paths forces every new cube author to pick, splits implementation effort
(cloud backends evolve on the unified path, sandbox backends on the legacy one), and
leaves dead code: `DockerImageConfig` is declared in `resource.py` but no `InfraConfig`
ever implements it.

Three cubes in cube-harness actually require the legacy path today:
`swebench-verified-cube`, `swebench-live-cube`, `terminalbench-cube`. The other five
(`arithmetic-cube`, `miniwob`, `workarena`, `webarena-verified`, `osworld-cube`)
accept the `container_backend` parameter purely to conform to base signatures and
never launch a container through it.

---

## Proposed change

Collapse to the unified path, in two PRs:

1. **This PR (deprecation + dead-code removal).** `ContainerBackend`, the four concrete
   backends, `TaskMetadata.container_config`, and the `container_backend` parameter on
   `Benchmark` / `Task` / `TaskConfig.make()` emit `DeprecationWarning` when
   constructed or passed a non-`None` value. `DockerImageConfig` — which has no
   implementation anywhere — is removed outright. No behavior change for existing
   cubes; CI stays green across the migration window.

2. **Follow-up PR (`remove-container-backend`).** After cube-harness migrates the
   three cubes and cube-resources ships `ToolkitInfraConfig` / `DaytonaInfraConfig`,
   delete `cube/container.py`, `cube/backends/`, strip the deprecated fields from
   base classes, and update scaffolding (`_template/`, examples).

Cloud/sandbox-native replacements for the four legacy backends:

| Legacy                     | Replacement                                           |
|----------------------------|-------------------------------------------------------|
| `LocalContainerBackend`    | `LocalInfraConfig` (exists)                           |
| `ToolkitContainerBackend`  | `ToolkitInfraConfig` (new, cube-resources)            |
| `DaytonaContainerBackend`  | `DaytonaInfraConfig` (new, cube-resources)            |
| `ModalContainerBackend`    | dropped — no migration target                         |

AWS/Azure per-task Docker is explicitly out of scope. Those backends are right for
long-running L2 stacks (WebArena, OSWorld), not ~500-image-per-run workloads like
SWE-bench. Sandbox-as-a-service (Toolkit, Daytona) is the natural home for per-task
containers and already operates on the per-image-per-task model.

---

## Alternatives considered

- **Keep both paths indefinitely.** Rejected: authors of new cubes have to pick, and
  the unified path already dominates for all cubes except the three covered here.
- **Hard-delete without deprecation.** Rejected: the migration PRs in cube-harness
  and cube-resources land separately; a deprecation window keeps CI green during
  the transition.
- **Resurrect `DockerImageConfig` with a dedicated `DockerInfraConfig`.** Rejected:
  `DockerServiceConfig` with `scope="task"`, one entry in `docker_images`, and one
  service port covers the same ground with no new type. The `DockerImageConfig`
  docstring references a `DockerInfraConfig` that has never existed.

---

## Runtime-context carrier convention

Benchmarks that own an `InfraConfig` put it on `self._runtime_context["infra"]` in
`_setup()`. `TaskConfig.make()` reads `runtime_context["infra"]` to launch per-task
resources. This keeps `TaskConfig.make()`'s signature stable and follows the
existing `RuntimeContext = dict[str, Any]` convention documented in
[task/spec.md](../../specs/task/spec.md).

---

## Migration guide (to be added to `DEPRECATED.md`)

**Before** — legacy `ContainerBackend`:

```python
class MyBenchmark(Benchmark):
    container_backend: ContainerBackend | None = None
    task_metadata = {
        "t1": TaskMetadata(id="t1", container_config=ContainerConfig(image="my:latest")),
    }

class MyTaskConfig(TaskConfig):
    def make(self, runtime_context=None, container_backend=None) -> MyTask:
        metadata = MyBenchmark.task_metadata[self.task_id]
        return MyTask(
            metadata=metadata,
            tool_config=self.tool_config,
            runtime_context=runtime_context,
            container_backend=container_backend,  # auto-launches container in model_post_init
        )
```

**After** — unified `InfraConfig` path:

```python
class MyTaskMetadata(TaskMetadata):
    image: str
    ram_gb: float = 4.0
    cpu_cores: float = 2.0

class MyBenchmark(Benchmark):
    infra: InfraConfig = LocalInfraConfig()

    def _setup(self) -> None:
        self.infra.cleanup_stale()
        self._runtime_context["infra"] = self.infra

class MyTaskConfig(TaskConfig):
    def make(self, runtime_context=None) -> MyTask:
        metadata = MyBenchmark.task_metadata[self.task_id]
        infra = runtime_context["infra"]
        port = 8080  # declared per-task or pooled
        resource = DockerServiceConfig(
            name=f"my-bench-{self.task_id}",
            scope="task",
            docker_images=[metadata.image],
            services={"main": port},
            launch_script=f"docker run -d -p {port}:{port} {metadata.image}",
        )
        handle = infra.launch(resource)
        return MyTask(
            metadata=metadata,
            tool_config=self.tool_config,
            runtime_context=runtime_context,
            _handle=handle,  # closed in Task.close()
        )
```

---

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Three cubes in cube-harness fail during the deprecation window | This PR is pure deprecation — no runtime behavior changes. The migration PRs land behind this one. |
| `DockerImageConfig` referenced in downstream code we don't control | The only references in The-AI-Alliance org are a single test case and docstrings, both owned by this PR. |
| Sandbox-backend semantics differ from Docker (e.g., no persistent volumes in Toolkit) | Out of scope here — each new `InfraConfig` subclass picks a shape that matches the sandbox's capabilities, surfaced via `capabilities()`. |

---

## Open questions

1. Should `Benchmark.resources` become the authoritative declaration of per-task
   resource *templates* (with the task_id substituted at launch time), or does the
   per-task `DockerServiceConfig` stay constructed inside `TaskConfig.make()` as
   shown in the migration guide? The guide defers this: both shapes are allowed.
2. When should `Benchmark._setup()` stash the infra in `_runtime_context`
   automatically (base-class behavior) vs. leave it to the subclass? Leaving it
   explicit for now; can be promoted to base-class convenience later.
