# Container Abstraction

**Module:** `cube.container`

## Purpose

The task-side container *requirement* plus the live single-container handle
abstraction. A `Task` declares WHAT container it needs via
`TaskMetadata.container_config` (`ContainerConfig`); *how* the container is
provisioned is owned entirely by the injected `InfraConfig` — see
[resource](../resource/spec.md). `Container` is the live handle the tool layer
talks to.

This is the narrow, single-container surface. For multi-container stacks and
cloud VMs, see [resource](../resource/spec.md). The local Docker driver
(`LocalContainer`) lives in `cube.local_container`; Daytona/Toolkit/Modal
drivers live in their respective `cube_infra_*` packages.

## Public API

### `ContainerConfig` (serializable, task-owned)
```python
class ContainerConfig(TypedBaseModel):
    image: str
    ram_gb: float = 4.0
    cpu_cores: float = 2.0
    gpu: bool = False
    disk_gb: float = 10.0
    ports: list[int] | None = None
```

Declares WHAT container a task needs. Set on `TaskMetadata.container_config`
and consumed by the `InfraConfig` path in `Task.model_post_init` (via
`cube.task_infra.launch_task_container`). When `metadata.container_config` is
set and `runtime_context["infra"]` is present, the container is provisioned
through the injected `InfraConfig` and the live handle is attached to the
Task. `ContainerConfig` is the live task-requirements type — it is not
deprecated.

### `Container` (live handle; NOT serializable)

`Container` IS a `ResourceHandle` (`class Container(ResourceHandle, ABC)`), so
an `InfraConfig.launch()` that returns a single `Container` satisfies the
`ResourceHandle` protocol directly — no wrapper indirection. It inherits the
`ResourceHandle` bookkeeping (`run_id`, `resource`, `infra`, `created_at`,
`expires_at`, `endpoint`, `endpoints`) and adds the container capability
surface.

```python
class Container(ResourceHandle, ABC):
    @abstractmethod
    def exec(self, command, timeout=None, workdir=None, env=None) -> ExecResult
    def exec_long_running(self, command, *, timeout, poll_interval=30,
                          workdir=None, env=None) -> ExecResult   # default: exec()
    @abstractmethod
    def forward_port(self, container_port: int) -> int       # reachable port on host
    @abstractmethod
    def get_url(self, container_port: int) -> str           # convenience URL
    @abstractmethod
    def stop(self, timeout: int = 10) -> None                # idempotent
    @abstractmethod
    def get_status(self) -> ContainerStatus

    @property @abstractmethod
    def id(self) -> str

    def close(self) -> None                                  # delegates to stop()
    @property
    def container(self) -> "Container"                       # returns self
```

`close()` is satisfied by delegating to `stop()` so callers can use the handle
uniformly via `with` / `close()`. The `container` property returns `self`
(kept for uniformity with legacy multi-container handles).

`exec_long_running` defaults to `exec(command, timeout=timeout)` — fine for
backends with reliable exec streaming (LocalContainer, DaytonaContainer,
ModalContainer). Backends whose exec primitive is unreliable on long-running
commands (ToolkitContainer — see `docs/toolkit-exec-relay-design.md`) override
it to background the command and poll a sentinel file for completion.

### `ExecResult` / `ContainerStatus` (dataclasses)
```python
@dataclass
class ExecResult:
    stdout: str = ""; stderr: str = ""; exit_code: int = 0; duration_seconds: float = 0.0

@dataclass
class ContainerStatus:
    running: bool = False
    healthy: bool = False
    resource_usage: dict[str, float] = {}
    backend_info: dict[str, Any] = {}
```

### Exceptions
- `ContainerError` (base)
- `ContainerLaunchError` — failed to start
- `HealthCheckError` — health check failed
- `ContainerExecError` — exec failed

### Utility
- `port_from_url(url)` → int — extracts effective port (443 for https, 80 for http)
- `relocate_if_readonly(container, working_dir, new_wd, *, extra_setup=None)` → str —
  copies `working_dir` to `new_wd` if it isn't writable by the runtime user;
  returns the effective working directory. Used by cubes in `_build_tool()`.

## Invariants

1. `stop()` is idempotent; `close()` delegates to `stop()`.
2. `forward_port()` returns a host-reachable port. On cloud backends this may be
   an SSH tunnel; on local Docker, the host-mapped port.
3. `Container` is a `ResourceHandle` — never serialize it; pass `run_id` across
   process boundaries and let the target call `infra.cleanup(run_id)`.

## Notes for implementers

Provisioning is owned by `InfraConfig` (see [resource](../resource/spec.md)).
A new container backend is an `InfraConfig` whose `launch()` returns a
`Container` subclass implementing all abstract methods. Keep credentials in
environment variables — never fields on `InfraConfig` (they'd be serialized).
