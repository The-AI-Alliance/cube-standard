# Container Abstraction

**Module:** `cube.container`

## Purpose

Abstracts away container runtimes (Docker, Modal, Daytona, Toolkit). A `Benchmark` or
`Task` declares WHAT container it needs (`ContainerConfig`); a `ContainerBackend`
knows HOW to run it. `Container` is the live handle.

This is the narrow, single-container abstraction. For multi-container stacks and
cloud VMs, see [resource](../resource/spec.md).

## Public API

### `ContainerConfig` (serializable, benchmark-owned)
```python
class ContainerConfig(TypedBaseModel):
    image: str
    ram_gb: float = 4.0
    cpu_cores: float = 2.0
    gpu: bool = False
    disk_gb: float = 10.0
    ports: list[int] | None = None
```

Declared on `TaskMetadata.container_config`. If a Task has a backend and a config, the
container is auto-launched in `Task.model_post_init`.

### `ContainerBackend` (serializable, harness-owned)
```python
class ContainerBackend(TypedBaseModel, ABC):
    timeout_seconds: int = 1800
    backend_config: dict[str, Any] = {}

    @abstractmethod
    def launch(self, config: ContainerConfig) -> Container    # blocks until ready

    def health_check(self, container: Container) -> bool      # override; default True
```

Concrete implementations live under `cube/backends/`: `local.py`, `modal.py`,
`daytona.py`, `toolkit.py`.

### `Container` (live handle; NOT serializable)
```python
class Container(ABC):
    @abstractmethod
    def exec(self, command, timeout=None, workdir=None, env=None) -> ExecResult
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
```

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

## Invariants

1. `launch()` blocks until the container is ready (health check included if defined).
2. `stop()` is idempotent.
3. `forward_port()` returns a host-reachable port. On cloud backends, this may be an
   SSH tunnel; on local Docker, it's the host-mapped port.
4. Health check failures call `container.stop()` before raising `HealthCheckError`.

## Contracts for implementers (new backend)

Subclass `ContainerBackend` and provide `launch()`. Return a `Container` subclass
implementing all abstract methods. Override `health_check()` if you need custom probes.

Keep credentials in environment variables — never fields on `ContainerBackend`
(they'd be serialized).
