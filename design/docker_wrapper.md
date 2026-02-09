# CUBE Container API Specification

## Overview

The Container API provides a unified abstraction for launching and communicating with Docker containers across different backends (local Docker, Modal, HPC via EAI Toolkit).

## Primary Use Case

Ray-based parallel evaluation where workers need to spin up containers, run benchmarks, and clean up. Blocking during container startup is acceptable because Ray handles concurrency at the worker level.

```python
@ray.remote
def evaluate_task(task_id, container_config):
    container = container_config.make()  # Blocks until ready (minutes OK)
    result = container.exec("run_benchmark.sh")
    container.stop()
    return result

# Launch 100 evaluations in parallel
futures = [evaluate_task.remote(i, config) for i in range(100)]
results = ray.get(futures)
```

## Key Design Decisions

**1. Separation of Config and Container**
- **ContainerConfig:** Serializable specification (can pass to Ray workers)
- **Container:** Live object with active connections (not serializable)
- Rationale: Ray workers need to serialize configs but not live connections

**2. Blocking `make()` Method**
- Blocks until container ready (or timeout)
- For Toolkit/HPC: may take 30 minutes (SLURM queue wait)
- Rationale: Ray workers can afford to block on I/O. Simpler than callbacks/polling.

**3. Backend as Abstract Classes**
- Each backend subclasses `Container` with different implementations
- Local uses docker exec, Modal uses HTTP RPC, Toolkit uses SSH
- Rationale: Backends are too different to share implementation

**4. Configurable Timeouts**
- Local: 30 seconds, Modal: 2 minutes, Toolkit: 30 minutes
- Different backends have vastly different startup times

## Core API

### ContainerConfig

Declarative, serializable specification of what container to create.

```python
@dataclass
class ContainerConfig:
    """Container specification that can be passed to Ray workers."""
    
    # Required
    image: str  # Docker image (assumes already in registry)
    backend: Literal["local", "modal", "toolkit"]
    
    # Resources
    ram_gb: float = 4.0
    cpu_cores: float = 2.0
    gpu: bool = False
    disk_gb: float = 10.0
    
    # Lifecycle
    timeout_seconds: int = 1800  # Max wait for container ready
    health_check: Callable[[Container], bool] | None = None
    
    # Networking
    ports: List[int] | None = None  # Ports to expose
    
    # Backend-specific options (SLURM partition, Modal region, etc)
    backend_config: Dict[str, Any] | None = None
    
    def make(self) -> Container:
        """
        Create and start container. Blocks until ready or timeout.
        
        Raises: TimeoutError, HealthCheckError, ConnectionError, BackendError
        """
```

**Why separate from Container?** Config is serializable (can pass to Ray workers). Container has live connections (not serializable).

### Container (Abstract Base Class)

Live, stateful object with active connections.

```python
class Container(ABC):
    """Running container with exec and port forwarding capabilities."""
    
    @abstractmethod
    def exec(
        self, 
        command: str,
        timeout: int | None = None,
        workdir: str | None = None,
        env: Dict[str, str] | None = None,
    ) -> ExecResult:
        """
        Execute command inside container.
        
        Backend implementations:
        - Local: docker exec
        - Modal: HTTP RPC to running container
        - Toolkit: SSH to HPC node, then docker exec
        
        Returns: ExecResult(stdout, stderr, exit_code, duration_seconds)
        Raises: TimeoutError, ContainerError
        """
    
    @abstractmethod
    def forward_port(self, container_port: int) -> int:
        """
        Make container port accessible from caller.
        
        Backend behavior:
        - Local: Direct port mapping (returns host port)
        - Modal: Extracts port from Modal's public URL
        - Toolkit: SSH tunnel through HPC head node
        
        Returns: Local port number that forwards to container_port
        Raises: PortNotExposed, PortInUse
        """
    
    @abstractmethod
    def get_url(self, container_port: int) -> str:
        """
        Get URL to access container port.
        
        Returns: http://localhost:PORT or https://xyz.modal.run
        """
    
    @abstractmethod  
    def stop(self, timeout: int = 10):
        """
        Stop container gracefully.
        
        Cleanup: Stop process, close SSH tunnels, release ports, remove container
        """
    
    @abstractmethod
    def get_status(self) -> ContainerStatus:
        """Check container health for debugging."""
    
    @property
    @abstractmethod
    def id(self) -> str:
        """Unique container identifier (backend-specific format)."""
    
    @property
    @abstractmethod  
    def backend(self) -> str:
        """Backend type: 'local', 'modal', 'toolkit'."""


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int
    duration_seconds: float


@dataclass
class ContainerStatus:
    running: bool
    healthy: bool  # Based on health_check if provided
    resource_usage: Dict[str, float]  # cpu_percent, ram_mb, etc
    backend_info: Dict[str, Any]  # Backend-specific details
```

### Backend Implementations

Each backend subclasses `Container`:

```python
class LocalContainer(Container):
    """Uses docker CLI/docker-py. Direct port mapping."""

class ModalContainer(Container):
    """Uses Modal API. HTTP RPC communication. Public URLs."""

class ToolkitContainer(Container):
    """Uses EAI Toolkit + SLURM. SSH tunnels for port forwarding."""
```

## Port Forwarding Strategy

Port forwarding handles containers on remote machines (HPC compute nodes, cloud VMs).

**User code (same for all backends):**
```python
container = config.make()
local_port = container.forward_port(8080)
response = requests.get(f"http://localhost:{local_port}/api")
```

**Backend implementations:**
- **Local:** Direct port mapping via Docker (`-p host_port:container_port`)
- **Modal:** No traditional forwarding - Modal provides public URLs. `forward_port()` extracts port number, `get_url()` returns full URL.
- **Toolkit:** Most complex - Container runs on compute node (e.g., `node042:8080`). Creates SSH tunnel: `localhost:random_port → head_node → node042:8080`. Returns `random_port` to user.

## Error Handling

**Fail-fast with rich diagnostics.** Error messages must indicate precisely where startup failed (job submission, queue wait, container start, port access, health check).

**Cleanup on failure.** If `make()` fails partway, must clean up: cancel jobs, stop containers, close SSH connections, release ports.

**Optional health checks.** Beyond "container running", validate it's truly ready (e.g., database accepting connections). If health check fails, `make()` raises `HealthCheckError` after cleanup.

## Backend-Specific Considerations

### LocalContainer
- Uses docker CLI or docker-py
- Startup: 5-30 seconds
- Port forwarding: Native Docker port mapping
- Health check: Poll `docker ps` until healthy

### ModalContainer
- Uses Modal API
- Startup: 30-120 seconds (cold) or 5 seconds (warm)
- Communication: HTTP RPC (not traditional exec)
- Port forwarding: Modal provides public HTTPS URLs
- No SSH needed

### ToolkitContainer
- Uses EAI Toolkit + SLURM
- Startup: 5-30 minutes (queue dependent)
- Communication: SSH to compute node + docker exec
- Port forwarding: SSH tunnel through head node
- Health check: SSH connection + docker ps
- Cleanup: Cancel SLURM job + remove container

## Open Questions for Review

1. **Async vs sync `make()`?**
   - Current: Sync (Ray workers can block)
   - Alternative: Provide both `make()` and `amake()`

2. **Context manager support?**
   - Would `with config.make() as container:` work with Ray?
   - Containers can't be serialized, so probably not

3. **Registry authentication?**
   - Current: Assume images already available or auth in env vars
   - Risk: Don't want credentials in serialized configs

4. **Port collision handling?**
   - Current: Auto-retry with different port
   - Alternative: Fail and require explicit port range

5. **Resource limit enforcement?**
   - Pass to backend and trust it?
   - Or monitor and kill containers that exceed limits?

## Success Criteria

The design succeeds if:
1. CUBE-Developer can swap backends by changing one parameter
2. Ray parallelization works without modification
3. Port forwarding is transparent across backends
4. Error messages clearly indicate failure point
5. No resource leaks on failures
6. Toolkit/HPC works despite complexity (SSH tunnels, SLURM, etc)

