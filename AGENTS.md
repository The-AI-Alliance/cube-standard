# CUBE Standard - Project Structure for Coding Agents

CUBE (Common Unified Benchmark Environments) is a standard library for defining and running agent benchmarks. It provides core abstractions for tasks, tools, environments, and a Container API for launching isolated execution environments across multiple backends.

## Directory Structure

```
cube-standard/
├── src/cube/                   # Core framework source code
│   ├── base.py                 # TypedBaseModel for serialization with type info
│   ├── core.py                 # Data structures: Action, Observation, Task
│   ├── tool.py                 # Tool abstraction for action spaces
│   ├── environment.py          # Environment and EnvConfig abstractions
│   ├── benchmark.py            # Benchmark interface for task collections
│   ├── container.py            # Container API: ContainerSpec, Container, ContainerBackend
│   └── backends/               # Container backend implementations
│       ├── __init__.py          # Re-exports all backends
│       ├── local.py             # LocalContainerBackend (docker-py)
│       ├── daytona.py           # DaytonaContainerBackend (Daytona SDK)
│       ├── modal.py             # ModalContainerBackend (Modal Sandbox)
│       └── toolkit.py           # ToolkitContainerBackend (HPC/SLURM stub)
├── scripts/                    # Integration test scripts
│   ├── test_harness.py          # Shared test logic for all backends
│   ├── test_local.py            # Docker backend tests
│   ├── test_daytona.py          # Daytona backend tests
│   └── test_modal.py            # Modal backend tests
├── design/                     # Design documents
│   ├── docker_wrapper.md        # Container API design spec
│   ├── main_specs.md            # Core CUBE specs
│   └── ...
└── pyproject.toml              # Project config and dependencies
```

## Core Abstractions

### Module Dependencies

```
base.py (TypedBaseModel for serialization)
    ↑
core.py (Action, Observation, Task, Content, ActionSchema)
    ↑
tool.py (Tool, ToolConfig, AbstractTool)
    ↑
environment.py (Environment, EnvConfig - composes Task + Tool)
    ↑
benchmark.py (Benchmark - task collections with tool_config)

container.py (ContainerSpec, Container, ContainerBackend - independent module)
    ↑
backends/ (Local, Daytona, Modal, Toolkit implementations)
```

## Key Classes

### base.py
- **TypedBaseModel**: Pydantic base that serializes/deserializes with `_type` field for polymorphism

### core.py
- **ActionSchema**: Function specification for LLM tool calls
- **Action**: Function call with id, name, arguments
- **Content**: Piece of content in an observation (text or bytes)
- **Observation**: List of Contents, created via `Observation.from_text()`
- **EnvironmentOutput**: Result of env step (obs, reward, done, info)
- **Task**: Abstract task with `setup()`, `validate_task()`, `filter_actions()`

### tool.py
- **ToolConfig**: Abstract config with `make() -> AbstractTool`
- **Tool**: Protocol-based implementation using `action_space` attribute
- **AbstractTool**: Abstract base with `execute_action()`, `get_actions()`, `close()`

### environment.py
- **EnvConfig**: Runtime config (task + tool_config), has `make() -> Environment`
- **Environment**: Composes Task + Tool, implements `reset()`, `step()`, `close()`
- **STOP_ACTION**: Special action to signal task completion

### benchmark.py
- **Benchmark**: Abstract with `setup()`, `close()`, `load_tasks()`, `env_configs()`

### container.py - Container API
- **ContainerSpec**: Dataclass defining *what* to run (image, ram_gb, cpu_cores, gpu, disk_gb, ports)
- **Container**: ABC for running container (exec, forward_port, get_url, stop, get_status, id)
- **ContainerBackend**: TypedBaseModel ABC for *how* to run (launch, timeout, health_check)
- **ExecResult**: Dataclass (stdout, stderr, exit_code, duration_seconds)
- **ContainerStatus**: Dataclass (running, healthy, resource_usage, backend_info)
- Exceptions: `ContainerError`, `ContainerLaunchError`, `HealthCheckError`, `ContainerExecError`

### backends/
- **LocalContainerBackend**: Docker-py, direct port mapping, `pull_policy`, `network_mode`
- **DaytonaContainerBackend**: Daytona SDK, session-based exec, signed preview URLs
- **ModalContainerBackend**: Modal Sandbox, tunnel-based URLs, encrypted ports
- **ToolkitContainerBackend**: HPC/SLURM stub (not yet implemented)

## Container API Pattern

```python
from cube.container import ContainerSpec
from cube.backends.local import LocalContainerBackend

# 1. User defines backend once
backend = LocalContainerBackend(timeout_seconds=60)

# 2. Benchmark defines spec per task
spec = ContainerSpec(image="python:3.12-slim", ram_gb=4, ports=[8080])

# 3. Launch and use
container = backend.launch(spec)
result = container.exec("echo hello")
url = container.get_url(8080)
container.stop()
```

## Development Commands

```bash
make install   # uv sync + pip install -e .
make format    # Format code with Ruff
make lint      # Lint and auto-fix with Ruff

# Integration tests (real backends, not mocks)
PYTHONPATH=scripts uv run python scripts/test_local.py     # Requires Docker
PYTHONPATH=scripts uv run python scripts/test_daytona.py   # Requires DAYTONA_API_KEY
PYTHONPATH=scripts uv run python scripts/test_modal.py     # Requires Modal token
```

## Project Configuration

- **Package manager**: `uv`
- **Python**: >= 3.12
- **Source layout**: `src/cube/`
- **Linter/formatter**: Ruff
- **Optional deps**: `docker`, `daytona`, `modal` (install with `uv pip install -e ".[docker,daytona,modal]"`)

## Development Notes

- All imports at the top of the module, never inside functions or classes
- Use `sh` not `bash` for container exec (POSIX compatibility, e.g. Alpine)
- Container backends use `tenacity` retry decorators for transient failures
- `ContainerBackend` is serializable (Pydantic + TypedBaseModel), `Container` is not
- `health_check: Callable` is excluded from serialization via `Field(exclude=True)`
- Test scripts share logic via `test_harness.py` — backend-specific tests append to the shared list
