# Dual API Architecture Plan: Python + JSON-RPC/HTML

## Executive Summary

Implement a dual-mode API system allowing CUBE benchmarks to be used via:
1. **Python API** - Direct in-process function calls for high-performance evaluation
2. **JSON-RPC/HTTP API** - Remote task execution via FastAPI servers for distributed evaluation, sandboxing, and cloud deployment

**Key Architecture**: Two-tier server model with benchmark server (manages tasks) and task servers (one subprocess per task). Users enable server mode via `server_mode=True` flag in `setup()`.

## Design Decisions (Based on User Input)

1. **Mode Selection**: Flag in `setup()` - users pass `server_mode=True` to enable HTTP API
2. **Server Framework**: FastAPI with uvicorn (async, auto-docs, type validation)
3. **Task Isolation**: Subprocess per task (true isolation, resource limits, sandboxing)
4. **Client Library**: Document HTTP API only (no Python client wrapper for now)

## Current Architecture Analysis

### What Already Exists ✅
- **TaskSession** (src/cube/task.py): Concrete implementation of all task-level APIs
  - MCP protocol: `list_tools()`, `call_tool()`, `list_resources()`, `read_resource()`
  - CUBE extensions: `evaluate()`, `step()`, `reset()`, `close()`
- **Task** (src/cube/task.py): Abstract base class for tasks
- **Benchmark** (src/cube/benchmark.py): Abstract base with benchmark-level methods
- **Environment** (src/cube/environment.py): Wraps Task + Tool
- **Type system** (src/cube/types.py): Pydantic models for most APIs

### What's Missing ❌
1. **Server infrastructure**: No FastAPI servers exist yet
2. **Session management**: No tracking of spawned task servers
3. **Benchmark spawn/shutdown implementations**: Currently abstract methods

### What's Already Perfect ✅
- **MCP type integration**: TaskSession already uses MCP types directly (`MCPListToolsResult`, `MCPCallToolRequest`, etc.) - this is the correct approach! No wrapper types needed.

## Design Decision: Use MCP Types Directly

**Question**: Should we create CUBE wrapper types (e.g., `ToolListResponse`) or use MCP types directly?

**Answer**: **Use MCP types directly** (already implemented in current codebase)

### Rationale

**Advantages of using MCP types:**
1. **No duplication** - DRY principle, avoids maintaining duplicate type definitions
2. **Automatic MCP compatibility** - Guaranteed compatibility with MCP clients by definition
3. **Less maintenance** - When MCP updates, just update the dependency
4. **Cleaner codebase** - Less code to maintain and test
5. **Semantically correct** - CUBE implements the MCP protocol, so using MCP types is appropriate

**When to use MCP types vs CUBE types:**
- ✅ **Use MCP types**: Protocol boundary (tool calls, resources, MCP request/response types)
  - Examples: `MCPListToolsResult`, `MCPCallToolRequest`, `MCPCallToolResult`, `MCPReadResourceResult`
- ✅ **Use CUBE types**: Internal domain model with different semantics from MCP
  - Examples: `Action`, `Observation`, `Content`, `EnvironmentOutput` (CUBE's task evaluation model)
  - Examples: `StepRequest`, `ResetRequest`, `CloseResponse` (CUBE extensions to MCP)

**Current implementation** (already correct):
```python
# In src/cube/types.py - Import MCP types with MCP prefix for clarity
from mcp.types import (
    CallToolRequest as MCPCallToolRequest,
    CallToolResult as MCPCallToolResult,
    ListToolsResult as MCPListToolsResult,
    # ... etc
)

# In src/cube/task.py - TaskSession uses MCP types directly
def list_tools(self) -> MCPListToolsResult:
    return MCPListToolsResult(tools=filtered_actions)

def call_tool(self, request: MCPCallToolRequest) -> MCPCallToolResult:
    return MCPCallToolResult(content=mcp_content, isError=False)
```

This approach maintains a clean separation: MCP types for protocol boundaries, CUBE types for domain logic.

## Design Decision: TaskSession vs MCP Server Class

**Question**: Should we use MCP's `Server` class (decorator-based) or keep CUBE's `TaskSession` class?

**Answer**: **Hybrid approach** - Keep `TaskSession` for domain logic, support multiple transport layers

### Analysis

**MCP Server** (from `mcp.server.lowlevel.server`):
- Decorator-based MCP server implementation
- Built-in support for stdio, SSE transports
- Focused on MCP protocol only

**CUBE TaskSession**:
- Class-based session management
- Contains CUBE-specific logic (evaluate, step, reset, profiling)
- Environment and task lifecycle management
- Implements MCP protocol methods + CUBE extensions

### Recommended Architecture

```
┌──────────────────────────────────────────────────┐
│    TaskSession (Core Domain Logic)               │
│  - Environment management                        │
│  - MCP protocol: list_tools, call_tool,         │
│    list_resources, read_resource                │
│  - CUBE extensions: evaluate, step, reset, close│
│  - Session state: step_count, profiling         │
└────────────────────┬─────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          │                     │
┌─────────▼──────────┐  ┌──────▼────────────┐
│  FastAPI Server    │  │   MCP Server      │
│  (HTTP/JSON-RPC)   │  │   (stdio/SSE)     │
│  For distributed   │  │   For native MCP  │
│  evaluation        │  │   clients         │
└────────────────────┘  └───────────────────┘
```

### Why This Approach?

1. **TaskSession is more than MCP**:
   - Contains CUBE-specific business logic
   - Manages environment lifecycle
   - Tracks session state and profiling
   - Has extensions beyond MCP spec

2. **Supports multiple transports**:
   - HTTP/JSON-RPC (this plan) - for distributed evaluation, cloud deployment
   - Native MCP stdio (future) - for direct MCP client integration
   - Future: WebSocket, gRPC, etc.

3. **Separation of concerns**:
   - TaskSession = domain logic (what to do)
   - Transport layers = protocol handling (how to communicate)

4. **Flexibility**:
   - Can add new transports without changing TaskSession
   - Can use TaskSession directly in Python without any transport

### Implementation Strategy

**Phase 1-4 (This Plan)**: FastAPI transport for HTTP/JSON-RPC
**Future (Optional)**: MCP Server transport for native stdio/SSE

Both transports call the same TaskSession methods, ensuring consistency.

## Implementation Plan

### Phase 1: Create Server Infrastructure

#### 1.1 Session Manager

**File**: [src/cube/server/session_manager.py](src/cube/server/session_manager.py) (new)

```python
"""Manages lifecycle of task server subprocesses."""
import multiprocessing
import uuid
from datetime import datetime
from typing import Dict

from cube.types import (
    SpawnRequest, SpawnResponse, StatusRequest, StatusResponse,
    ShutdownRequest, ShutdownResponse, TaskStatus, TaskStatusEnum
)
from cube.benchmark import Benchmark
from cube.environment import EnvConfig


class TaskServerProcess:
    """Represents a running task server subprocess."""
    def __init__(self, session_id: str, task_id: str, port: int, process: multiprocessing.Process):
        self.session_id = session_id
        self.task_id = task_id
        self.port = port
        self.process = process
        self.created_at = datetime.now()
        self.step_count = 0


class SessionManager:
    """
    Manages spawned task server subprocesses.

    Responsibilities:
    - Allocate ports from available pool
    - Spawn task server subprocesses
    - Track active sessions
    - Report status
    - Shutdown and cleanup
    """

    def __init__(self, benchmark: Benchmark, available_ports: list[int], host: str = "localhost"):
        self.benchmark = benchmark
        self.host = host
        self.available_ports = list(available_ports)
        self.used_ports: list[int] = []
        self.active_sessions: Dict[str, TaskServerProcess] = {}

    def spawn(self, request: SpawnRequest) -> SpawnResponse:
        """
        Spawn a new task server subprocess.

        Steps:
        1. Find task by ID from benchmark.load_tasks()
        2. Create EnvConfig and Environment
        3. Create TaskSession
        4. Allocate port from pool
        5. Start FastAPI server in subprocess
        6. Return URL and session_id
        """
        # Generate session ID
        session_id = str(uuid.uuid4())

        # Get port from pool
        if not self.available_ports:
            raise RuntimeError("No available ports for task server")
        port = self.available_ports.pop(0)
        self.used_ports.append(port)

        # Find task
        tasks = self.benchmark.load_tasks()
        task = next((t for t in tasks if t.id == request.task_id), None)
        if not task:
            raise ValueError(f"Task {request.task_id} not found")

        # Create environment config
        env_config = EnvConfig(task=task, tool_config=self.benchmark.tool_config)

        # Start task server in subprocess
        process = multiprocessing.Process(
            target=_run_task_server,
            args=(session_id, request.task_id, env_config, self.host, port, request.seed)
        )
        process.start()

        # Track session
        server_process = TaskServerProcess(session_id, request.task_id, port, process)
        self.active_sessions[session_id] = server_process

        # Update task status
        task.status = TaskStatus(
            session_id=session_id,
            task_id=request.task_id,
            status=TaskStatusEnum.running,
            created_at=server_process.created_at,
            step_count=0,
            last_updated=None,
            other={}
        )

        return SpawnResponse(
            url=f"http://{self.host}:{port}",
            session_id=session_id,
            other={"task_id": request.task_id}
        )

    def get_status(self, request: StatusRequest) -> StatusResponse:
        """Get status of one or all task sessions."""
        if request.session_id:
            # Single session status
            if request.session_id not in self.active_sessions:
                return StatusResponse(tasks=[])

            server_proc = self.active_sessions[request.session_id]
            task = next((t for t in self.benchmark.load_tasks() if t.id == server_proc.task_id), None)

            if task and task.status:
                return StatusResponse(tasks=[task.status])
            return StatusResponse(tasks=[])

        # All sessions status
        all_statuses = []
        for session_id, server_proc in self.active_sessions.items():
            task = next((t for t in self.benchmark.load_tasks() if t.id == server_proc.task_id), None)
            if task and task.status:
                all_statuses.append(task.status)

        return StatusResponse(tasks=all_statuses)

    def shutdown(self, request: ShutdownRequest) -> ShutdownResponse:
        """Shutdown one or all task server subprocesses."""
        cleaned = []

        if request.session_id:
            # Shutdown single session
            if request.session_id in self.active_sessions:
                server_proc = self.active_sessions[request.session_id]
                server_proc.process.terminate()
                server_proc.process.join(timeout=5)

                # Return port to pool
                self.available_ports.append(server_proc.port)
                self.used_ports.remove(server_proc.port)

                del self.active_sessions[request.session_id]
                cleaned.append(request.session_id)
        else:
            # Shutdown all sessions
            for session_id, server_proc in list(self.active_sessions.items()):
                server_proc.process.terminate()
                server_proc.process.join(timeout=5)

                # Return port to pool
                self.available_ports.append(server_proc.port)
                self.used_ports.remove(server_proc.port)

                cleaned.append(session_id)

            self.active_sessions.clear()

        return ShutdownResponse(success=True, cleaned=cleaned)


def _run_task_server(session_id: str, task_id: str, env_config: EnvConfig,
                     host: str, port: int, seed: int | None):
    """
    Function run in subprocess to start task server.

    This function:
    1. Creates Environment from EnvConfig
    2. Calls env.reset(seed)
    3. Creates TaskSession
    4. Starts FastAPI task server
    """
    from cube.server.task_server import create_task_server_app
    import uvicorn

    # Create environment and reset
    env = env_config.make()
    env.reset()

    # Create task session
    from cube.task import TaskSession
    session = TaskSession(session_id=session_id, task_id=task_id, env=env)

    # Create FastAPI app
    app = create_task_server_app(session)

    # Run server (blocking)
    uvicorn.run(app, host=host, port=port, log_level="info")
```

**Purpose**: Manages the lifecycle of task server subprocesses, handles port allocation, tracks active sessions.

#### 1.2 Task Server (FastAPI)

**File**: [src/cube/server/task_server.py](src/cube/server/task_server.py) (new)

```python
"""FastAPI server exposing task-level APIs for a single task."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any

from cube.task import TaskSession, TaskClosedException, ResourceNotFoundException
from cube.types import (
    MCPListToolsResult, MCPCallToolRequest, MCPCallToolResult,
    MCPListResourcesResult, MCPReadResourceResult,
    StepRequest, StepResponse, ResetRequest, ResetResponse, CloseResponse
)


class JSONRPCRequest(BaseModel):
    """JSON-RPC 2.0 request format."""
    jsonrpc: str = "2.0"
    method: str
    params: dict[str, Any] | None = None
    id: str | int | None = None


class JSONRPCResponse(BaseModel):
    """JSON-RPC 2.0 response format."""
    jsonrpc: str = "2.0"
    result: Any | None = None
    error: dict[str, Any] | None = None
    id: str | int | None = None


def create_task_server_app(session: TaskSession) -> FastAPI:
    """
    Create FastAPI application for a single task session.

    Exposes all TaskSession methods as JSON-RPC endpoints:
    - MCP Protocol: tools/list, tools/call, resources/list, resources/read
    - CUBE Extensions: cube/evaluation, cube/step, cube/reset, cube/close
    """
    app = FastAPI(
        title=f"CUBE Task Server - {session.task_id}",
        description=f"Task-level API for session {session.session_id}",
        version="1.0.0"
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # =============================================================================
    # MCP Protocol Endpoints
    # =============================================================================

    @app.post("/tools/list")
    async def tools_list(request: JSONRPCRequest) -> JSONRPCResponse:
        """List available tools (MCP tools/list)."""
        try:
            response = session.list_tools()
            return JSONRPCResponse(
                result=response.model_dump(),
                id=request.id
            )
        except TaskClosedException as e:
            return JSONRPCResponse(
                error={"code": "SESSION_CLOSED", "message": str(e)},
                id=request.id
            )
        except Exception as e:
            return JSONRPCResponse(
                error={"code": "INTERNAL_ERROR", "message": str(e)},
                id=request.id
            )

    @app.post("/tools/call")
    async def tools_call(request: JSONRPCRequest) -> JSONRPCResponse:
        """Execute a tool (MCP tools/call)."""
        try:
            if not request.params:
                raise ValueError("Missing params for tools/call")

            tool_request = MCPCallToolRequest(**request.params)
            response = session.call_tool(tool_request)

            return JSONRPCResponse(
                result=response.model_dump(),
                id=request.id
            )
        except TaskClosedException as e:
            return JSONRPCResponse(
                error={"code": "SESSION_CLOSED", "message": str(e)},
                id=request.id
            )
        except Exception as e:
            return JSONRPCResponse(
                error={"code": "TOOL_EXECUTION_FAILED", "message": str(e)},
                id=request.id
            )

    @app.post("/resources/list")
    async def resources_list(request: JSONRPCRequest) -> JSONRPCResponse:
        """List available resources (MCP resources/list)."""
        try:
            response = session.list_resources()
            return JSONRPCResponse(
                result=response.model_dump(),
                id=request.id
            )
        except TaskClosedException as e:
            return JSONRPCResponse(
                error={"code": "SESSION_CLOSED", "message": str(e)},
                id=request.id
            )
        except Exception as e:
            return JSONRPCResponse(
                error={"code": "INTERNAL_ERROR", "message": str(e)},
                id=request.id
            )

    @app.post("/resources/read")
    async def resources_read(request: JSONRPCRequest) -> JSONRPCResponse:
        """Read a specific resource (MCP resources/read)."""
        try:
            if not request.params or "uri" not in request.params:
                raise ValueError("Missing 'uri' in params for resources/read")

            uri = request.params["uri"]
            response = session.read_resource(uri)

            return JSONRPCResponse(
                result=response.model_dump(),
                id=request.id
            )
        except ResourceNotFoundException as e:
            return JSONRPCResponse(
                error={"code": "RESOURCE_NOT_FOUND", "message": str(e)},
                id=request.id
            )
        except TaskClosedException as e:
            return JSONRPCResponse(
                error={"code": "SESSION_CLOSED", "message": str(e)},
                id=request.id
            )
        except Exception as e:
            return JSONRPCResponse(
                error={"code": "INTERNAL_ERROR", "message": str(e)},
                id=request.id
            )

    # =============================================================================
    # CUBE Extension Endpoints
    # =============================================================================

    @app.post("/cube/evaluation")
    async def cube_evaluation(request: JSONRPCRequest) -> JSONRPCResponse:
        """Get current evaluation state (CUBE cube/evaluation)."""
        try:
            response = session.evaluate()
            return JSONRPCResponse(
                result=response.model_dump(),
                id=request.id
            )
        except TaskClosedException as e:
            return JSONRPCResponse(
                error={"code": "SESSION_CLOSED", "message": str(e)},
                id=request.id
            )
        except Exception as e:
            return JSONRPCResponse(
                error={"code": "INTERNAL_ERROR", "message": str(e)},
                id=request.id
            )

    @app.post("/cube/step")
    async def cube_step(request: JSONRPCRequest) -> JSONRPCResponse:
        """Execute tool and get evaluation (CUBE cube/step)."""
        try:
            if not request.params:
                raise ValueError("Missing params for cube/step")

            step_request = StepRequest(**request.params)
            response = session.step(step_request)

            return JSONRPCResponse(
                result=response.model_dump(),
                id=request.id
            )
        except TaskClosedException as e:
            return JSONRPCResponse(
                error={"code": "SESSION_CLOSED", "message": str(e)},
                id=request.id
            )
        except Exception as e:
            return JSONRPCResponse(
                error={"code": "STEP_FAILED", "message": str(e)},
                id=request.id
            )

    @app.post("/cube/reset")
    async def cube_reset(request: JSONRPCRequest) -> JSONRPCResponse:
        """Reset task to initial state (CUBE cube/reset)."""
        try:
            params = request.params or {}
            reset_request = ResetRequest(**params)
            response = session.reset(reset_request)

            return JSONRPCResponse(
                result=response.model_dump(),
                id=request.id
            )
        except TaskClosedException as e:
            return JSONRPCResponse(
                error={"code": "SESSION_CLOSED", "message": str(e)},
                id=request.id
            )
        except Exception as e:
            return JSONRPCResponse(
                error={"code": "RESET_FAILED", "message": str(e)},
                id=request.id
            )

    @app.post("/cube/close")
    async def cube_close(request: JSONRPCRequest) -> JSONRPCResponse:
        """Close task session (CUBE cube/close)."""
        try:
            response = session.close()
            return JSONRPCResponse(
                result=response.model_dump(),
                id=request.id
            )
        except Exception as e:
            return JSONRPCResponse(
                error={"code": "CLOSE_FAILED", "message": str(e)},
                id=request.id
            )

    return app
```

**Purpose**: FastAPI server that wraps a TaskSession and exposes all methods as JSON-RPC 2.0 endpoints.

#### 1.3 Benchmark Server (FastAPI)

**File**: [src/cube/server/benchmark_server.py](src/cube/server/benchmark_server.py) (new)

```python
"""FastAPI server exposing benchmark-level APIs."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any

from cube.benchmark import Benchmark
from cube.types import (
    BenchmarkMetadata, TaskRequest, TaskListResponse,
    SpawnRequest, SpawnResponse, StatusRequest, StatusResponse,
    ShutdownRequest, ShutdownResponse
)
from cube.server.session_manager import SessionManager


class JSONRPCRequest(BaseModel):
    """JSON-RPC 2.0 request format."""
    jsonrpc: str = "2.0"
    method: str
    params: dict[str, Any] | None = None
    id: str | int | None = None


class JSONRPCResponse(BaseModel):
    """JSON-RPC 2.0 response format."""
    jsonrpc: str = "2.0"
    result: Any | None = None
    error: dict[str, Any] | None = None
    id: str | int | None = None


def create_benchmark_server_app(benchmark: Benchmark, session_manager: SessionManager) -> FastAPI:
    """
    Create FastAPI application for benchmark-level operations.

    Exposes benchmark-level endpoints:
    - cube/info - Get benchmark metadata
    - cube/tasks - List available tasks
    - cube/spawn - Spawn new task server
    - cube/status - Get task session status
    - cube/shutdown - Shutdown task sessions
    """
    app = FastAPI(
        title=f"CUBE Benchmark Server - {benchmark.metadata.name}",
        description=benchmark.metadata.description,
        version=benchmark.metadata.version
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/cube/info")
    async def cube_info(request: JSONRPCRequest) -> JSONRPCResponse:
        """Get benchmark metadata (CUBE cube/info)."""
        try:
            metadata = benchmark.info()
            return JSONRPCResponse(
                result=metadata.model_dump(),
                id=request.id
            )
        except Exception as e:
            return JSONRPCResponse(
                error={"code": "INTERNAL_ERROR", "message": str(e)},
                id=request.id
            )

    @app.post("/cube/tasks")
    async def cube_tasks(request: JSONRPCRequest) -> JSONRPCResponse:
        """List available tasks (CUBE cube/tasks)."""
        try:
            params = request.params or {}
            task_request = TaskRequest(**params)
            response = benchmark.list_tasks(task_request)

            return JSONRPCResponse(
                result=response.model_dump(),
                id=request.id
            )
        except Exception as e:
            return JSONRPCResponse(
                error={"code": "INTERNAL_ERROR", "message": str(e)},
                id=request.id
            )

    @app.post("/cube/spawn")
    async def cube_spawn(request: JSONRPCRequest) -> JSONRPCResponse:
        """Spawn a new task server (CUBE cube/spawn)."""
        try:
            if not request.params:
                raise ValueError("Missing params for cube/spawn")

            spawn_request = SpawnRequest(**request.params)
            response = session_manager.spawn(spawn_request)

            return JSONRPCResponse(
                result=response.model_dump(),
                id=request.id
            )
        except ValueError as e:
            return JSONRPCResponse(
                error={"code": "INVALID_TASK", "message": str(e)},
                id=request.id
            )
        except RuntimeError as e:
            return JSONRPCResponse(
                error={"code": "RESOURCE_UNAVAILABLE", "message": str(e)},
                id=request.id
            )
        except Exception as e:
            return JSONRPCResponse(
                error={"code": "SPAWN_FAILED", "message": str(e)},
                id=request.id
            )

    @app.post("/cube/status")
    async def cube_status(request: JSONRPCRequest) -> JSONRPCResponse:
        """Get task session status (CUBE cube/status)."""
        try:
            params = request.params or {}
            status_request = StatusRequest(**params)
            response = session_manager.get_status(status_request)

            return JSONRPCResponse(
                result=response.model_dump(),
                id=request.id
            )
        except Exception as e:
            return JSONRPCResponse(
                error={"code": "INTERNAL_ERROR", "message": str(e)},
                id=request.id
            )

    @app.post("/cube/shutdown")
    async def cube_shutdown(request: JSONRPCRequest) -> JSONRPCResponse:
        """Shutdown task sessions (CUBE cube/shutdown)."""
        try:
            params = request.params or {}
            shutdown_request = ShutdownRequest(**params)
            response = session_manager.shutdown(shutdown_request)

            return JSONRPCResponse(
                result=response.model_dump(),
                id=request.id
            )
        except Exception as e:
            return JSONRPCResponse(
                error={"code": "SHUTDOWN_FAILED", "message": str(e)},
                id=request.id
            )

    return app
```

**Purpose**: FastAPI server that exposes benchmark-level operations (info, tasks, spawn, status, shutdown).

#### 1.4 Server Package Init

**File**: [src/cube/server/__init__.py](src/cube/server/__init__.py) (new)

```python
"""CUBE Server Infrastructure."""

from cube.server.benchmark_server import create_benchmark_server_app
from cube.server.task_server import create_task_server_app
from cube.server.session_manager import SessionManager

__all__ = [
    "create_benchmark_server_app",
    "create_task_server_app",
    "SessionManager"
]
```

### Phase 2: Integrate Server with Benchmark Class

**File**: [src/cube/benchmark.py](src/cube/benchmark.py)

Modify the Benchmark class to support server mode:

```python
# Add import at top
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from cube.server.session_manager import SessionManager

class Benchmark(TypedBaseModel, ABC):
    """Represents a benchmark consisting of multiple tasks and an environment."""

    metadata: BenchmarkMetadata
    tool_config: ToolConfig
    _session_manager: "SessionManager | None" = None  # Private field for server mode
    _server_process: Any = None  # Server thread/process

    @abstractmethod
    def setup(
        self,
        available_ports: list[int],
        tool_config: ToolConfig,
        server_mode: bool = False,  # NEW: Enable server mode
        server_host: str = "localhost",  # NEW: Server host
        server_port: int = 8000  # NEW: Benchmark server port
    ) -> str | None:
        """
        Perform common steps necessary to prepare the environment for all tasks.

        Args:
            available_ports: List of ports available for task servers
            tool_config: Tool configuration
            server_mode: If True, start benchmark server
            server_host: Host for servers
            server_port: Port for benchmark server

        Returns:
            If server_mode=True, returns benchmark server URL (e.g., "http://localhost:8000")
            Otherwise returns None
        """
        self.tool_config = tool_config

        if server_mode:
            # Start benchmark server
            from cube.server import SessionManager, create_benchmark_server_app
            import uvicorn
            import threading

            # Create session manager
            self._session_manager = SessionManager(
                benchmark=self,
                available_ports=available_ports,
                host=server_host
            )

            # Create FastAPI app
            app = create_benchmark_server_app(self, self._session_manager)

            # Start server in background thread
            def run_server():
                uvicorn.run(app, host=server_host, port=server_port, log_level="info")

            self._server_process = threading.Thread(target=run_server, daemon=True)
            self._server_process.start()

            return f"http://{server_host}:{server_port}"

        return None

    # Make spawn() and other methods concrete using session_manager
    def spawn(self, request: SpawnRequest) -> SpawnResponse:
        """
        Spawn a new task session.

        If server_mode=True, spawns task server subprocess.
        Otherwise, creates TaskSession directly.
        """
        if self._session_manager:
            # Server mode: spawn subprocess
            return self._session_manager.spawn(request)
        else:
            # Python mode: create TaskSession directly
            # (Benchmark authors can override for custom behavior)
            raise NotImplementedError(
                "Python mode spawn() must be implemented by benchmark author, "
                "or use server_mode=True in setup()"
            )

    def get_task_status(self, request: StatusRequest) -> StatusResponse:
        """Get status of task sessions."""
        if self._session_manager:
            return self._session_manager.get_status(request)
        else:
            raise NotImplementedError(
                "Python mode get_task_status() must be implemented by benchmark author, "
                "or use server_mode=True in setup()"
            )

    def shutdown(self, request: ShutdownRequest) -> ShutdownResponse:
        """Shutdown task sessions."""
        if self._session_manager:
            return self._session_manager.shutdown(request)
        else:
            raise NotImplementedError(
                "Python mode shutdown() must be implemented by benchmark author, "
                "or use server_mode=True in setup()"
            )
```

**Key Change**: Added `server_mode` parameter to `setup()`. When enabled, starts benchmark server and provides concrete implementations of spawn/status/shutdown via SessionManager.

### Phase 3: Update Package Exports

**File**: [src/cube/__init__.py](src/cube/__init__.py)

Add server exports:

```python
# Add at end of file
from cube.server import (
    create_benchmark_server_app,
    create_task_server_app,
    SessionManager
)
```

### Phase 4: Add Dependencies

**File**: [pyproject.toml](pyproject.toml)

Add FastAPI dependencies:

```toml
[project]
dependencies = [
    # ... existing dependencies ...
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
]
```

## Usage Examples

### Python API Mode (Direct)

```python
from my_cube import MyBenchmark

# Instantiate benchmark
bench = MyBenchmark()

# Setup (Python mode - no server)
bench.setup(
    available_ports=list(range(8000, 8100)),
    tool_config=MyToolConfig(),
    server_mode=False  # Python API mode
)

# Create task session directly (requires custom spawn implementation)
response = bench.spawn(SpawnRequest(task_id="task-1", seed=42))
session = response.session  # TaskSession instance

# Use Python API
tools = session.list_tools()  # Returns MCPListToolsResult
result = session.call_tool(MCPCallToolRequest(name="click", arguments={"x": 100}))  # Returns MCPCallToolResult
state = session.evaluate()
```

### JSON-RPC/HTTP API Mode (Remote)

```python
from my_cube import MyBenchmark

# Instantiate benchmark
bench = MyBenchmark()

# Setup (Server mode)
endpoint = bench.setup(
    available_ports=list(range(8001, 8100)),
    tool_config=MyToolConfig(),
    server_mode=True,  # Enable server mode
    server_host="localhost",
    server_port=8000
)
# Returns: "http://localhost:8000"

# Benchmark server is now running at http://localhost:8000
# Can call benchmark-level APIs via HTTP:

import requests

# Get benchmark info
response = requests.post(
    "http://localhost:8000/cube/info",
    json={"jsonrpc": "2.0", "method": "cube/info", "params": {}, "id": 1}
)
print(response.json())  # {"result": {"name": "...", "version": "..."}}

# Spawn a task server
response = requests.post(
    "http://localhost:8000/cube/spawn",
    json={
        "jsonrpc": "2.0",
        "method": "cube/spawn",
        "params": {"task_id": "task-1", "seed": 42},
        "id": 2
    }
)
task_url = response.json()["result"]["url"]  # "http://localhost:8001"

# Task server is now running, call task-level APIs:
response = requests.post(
    f"{task_url}/tools/list",
    json={"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": 3}
)
tools = response.json()["result"]["tools"]

# Call a tool
response = requests.post(
    f"{task_url}/tools/call",
    json={
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {"name": "click", "arguments": {"x": 100, "y": 200}},
        "id": 4
    }
)
result = response.json()["result"]
```

## Critical Files to Modify/Create

1. **[src/cube/server/session_manager.py](src/cube/server/session_manager.py)** - Create session management (NEW)
2. **[src/cube/server/task_server.py](src/cube/server/task_server.py)** - Create task FastAPI server (NEW)
3. **[src/cube/server/benchmark_server.py](src/cube/server/benchmark_server.py)** - Create benchmark FastAPI server (NEW)
4. **[src/cube/server/__init__.py](src/cube/server/__init__.py)** - Server package exports (NEW)
5. **[src/cube/benchmark.py](src/cube/benchmark.py)** - Add server_mode to setup(), integrate SessionManager
6. **[pyproject.toml](pyproject.toml)** - Add FastAPI and uvicorn dependencies

## Verification Plan

### Test 1: Python API Mode
```bash
# Create test benchmark
python examples/test_python_mode.py

# Expected: Direct TaskSession usage works, all methods callable
```

### Test 2: HTTP Server Mode
```bash
# Start benchmark in server mode
python examples/test_server_mode.py

# Expected: Benchmark server starts on port 8000
# Open browser: http://localhost:8000/docs (FastAPI auto-docs)
```

### Test 3: End-to-End RPC Flow
```bash
# Run full workflow test
python examples/test_rpc_workflow.py

# Expected:
# 1. Benchmark server starts
# 2. Call cube/info via HTTP - gets metadata
# 3. Call cube/spawn via HTTP - gets task server URL
# 4. Call tools/list on task server - gets tools
# 5. Call tools/call on task server - executes action
# 6. Call cube/evaluation - gets state
# 7. Call cube/close - cleans up
```

### Test 4: Subprocess Isolation
```bash
# Verify task servers run in separate processes
python examples/test_isolation.py

# Expected: ps aux shows multiple Python processes (benchmark + task servers)
```

### Test 5: MCP Type Compatibility
```bash
# Verify MCP clients can connect
python examples/test_mcp_client.py

# Expected: MCP protocol methods work correctly
```

## Performance Expectations

- **Python API**: ~0.1ms per action (direct function call)
- **RPC API**: ~1-5ms per action (HTTP + JSON serialization)
- **Task spawn**: ~50-100ms (subprocess creation)
- **Memory**: ~50MB per task server subprocess

## Architecture Benefits

1. **Same Implementation**: TaskSession used by both Python and RPC modes
2. **Type Safety**: Pydantic models ensure validation on both sides
3. **Isolation**: Subprocesses prevent cross-task interference
4. **Scalability**: Task servers can run on different machines
5. **Debugging**: FastAPI auto-docs at /docs endpoint
6. **Backward Compatible**: Existing Python code continues to work

## Future Enhancements (Out of Scope)

1. **Client Library**: CubeTaskClient for transparent remote access
2. **Thread Mode**: Option for faster but less isolated task servers
3. **Docker Support**: Run task servers in containers
4. **WebSocket Support**: For streaming observations
5. **gRPC Support**: Alternative to JSON-RPC for better performance
