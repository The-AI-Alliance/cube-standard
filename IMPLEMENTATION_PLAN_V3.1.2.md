# Benchmark-Level Server API Implementation Plan

## Executive Summary

**Scope**: Implement **benchmark-level server infrastructure only** (Phase 1)

This plan focuses on:
1. ✅ Benchmark server (FastAPI) exposing cube/info, cube/tasks, cube/spawn, cube/status, cube/shutdown
2. ✅ Session management for spawned task servers
3. ✅ `spawn()` creates minimal task servers (empty placeholder, no endpoints yet)
4. ❌ Task-level APIs (tools/*, resources/*, cube/*) - **deferred to Phase 2**

**Key Architecture**: Single benchmark server that manages task server lifecycle. Users enable server mode via `server_mode=True` flag in `setup()`. Task servers are spawned but don't implement endpoints yet (Phase 2).

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

1. **Benchmark server infrastructure**: No FastAPI server for benchmark-level endpoints
2. **Session management**: No tracking of spawned task servers
3. **Benchmark spawn/shutdown implementations**: Currently abstract methods
4. **Task server placeholder**: Need minimal server that spawned tasks can run (empty for now)

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

### Handling Name Collisions: MCP Task vs CUBE Task

**Question**: MCP has a `Task` class and CUBE has a `Task` class. Is this a problem?

**Answer**: **No problem** - they serve completely different purposes and can coexist

**MCP Task** (mcp.types.Task):
- **Purpose**: Metadata for task-augmented execution (async long-running operations)
- **Domain**: Protocol/transport layer
- **Use case**: When `tools/call` takes 5 minutes, return a task ID for client to poll
- **Fields**: `taskId`, `status` ("working"/"completed"/"failed"), `createdAt`, `lastUpdatedAt`, `ttl`, `pollInterval`

**CUBE Task** (cube.task.Task):
- **Purpose**: Benchmark evaluation task definition (a challenge for an agent to solve)
- **Domain**: Benchmark/evaluation domain
- **Use case**: Define challenges like "Book a flight from NYC to LAX for under $500"
- **Fields**: `id`, `seed`, `metadata`, abstract methods (`setup()`, `validate_task()`, `filter_actions()`)

**Key Differences**:
- **MCP Task**: Tracks progress of ONE async operation (lifetime: seconds to minutes)
- **CUBE Task**: Defines a benchmark challenge (lifetime: entire evaluation session)

**Naming Strategy**:
```python
# Import MCP Task with prefix for clarity
from mcp.types import Task as MCPTask

# CUBE Task remains as is
from cube.task import Task

# No confusion because:
# - MCPTask is used for async operation metadata (protocol layer)
# - Task is used for benchmark challenges (domain layer)
```

**Future Enhancement** (out of scope for this plan):
CUBE could support MCP's task-augmented execution by:
- Returning `CreateTaskResult(task=MCPTask(...))` for long-running operations
- Implementing `tasks/get` and `tasks/result` endpoints
- Useful for tool calls or steps that take >30 seconds

For now, CUBE operations are synchronous (immediate response).

## Design Decision: TaskSession for Both Modes (No Duplication)

**Question**: Won't having both Python mode and Server mode cause code duplication?

**Answer**: **No duplication** - TaskSession is used in BOTH modes!

### Single Source of Truth: TaskSession

**TaskSession** contains ALL task logic. Task Server is just a **thin HTTP wrapper**.

```
Python Mode (in-process):
bench.spawn() → Creates TaskSession → Direct access
                     ↓
                session.list_tools()  # Direct method call

Server Mode (subprocess):
bench.spawn() → Creates subprocess → Creates TaskSession → FastAPI wraps it
                                           ↓                    ↓
                                    session.list_tools()  HTTP /tools/list
                                                              ↓
                                                          calls session.list_tools()
```

### Architecture Layers

| Layer | Python Mode | Server Mode |
|-------|-------------|-------------|
| **Business Logic** | TaskSession (in-process) | TaskSession (in subprocess) |
| **Access Pattern** | Direct method calls | HTTP endpoints → method calls |
| **Transport** | None | FastAPI routes |

**Key insight:** Both modes use the **same TaskSession implementation**. Server mode just adds an HTTP transport layer on top.

### Phase 2 Will Complete This

**Current (Phase 1):** Task server is minimal placeholder (no TaskSession integration)

**Future (Phase 2):**
```python
# session_manager.py - _run_task_server()
session = TaskSession(session_id, task_id, env)  # Create TaskSession
app = create_task_server_app(session)  # Wrap with HTTP

# task_server.py
def create_task_server_app(session: TaskSession):
    @app.post("/tools/list")
    async def tools_list():
        return session.list_tools()  # Just calls TaskSession method!
```

**Result:** No duplication, just different access patterns (direct vs HTTP).

## Out of Scope (Phase 2)

**Task-level endpoints** will be implemented in a separate plan:
- MCP protocol endpoints: tools/list, tools/call, resources/list, resources/read
- CUBE extension endpoints: cube/evaluation, cube/step, cube/reset, cube/close
- TaskSession integration with Task Server (HTTP wrapper)
- Complete the architecture shown above

For this plan, `spawn()` will create a **minimal placeholder task server** (just responds to health check, no task endpoints).

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
    Function run in subprocess to start task server (Phase 1: minimal placeholder).

    Phase 1: Starts minimal server with health check only
    Phase 2: Will create TaskSession and expose task endpoints

    Args:
        session_id: Session identifier
        task_id: Task identifier
        env_config: Environment configuration (unused in Phase 1)
        host: Server host
        port: Server port
        seed: Random seed (unused in Phase 1)
    """
    from cube.server.task_server import create_task_server_app
    import uvicorn

    # Phase 1: Create minimal placeholder server
    app = create_task_server_app(session_id=session_id, task_id=task_id)

    # Phase 2 TODO: Create environment and TaskSession
    # env = env_config.make()
    # env.reset(seed)
    # session = TaskSession(session_id=session_id, task_id=task_id, env=env)
    # app = create_task_server_app(session)  # Pass session instead

    # Run server (blocking)
    uvicorn.run(app, host=host, port=port, log_level="info")
```

**Purpose**: Manages the lifecycle of task server subprocesses, handles port allocation, tracks active sessions.

#### 1.2 Task Server Placeholder (Minimal)

**File**: [src/cube/server/task_server.py](src/cube/server/task_server.py) (new)

```python
"""Minimal placeholder task server (Phase 1 - no task endpoints yet)."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def create_task_server_app(session_id: str, task_id: str) -> FastAPI:
    """
    Create minimal FastAPI application for a task server placeholder.

    Phase 1: Only health check endpoint
    Phase 2: Will add task-level endpoints (tools/*, resources/*, cube/*)

    Args:
        session_id: Session identifier
        task_id: Task identifier

    Returns:
        FastAPI application
    """
    app = FastAPI(
        title=f"CUBE Task Server - {task_id}",
        description=f"Task-level API placeholder for session {session_id}",
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

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {
            "status": "ok",
            "session_id": session_id,
            "task_id": task_id,
            "message": "Task server running (Phase 1 - no task endpoints yet)"
        }

    # TODO Phase 2: Add task-level endpoints here
    # - POST /tools/list (MCP)
    # - POST /tools/call (MCP)
    # - POST /resources/list (MCP)
    # - POST /resources/read (MCP)
    # - POST /cube/evaluation (CUBE)
    # - POST /cube/step (CUBE)
    # - POST /cube/reset (CUBE)
    # - POST /cube/close (CUBE)

    return app
```

**Purpose**: Minimal placeholder server that spawned tasks run. Phase 2 will add actual task endpoints.

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

    # Concrete implementations for both modes
    def spawn(self, request: SpawnRequest) -> SpawnResponse:
        """
        Spawn a new task session.

        Server mode: Creates subprocess with task server (returns URL)
        Python mode: Creates TaskSession in-process (returns session object)
        """
        if self._session_manager:
            # Server mode: spawn subprocess
            return self._session_manager.spawn(request)
        else:
            # Python mode: create TaskSession directly in this process
            import uuid
            from cube.task import TaskSession

            # Find task
            tasks = self.load_tasks()
            task = next((t for t in tasks if t.id == request.task_id), None)
            if not task:
                raise ValueError(f"Task {request.task_id} not found")

            # Create environment
            env_config = EnvConfig(task=task, tool_config=self.tool_config)
            env = env_config.make()
            env.reset(request.seed)

            # Create session
            session_id = str(uuid.uuid4())
            session = TaskSession(session_id=session_id, task_id=request.task_id, env=env)

            # Track locally
            if not hasattr(self, '_local_sessions'):
                self._local_sessions = {}
            self._local_sessions[session_id] = session

            # Update task status
            task.status = TaskStatus(
                session_id=session_id,
                task_id=request.task_id,
                status=TaskStatusEnum.running,
                created_at=datetime.now(),
                step_count=0,
                last_updated=None,
                other={}
            )

            return SpawnResponse(
                url=None,  # No URL in Python mode
                session_id=session_id,
                other={"session": session}  # Return actual session object!
            )

    def get_task_status(self, request: StatusRequest) -> StatusResponse:
        """Get status of task sessions."""
        if self._session_manager:
            # Server mode: get from session manager
            return self._session_manager.get_status(request)
        else:
            # Python mode: get from local sessions
            if not hasattr(self, '_local_sessions'):
                return StatusResponse(tasks=[])

            if request.session_id:
                # Single session
                session = self._local_sessions.get(request.session_id)
                if session:
                    task = next((t for t in self.load_tasks() if t.id == session.task_id), None)
                    if task and task.status:
                        return StatusResponse(tasks=[task.status])
                return StatusResponse(tasks=[])
            else:
                # All sessions
                all_statuses = []
                for session in self._local_sessions.values():
                    task = next((t for t in self.load_tasks() if t.id == session.task_id), None)
                    if task and task.status:
                        all_statuses.append(task.status)
                return StatusResponse(tasks=all_statuses)

    def shutdown(self, request: ShutdownRequest) -> ShutdownResponse:
        """Shutdown task sessions."""
        if self._session_manager:
            # Server mode: shutdown subprocesses
            return self._session_manager.shutdown(request)
        else:
            # Python mode: close local sessions
            if not hasattr(self, '_local_sessions'):
                return ShutdownResponse(success=True, cleaned=[])

            cleaned = []
            if request.session_id:
                # Shutdown single session
                session = self._local_sessions.pop(request.session_id, None)
                if session:
                    session.close()
                    cleaned.append(request.session_id)
            else:
                # Shutdown all sessions
                for session_id, session in list(self._local_sessions.items()):
                    session.close()
                    cleaned.append(session_id)
                self._local_sessions.clear()

            return ShutdownResponse(success=True, cleaned=cleaned)
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

### Python API Mode (Direct) - Phase 1

```python
from my_cube import MyBenchmark
from cube.types import SpawnRequest, TaskRequest, StatusRequest, ShutdownRequest

# Instantiate benchmark
bench = MyBenchmark()

# Setup (Python mode - no HTTP server)
bench.setup(
    available_ports=list(range(8000, 8100)),  # Not used in Python mode
    tool_config=MyToolConfig(),
    server_mode=False  # Python API mode - everything in same process
)

# 1. Get benchmark info
metadata = bench.info()
print(f"Benchmark: {metadata.name} v{metadata.version}")

# 2. List tasks
task_list = bench.list_tasks(TaskRequest())
print(f"Found {task_list.total} tasks")

# 3. Spawn task (creates TaskSession in-process, no subprocess)
response = bench.spawn(SpawnRequest(task_id="task-1", seed=42))
print(f"Session ID: {response.session_id}")
print(f"URL: {response.url}")  # None in Python mode
session = response.other["session"]  # Direct TaskSession object!

# 4. Use TaskSession directly (Phase 2 - when task endpoints are implemented)
# tools = session.list_tools()  # Direct method call, no HTTP
# result = session.call_tool(...)

# 5. Get status
status = bench.get_task_status(StatusRequest(session_id=response.session_id))
print(f"Task status: {status.tasks[0].status}")  # "running"

# 6. Shutdown task
shutdown_result = bench.shutdown(ShutdownRequest(session_id=response.session_id))
print(f"Cleaned sessions: {shutdown_result.cleaned}")
```

**What works in Phase 1 (Python mode):**
- ✅ Benchmark-level API: info(), list_tasks(), spawn(), status(), shutdown()
- ✅ In-process TaskSession creation (no subprocesses)
- ✅ Direct access to session object
- ❌ Task-level methods (Phase 2): list_tools(), call_tool(), etc. on TaskSession

**Key difference from server mode:**
- **No HTTP**, **no subprocesses**, **no servers**
- Everything runs in your Python script's process
- Faster, simpler, but tasks don't survive if script crashes

### JSON-RPC/HTTP API Mode (Remote) - Phase 1

```python
from my_cube import MyBenchmark
import requests

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

# 1. Get benchmark info
response = requests.post(
    "http://localhost:8000/cube/info",
    json={"jsonrpc": "2.0", "method": "cube/info", "params": {}, "id": 1}
)
print(response.json())  # {"result": {"name": "...", "version": "..."}}

# 2. List available tasks
response = requests.post(
    "http://localhost:8000/cube/tasks",
    json={"jsonrpc": "2.0", "method": "cube/tasks", "params": {}, "id": 2}
)
tasks = response.json()["result"]["tasks"]

# 3. Spawn a task server
response = requests.post(
    "http://localhost:8000/cube/spawn",
    json={
        "jsonrpc": "2.0",
        "method": "cube/spawn",
        "params": {"task_id": "task-1", "seed": 42},
        "id": 3
    }
)
result = response.json()["result"]
task_url = result["url"]  # "http://localhost:8001"
session_id = result["session_id"]

# 4. Task server is now running (minimal placeholder in Phase 1)
# Health check works:
health = requests.get(f"{task_url}/health")
print(health.json())  # {"status": "ok", "session_id": "...", "task_id": "task-1"}

# Phase 1: Task server has NO task-level endpoints yet
# Phase 2 will add: tools/list, tools/call, resources/*, cube/evaluation, etc.

# 5. Get task status
response = requests.post(
    "http://localhost:8000/cube/status",
    json={"jsonrpc": "2.0", "method": "cube/status", "params": {"session_id": session_id}, "id": 4}
)
status = response.json()["result"]

# 6. Shutdown task server
response = requests.post(
    "http://localhost:8000/cube/shutdown",
    json={"jsonrpc": "2.0", "method": "cube/shutdown", "params": {"session_id": session_id}, "id": 5}
)
shutdown_result = response.json()["result"]  # {"success": true, "cleaned": ["session-id"]}
```

**What works in Phase 1:**
- ✅ Benchmark-level APIs: info, tasks, spawn, status, shutdown
- ✅ Task server spawning and lifecycle management
- ✅ Process isolation (subprocess per task)
- ❌ Task-level APIs (Phase 2): tools/*, resources/*, cube/evaluation, cube/step, etc.

## Critical Files to Modify/Create

### Phase 1 (This Plan) - Benchmark Server Only

1. **[src/cube/server/session_manager.py](src/cube/server/session_manager.py)** - Session lifecycle management (NEW)
2. **[src/cube/server/task_server.py](src/cube/server/task_server.py)** - Minimal task server placeholder (NEW)
3. **[src/cube/server/benchmark_server.py](src/cube/server/benchmark_server.py)** - Benchmark FastAPI server (NEW)
4. **[src/cube/server/__init__.py](src/cube/server/__init__.py)** - Server package exports (NEW)
5. **[src/cube/benchmark.py](src/cube/benchmark.py)** - Add server_mode to setup()
6. **[src/cube/__init__.py](src/cube/__init__.py)** - Export server components
7. **[pyproject.toml](pyproject.toml)** - Add FastAPI and uvicorn dependencies

### Phase 2 (Future) - Task Endpoints

- Update task_server.py to add actual task endpoints
- Integrate TaskSession with task server
- Implement MCP protocol handlers
- Implement CUBE extension handlers

## Verification Plan (Phase 1)

### Test 1: Benchmark Server Starts
```bash
# Create simple test script
python examples/test_benchmark_server.py

# Expected:
# - Benchmark server starts on port 8000
# - FastAPI auto-docs available at http://localhost:8000/docs
# - Server shows 5 endpoints: cube/info, cube/tasks, cube/spawn, cube/status, cube/shutdown
```

### Test 2: cube/info Endpoint
```bash
# Via HTTP
curl -X POST http://localhost:8000/cube/info \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "cube/info", "params": {}, "id": 1}'

# Expected: Returns benchmark metadata (name, version, description)
```

### Test 3: cube/tasks Endpoint
```bash
# Via HTTP
curl -X POST http://localhost:8000/cube/tasks \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "cube/tasks", "params": {}, "id": 2}'

# Expected: Returns list of available tasks
```

### Test 4: cube/spawn Endpoint
```bash
# Via HTTP
curl -X POST http://localhost:8000/cube/spawn \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "cube/spawn", "params": {"task_id": "task-1"}, "id": 3}'

# Expected:
# - Returns: {"result": {"url": "http://localhost:8001", "session_id": "..."}}
# - Task server starts on port 8001
# - Health check works: curl http://localhost:8001/health
```

### Test 5: cube/status Endpoint
```bash
# Via HTTP
curl -X POST http://localhost:8000/cube/status \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "cube/status", "params": {}, "id": 4}'

# Expected: Returns list of running task sessions
```

### Test 6: cube/shutdown Endpoint
```bash
# Via HTTP
curl -X POST http://localhost:8000/cube/shutdown \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "cube/shutdown", "params": {"session_id": "..."}, "id": 5}'

# Expected:
# - Task server at port 8001 stops
# - Returns: {"result": {"success": true, "cleaned": ["session-id"]}}
```

### Test 7: Process Isolation
```bash
# After spawning 2 tasks
ps aux | grep uvicorn

# Expected: 3 Python processes
# - 1 benchmark server (port 8000)
# - 2 task servers (ports 8001, 8002)
```

### Test 8: Python API (Non-Server Mode)
```python
# Direct Python usage (server_mode=False)
bench = MyBenchmark()
bench.setup(available_ports=[...], tool_config=..., server_mode=False)

# Expected: Benchmark methods work directly
metadata = bench.info()
tasks = bench.list_tasks(TaskRequest())
```

## Phase 2 Verification (Future)

- Test task-level endpoints (tools/*, resources/*, cube/*)
- Test TaskSession integration
- Test MCP client compatibility
- End-to-end agent evaluation workflow

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
