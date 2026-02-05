# MCP Architecture: Two Approaches

## Overview

Both approaches eliminate `tool.py` and use FastMCP for task-specific tools. The key difference is **how agents access these tools**.

---

## Approach 1: HTTP Proxy to MCP (Recommended)

**Flow**: Agent → HTTP (FastAPI) → TaskSession.call_tool() → MCP Server

### Architecture Diagram
```
┌─────────┐     HTTP POST      ┌──────────────┐
│  Agent  │ ──────────────────>│  FastAPI     │
└─────────┘   /tools/call      │  Task Server │
                                └──────┬───────┘
                                       │ in-process call
                                       v
                                ┌──────────────┐
                                │ MCP Server   │
                                │ (in-memory)  │
                                └──────┬───────┘
                                       │
                                       v
                                ┌──────────────┐
                                │ Task State   │
                                │ (self.counter)│
                                └──────────────┘
```

### Implementation

#### 1. SessionManager.spawn() - Create in-memory MCP server
```python
# src/cube/server/task_server.py

class SessionManager:
    def spawn(self, request: SpawnRequest) -> SpawnResponse:
        # Get task
        tasks = self.benchmark.load_tasks()
        task = next((t for t in tasks if t.id == request.task_id), None)
        if not task:
            raise ValueError(f"Task {request.task_id} not found")

        # Get port
        port = self.available_ports.pop(0)
        self.used_ports.append(port)

        # Create MCP server (in-memory, not running as subprocess)
        tool_config = getattr(self.benchmark, "tool_config", None)
        if tool_config is None:
            raise ValueError(f"ToolConfig is required for task {task.id}")
        mcp_server = tool_config.create_mcp_server(task)

        # Create Environment (still needed for lifecycle)
        env_config = EnvConfig(task=task)
        env = env_config.make()
        env.mcp_server = mcp_server  # Store MCP server reference
        env.reset()

        # Create TaskSession with MCP server reference
        session = TaskSession(task_id=request.task_id, env=env, mcp_server=mcp_server)

        # Create FastAPI app with task session
        app = create_task_server_app(session)

        # Start FastAPI server in subprocess
        def run_server():
            uvicorn.run(app, host=self.host, port=port, log_level="info")

        task_process = multiprocessing.Process(target=run_server)
        task_process.start()

        # Track session
        server_process = TaskServerProcess(session, port, task_process)
        self.active_sessions[session.session_id] = server_process
        session.status = TaskStatusEnum.running

        return SpawnResponse(
            url=f"http://{self.host}:{port}",  # Single HTTP URL
            session_id=session.session_id,
            other={"session": session}
        )
```

#### 2. TaskSession - Store MCP server reference
```python
# src/cube/task.py

class TaskSession:
    def __init__(self, task_id: str, env: Environment, mcp_server: Any = None):
        self.session_id = str(uuid.uuid4())
        self.task_id = task_id
        self.env = env
        self.mcp_server = mcp_server  # In-memory MCP server
        self.status = TaskStatusEnum.created
        self.step_count = 0
        self.total_reward = 0.0
        # ... other fields
```

#### 3. TaskSession.call_tool() - Proxy to MCP
```python
# src/cube/task.py

def call_tool(self, request: MCPCallToolRequest) -> MCPCallToolResult:
    """Execute tool via MCP server (HTTP proxy approach)"""
    if self.status == TaskStatusEnum.stopped:
        raise TaskClosedException(self.session_id)

    try:
        # Call MCP server directly (in-memory, no HTTP)
        tool_name = request.params.name
        tool_args = request.params.arguments or {}

        # Call the registered MCP tool
        # The MCP server will execute the tool and return result
        result_text = self.mcp_server.call_tool(tool_name, tool_args)

        # Update tracking
        self.step_count += 1
        self.last_updated = datetime.now()

        # Check if task is complete
        obs = Observation.from_text(result_text)
        if self.env.task.validate_per_step or self.env.task.finished():
            reward, info = self.env.task.validate_task(obs)
            self.total_reward += reward

        # Return MCP format
        mcp_content = [MCPTextContent(type="text", text=result_text)]
        return MCPCallToolResult(content=mcp_content, isError=False)

    except Exception as e:
        logger.exception(f"Tool execution failed: {e}")
        error_content = [MCPTextContent(type="text", text=f"Error: {str(e)}")]
        return MCPCallToolResult(content=error_content, isError=True)
```

#### 4. FastAPI endpoints remain same
```python
# src/cube/server/task_server.py (lines 200-237)

def create_task_server_app(session: TaskSession) -> FastAPI:
    app = FastAPI()

    # MCP endpoints (proxy to MCP server)
    @app.post("/tools/list")
    async def list_tools():
        result = session.list_tools()
        return result

    @app.post("/tools/call")
    async def call_tool(request: MCPCallToolRequest):
        result = session.call_tool(request)
        return result

    # CUBE lifecycle endpoints
    @app.post("/cube/reset")
    async def reset_task(request: ResetRequest):
        return session.reset(request)

    @app.post("/cube/evaluate")
    async def evaluate_task():
        return session.evaluate()

    @app.post("/cube/close")
    async def close_task():
        return session.close()

    return app
```

### Agent Usage (Approach 1)
```python
import requests

# Agent calls HTTP endpoints
session_url = "http://localhost:9000"

# List available tools
response = requests.post(f"{session_url}/tools/list")
tools = response.json()

# Call a tool
response = requests.post(
    f"{session_url}/tools/call",
    json={"params": {"name": "increment", "arguments": {}}}
)
result = response.json()

# Evaluate task
response = requests.post(f"{session_url}/cube/evaluate")
evaluation = response.json()
```

### Pros/Cons (Approach 1)
✅ **Pros**:
- Single HTTP interface for agents (simpler client)
- No MCP client library needed for agents
- Easy to add middleware (auth, rate limiting, logging)
- Unified API surface
- Lower complexity for agent developers

❌ **Cons**:
- Extra HTTP layer (minimal overhead since MCP is in-memory)
- Not using MCP protocol end-to-end

---

## Approach 2: Direct MCP Connection

**Flow**: Agent → MCP Server (direct via SSE/stdio)

### Architecture Diagram
```
┌─────────┐     MCP Protocol    ┌──────────────┐
│  Agent  │ ───────────────────>│  MCP Server  │
│ (MCP    │     (SSE/stdio)     │  (port 9001) │
│ client) │                     └──────┬───────┘
└─────────┘                            │
                                       v
     ┌──────────────┐           ┌──────────────┐
     │  FastAPI     │           │ Task State   │
     │  (port 9000) │           │ (self.counter)│
     └──────────────┘           └──────────────┘
      CUBE Lifecycle
      (separate port)
```

### Implementation

#### 1. SessionManager.spawn() - Run MCP as subprocess
```python
# src/cube/server/task_server.py

class SessionManager:
    def spawn(self, request: SpawnRequest) -> SpawnResponse:
        # Get task
        tasks = self.benchmark.load_tasks()
        task = next((t for t in tasks if t.id == request.task_id), None)
        if not task:
            raise ValueError(f"Task {request.task_id} not found")

        # Allocate TWO ports: one for MCP, one for FastAPI
        if len(self.available_ports) < 2:
            raise RuntimeError("Need 2 ports: MCP + FastAPI")

        mcp_port = self.available_ports.pop(0)
        fastapi_port = self.available_ports.pop(0)
        self.used_ports.extend([mcp_port, fastapi_port])

        # Create MCP server
        tool_config = getattr(self.benchmark, "tool_config", None)
        if tool_config is None:
            raise ValueError(f"ToolConfig is required for task {task.id}")
        mcp_server = tool_config.create_mcp_server(task)

        # Start MCP server in subprocess
        def run_mcp_server():
            transport = request.transport or "sse"
            mcp_server.run(transport=transport, host=self.host, port=mcp_port)

        mcp_process = multiprocessing.Process(target=run_mcp_server)
        mcp_process.start()

        # Create Environment (for CUBE lifecycle only)
        env_config = EnvConfig(task=task)
        env = env_config.make()
        env.reset()

        # Create TaskSession (NO MCP server reference)
        session = TaskSession(task_id=request.task_id, env=env)

        # Create FastAPI app for CUBE lifecycle ONLY
        app = create_cube_lifecycle_app(session)

        # Start FastAPI server in subprocess
        def run_fastapi_server():
            uvicorn.run(app, host=self.host, port=fastapi_port, log_level="info")

        fastapi_process = multiprocessing.Process(target=run_fastapi_server)
        fastapi_process.start()

        # Track both processes
        server_process = TaskServerProcess(
            session=session,
            mcp_port=mcp_port,
            fastapi_port=fastapi_port,
            mcp_process=mcp_process,
            fastapi_process=fastapi_process
        )
        self.active_sessions[session.session_id] = server_process
        session.status = TaskStatusEnum.running

        return SpawnResponse(
            url=f"http://{self.host}:{mcp_port}",  # MCP server URL
            session_id=session.session_id,
            other={
                "mcp_url": f"http://{self.host}:{mcp_port}",
                "lifecycle_url": f"http://{self.host}:{fastapi_port}",
                "session": session
            }
        )
```

#### 2. TaskSession - No MCP reference needed
```python
# src/cube/task.py

class TaskSession:
    def __init__(self, task_id: str, env: Environment):
        self.session_id = str(uuid.uuid4())
        self.task_id = task_id
        self.env = env
        # No mcp_server attribute - agents call MCP directly
        self.status = TaskStatusEnum.created
        self.step_count = 0
        self.total_reward = 0.0
        # ... other fields
```

#### 3. FastAPI app - CUBE lifecycle ONLY
```python
# src/cube/server/task_server.py

def create_cube_lifecycle_app(session: TaskSession) -> FastAPI:
    """FastAPI app for CUBE lifecycle endpoints only (no MCP proxy)"""
    app = FastAPI()

    # NO /tools/* endpoints - agents use MCP directly

    # CUBE lifecycle endpoints only
    @app.post("/cube/reset")
    async def reset_task(request: ResetRequest):
        return session.reset(request)

    @app.post("/cube/evaluate")
    async def evaluate_task():
        return session.evaluate()

    @app.post("/cube/status")
    async def get_status():
        return session.get_status()

    @app.post("/cube/close")
    async def close_task():
        return session.close()

    return app
```

#### 4. TaskSession methods - No call_tool needed
```python
# src/cube/task.py

class TaskSession:
    # NO call_tool() method - agents use MCP directly

    def evaluate(self) -> EvaluateResponse:
        """Evaluate task state"""
        obs = Observation.from_text("Evaluating task")
        reward, info = self.env.task.validate_task(obs)
        self.total_reward += reward

        return EvaluateResponse(
            reward=reward,
            total_reward=self.total_reward,
            info=info
        )

    def reset(self, request: ResetRequest) -> ResetResponse:
        """Reset task state"""
        result = self.env.reset()
        self.step_count = 0
        self.total_reward = 0.0
        self.status = TaskStatusEnum.running

        return ResetResponse(
            observation=result.obs,
            info=result.info
        )
```

### Agent Usage (Approach 2)
```python
from mcp import ClientSession
from mcp.client.sse import sse_client
import requests

# Agent connects to MCP server directly
mcp_url = "http://localhost:9001"
lifecycle_url = "http://localhost:9000"

# Use MCP client for tools
async with sse_client(mcp_url) as (read, write):
    async with ClientSession(read, write) as session:
        # Initialize
        await session.initialize()

        # List tools
        tools = await session.list_tools()

        # Call tools directly via MCP
        result = await session.call_tool("increment", {})
        print(result)

# Use HTTP for CUBE lifecycle
response = requests.post(f"{lifecycle_url}/cube/evaluate")
evaluation = response.json()
```

### Pros/Cons (Approach 2)
✅ **Pros**:
- Native MCP protocol end-to-end
- True separation: MCP for tools, HTTP for lifecycle
- Lower latency (no HTTP proxy)
- MCP features (streaming, resources) available

❌ **Cons**:
- Agents need MCP client library
- Two separate ports/processes per task
- More complex deployment
- Higher resource usage (2 processes per task)
- More complex for agent developers

---

## Comparison Table

| Aspect | Approach 1: HTTP Proxy | Approach 2: Direct MCP |
|--------|------------------------|------------------------|
| **Agent Interface** | HTTP only | MCP client required |
| **Ports per task** | 1 (FastAPI) | 2 (MCP + FastAPI) |
| **Processes per task** | 1 | 2 |
| **Tool call flow** | HTTP → TaskSession → MCP (in-memory) | MCP direct → Task tools |
| **Complexity** | Low | Medium |
| **Agent code** | `requests` library | `mcp` client library |
| **Middleware** | Easy (FastAPI) | Limited |
| **Latency** | +1 HTTP hop | Direct |
| **Resource usage** | Lower | Higher |
| **MCP features** | Limited | Full |

---

## Recommendation: **Approach 1 (HTTP Proxy)**

### Reasons:
1. **Simpler for agents** - Just HTTP, no MCP client needed
2. **Lower resource usage** - 1 process per task instead of 2
3. **Easier middleware** - Auth, logging, rate limiting in FastAPI
4. **Unified interface** - Single URL for all operations
5. **Latency negligible** - MCP server is in-memory (no network)

### When to use Approach 2:
- Need MCP streaming features
- Agents already have MCP clients
- Want strict protocol separation
- Need lowest possible latency

---

## Recommended Implementation Steps

1. **Implement Approach 1** (HTTP Proxy) first
2. Keep MCP server in-memory (not subprocess)
3. TaskSession proxies tool calls to MCP
4. FastAPI exposes both `/tools/*` (MCP proxy) and `/cube/*` (lifecycle)
5. Later: Can support both approaches by checking request headers/transport param

## Current Implementation Status

### 🚧 Work-in-Progress (NOT YET COMMITTED)

**All changes below are uncommitted and being reviewed:**

#### New Files Created:
1. **src/cube/server/mcp_task_server.py** - REMOVED (no longer needed)
   - The wrapper function was unnecessary indirection
   - Now call `tool_config.create_mcp_server(task)` directly

2. **docs/mcp-architecture-comparison.md** - NEW FILE (untracked)
   - This comparison document

#### Modified Files:
3. **src/cube/task.py** - MODIFIED (uncommitted)
   - Added `register_mcp_tools(mcp: FastMCP)` abstract method to Task ABC
   - Added import: `from mcp.server.fastmcp import FastMCP`

4. **examples/toy_benchmark/counter.py** - MODIFIED (uncommitted)
   - Removed `CounterActions` Protocol, `CounterTool`, `CounterToolConfig` classes
   - Added `register_mcp_tools()` implementation to `ReachTargetTask`
   - State as Task attributes (self.counter, self.target, self.history)
   - Updated `CounterBenchmark.__init__` to pass `tool_config=None`

5. **src/cube/tool.py** - DELETED (uncommitted)
   - Entire file removed (~200 LOC)

6. **src/cube/__init__.py** - MODIFIED (uncommitted)
   - Removed imports: `AbstractTool`, `Tool`, `ToolConfig`
   - Removed from `__all__` exports

7. **src/cube/benchmark.py** - MODIFIED (uncommitted, partial)
   - Partially removed `tool_config` field from Benchmark class
   - **Still needs**: Complete removal of tool_config usage

8. **src/cube/environment.py** - MODIFIED (uncommitted, partial)
   - Updated EnvConfig to take only task (no tool_config)
   - Updated Environment to not require tool
   - **Still needs**: Complete MCP integration in step() method

### 🔴 Still TODO (Not Yet Started):

1. **src/cube/server/task_server.py** - MODIFIED
   - Update `SessionManager.spawn()`:
     - Create in-memory MCP server via `tool_config.create_mcp_server(task)`
     - Pass MCP server to TaskSession
     - Update EnvConfig to not use tool_config
   - Keep single subprocess (FastAPI only)

2. **src/cube/task.py** - NEEDS MORE WORK
   - Update `TaskSession.__init__()` to accept `mcp_server` parameter
   - Modify `TaskSession.call_tool()` to call MCP server instead of `env.step()`

3. **Complete benchmark.py and environment.py** refactoring

### 📝 Important Note
**NOTHING has been committed yet.** The last commit is:
```
12a206f Optimize task loading in CounterBenchmark to utilize cached tasks
```

All changes shown in this document are uncommitted work-in-progress and subject to review before committing.

---

## ✅ Implementation Complete (2026-02-04)

### Final Architecture: Approach 1 with ToolConfig

We implemented **Approach 1 (HTTP Proxy to MCP)** with a key modification: instead of `Task.register_mcp_tools()`, we use **ToolConfig** for research flexibility.

**Flow**: Agent → HTTP (FastAPI) → TaskSession.call_tool() → MCP Server (in-memory) → ToolConfig-defined tools

### Key Design Decision: ToolConfig as the Single Source of Truth

Instead of defining tools on the Task class via `register_mcp_tools()`, we introduced **ToolConfig** as an abstract base class that:
- Enables researchers to swap tool implementations without modifying benchmark code
- Provides configuration parameters for tool behavior (e.g., browser type, enable_decrement)
- Allows different tool sets (basic vs advanced)
- Maintains clean separation between task logic and action space definition

**Rationale**: This preserves tool variability for research while keeping the HTTP proxy architecture simple for agents.

### Implementation Summary

#### 6 Commits Created (Signed-off):

1. **c2947ea** - Add ToolConfig abstract base class for defining task action spaces
   - New `src/cube/tool.py` with ToolConfig.create_mcp_server() abstract method
   - Replaced old Tool/AbstractTool/ActionSpace architecture
   - Task state accessed via closure in tool implementations
   - Updated exports in `src/cube/__init__.py`

2. **852b7f0** - Add tool_config field to Benchmark for MCP server creation
   - Added `tool_config: ToolConfig | None = None` as regular field
   - Enables `setup_benchmark_resources()` to set tool configuration
   - SessionManager.spawn() can access benchmark's tool_config

3. **b6c8efb** - Update TaskSession to use in-memory MCP server for tool execution
   - TaskSession.__init__() accepts optional `mcp_server` parameter
   - `list_tools()` is now async, gets tools from MCP server
   - `call_tool()` is now async, routes to MCP server
   - `step()` is now async (awaits call_tool)
   - Fallback to env.step() for backwards compatibility

4. **dd590a6** - Add MCP server factory and update SessionManager to create MCP servers (later simplified)
   - Originally added `src/cube/server/mcp_task_server.py` with wrapper function
   - Later removed as unnecessary - now call `tool_config.create_mcp_server(task)` directly
   - SessionManager.spawn() creates MCP server via ToolConfig
   - Passes tool_config from benchmark to factory
   - TaskSession receives mcp_server in constructor

5. **27b11ef** - Update counter benchmark to use ToolConfig for tool definition
   - Added CounterToolConfig class with create_mcp_server() method
   - Removed register_mcp_tools() from ReachTargetTask
   - CounterBenchmark.setup_benchmark_resources() provides default ToolConfig
   - Tools (increment, get_value) now defined in CounterToolConfig

6. **614c264** - Add counter_with_toolconfig.py example demonstrating research flexibility
   - CounterToolConfig with enable_decrement/enable_reset parameters
   - DoubleIncrementToolConfig that increments by 2
   - Test showing default CounterToolConfig usage
   - All tests pass

### Architecture Diagram (Final)

```
┌─────────┐     HTTP POST         ┌──────────────────┐
│  Agent  │ ────────────────────> │  FastAPI         │
└─────────┘   /tools/call         │  Task Server     │
                                   └────────┬─────────┘
                                            │ in-process call
                                            v
                                   ┌────────────────────┐
                                   │ TaskSession        │
                                   │ .call_tool()       │
                                   └────────┬───────────┘
                                            │ await
                                            v
                                   ┌────────────────────┐
                                   │ MCP Server         │
                                   │ (in-memory FastMCP)│
                                   └────────┬───────────┘
                                            │
                                            v
                                   ┌────────────────────┐
                                   │ ToolConfig         │
                                   │ .create_mcp_server │
                                   └────────┬───────────┘
                                            │ closure
                                            v
                                   ┌────────────────────┐
                                   │ Task State         │
                                   │ (self.counter, etc)│
                                   └────────────────────┘
```

### Key Benefits Achieved

1. **Simple for Agents** - HTTP only, no MCP client needed
2. **Research Flexibility** - ToolConfig enables tool experimentation
3. **Low Resource Usage** - 1 process per task (FastAPI only)
4. **Clean Architecture** - Clear separation of concerns
5. **Type Safety** - ToolConfig uses Pydantic for validation
6. **Backwards Compatible** - Falls back to env.step() if no MCP server

### Breaking Changes

- TaskSession methods now async: `list_tools()`, `call_tool()`, `step()`
- Callers must use `await` when invoking these methods
- FastAPI endpoints already async, no changes needed for HTTP clients
- ToolConfig is now the only way to define task tools (no Task.register_mcp_tools)

### Example: Counter Benchmark with ToolConfig

```python
from cube.tool import ToolConfig
from mcp.server.fastmcp import FastMCP

class CounterToolConfig(ToolConfig):
    """Tool configuration for counter benchmark."""

    def create_mcp_server(self, task: Task) -> FastMCP:
        mcp = FastMCP(f"Counter Task: {task.metadata.id}")

        # Cast to specific task type
        assert isinstance(task, ReachTargetTask)
        reach_task = task

        @mcp.tool()
        def increment() -> str:
            """Increment the counter by 1"""
            reach_task.counter += 1  # Access task state via closure
            reach_task.history.append("increment")
            return f"Counter is now {reach_task.counter}"

        @mcp.tool()
        def get_value() -> str:
            """Get the current counter value"""
            return f"Counter value is: {reach_task.counter}"

        return mcp

class CounterBenchmark(Benchmark):
    def setup_benchmark_resources(self, tool_config: Any = None, **kwargs):
        # Use provided tool_config or default to CounterToolConfig
        if tool_config is None:
            tool_config = CounterToolConfig()
        return super().setup_benchmark_resources(tool_config=tool_config, **kwargs)
```

### Research Flexibility Example

Researchers can swap tool implementations without modifying benchmark code:

```python
# Example 1: Add optional tools
class ExtendedCounterToolConfig(ToolConfig):
    enable_decrement: bool = True  # Add decrement tool
    enable_reset: bool = False      # Add reset tool

    def create_mcp_server(self, task: Task) -> FastMCP:
        mcp = FastMCP(f"Counter: {task.metadata.id}")

        # ... register increment, get_value ...

        if self.enable_decrement:
            @mcp.tool()
            def decrement() -> str:
                task.counter -= 1
                return f"Counter is now {task.counter}"

        return mcp

# Example 2: Change tool behavior
class DoubleIncrementToolConfig(ToolConfig):
    def create_mcp_server(self, task: Task) -> FastMCP:
        mcp = FastMCP(f"Counter: {task.metadata.id}")

        @mcp.tool()
        def increment() -> str:
            task.counter += 2  # Increment by 2 instead of 1
            return f"Counter is now {task.counter}"

        return mcp
```

### Tests Passing

- `examples/toy_benchmark/counter.py::test_simple_counting` ✅
- `examples/toy_benchmark/counter_with_toolconfig.py` (all 3 tests) ✅

### Files Modified

**New Files:**
- `src/cube/tool.py` - ToolConfig abstract base class
- `examples/toy_benchmark/counter_with_toolconfig.py` - Research flexibility examples

**Removed Files:**
- `src/cube/server/mcp_task_server.py` - Wrapper function removed as unnecessary

**Modified Files:**
- `src/cube/__init__.py` - Export ToolConfig
- `src/cube/benchmark.py` - Add tool_config field
- `src/cube/task.py` - Async TaskSession methods with MCP server
- `src/cube/server/task_server.py` - SessionManager creates MCP servers
- `src/cube/environment.py` - Updated to work with ToolConfig
- `examples/toy_benchmark/counter.py` - Use CounterToolConfig

### Commit Log

```bash
git log --oneline -6
614c264 Add counter_with_toolconfig.py example demonstrating research flexibility
27b11ef Update counter benchmark to use ToolConfig for tool definition
dd590a6 Add MCP server factory and update SessionManager to create MCP servers
b6c8efb Update TaskSession to use in-memory MCP server for tool execution
852b7f0 Add tool_config field to Benchmark for MCP server creation
c2947ea Add ToolConfig abstract base class for defining task action spaces
```

All commits include DCO sign-off and passed pre-commit hooks (ruff, formatting).
