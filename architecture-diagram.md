# CUBE Architecture Diagram

## Component Relationships

```mermaid
graph TD
    %% Main Components
    Benchmark[Benchmark<br/>Container for multiple tasks]
    Task[Task<br/>Defines task logic & validation]
    Environment[Environment<br/>Lifecycle manager]
    TaskSession[TaskSession<br/>MCP + CUBE API]
    EnvConfig[EnvConfig<br/>Factory for Environment]
    ToolConfig[ToolConfig<br/>Defines MCP server with tools]
    MCPServer[MCP Server<br/>In-memory FastMCP]

    %% Benchmark contains Tasks and ToolConfig
    Benchmark -->|"load_tasks()"| TaskList[List of Tasks]
    TaskList -->|contains| Task
    Benchmark -->|"tool_config field"| ToolConfig

    %% Benchmark spawns sessions
    Benchmark -->|"spawn()"| EnvConfig
    EnvConfig -->|"make()"| Environment
    Environment -->|wraps| Task
    Environment -->|"reset()"| TaskSession

    %% ToolConfig creates MCP server
    ToolConfig -->|"create_mcp_server(task)"| MCPServer
    MCPServer -->|"closure access"| Task

    %% TaskSession lifecycle
    TaskSession -->|delegates to| Environment
    TaskSession -->|"call_tool() via"| MCPServer
    TaskSession -->|"mcp_server field"| MCPServer

    %% Environment delegates to Task
    Environment -.->|"reset() → task.setup()"| Task
    Environment -.->|"close() → task.teardown()"| Task

    %% Task validation
    TaskSession -->|"validate_task()"| Task
    TaskSession -->|"filter_actions()"| Task

    %% Styling
    classDef container fill:#e1f5ff,stroke:#0288d1
    classDef core fill:#fff3e0,stroke:#f57c00
    classDef session fill:#f3e5f5,stroke:#7b1fa2
    classDef factory fill:#e8f5e9,stroke:#388e3c

    class Benchmark container
    class Task,Environment core
    class TaskSession session
    class EnvConfig,ToolConfig,MCPServer factory
```

## Flow Diagram: From Benchmark to Task Execution

```mermaid
sequenceDiagram
    participant User
    participant Benchmark
    participant Task
    participant ToolConfig
    participant EnvConfig
    participant Environment
    participant TaskSession
    participant MCPServer
    participant Agent

    %% Setup Phase
    User->>Benchmark: setup(tool_config)
    Benchmark->>Benchmark: setup_benchmark_resources()
    Note over Benchmark: Sets tool_config field

    %% Task Loading
    User->>Benchmark: load_tasks()
    Benchmark-->>User: List[Task]

    %% Spawning a Session
    User->>Benchmark: spawn(task_id, seed)
    Benchmark->>Benchmark: Find task by ID

    %% Create MCP Server first
    Benchmark->>ToolConfig: create_mcp_server(task)
    ToolConfig->>MCPServer: FastMCP instance with tools
    Note over MCPServer: Tools have closure access to task state

    %% Create Environment
    Benchmark->>EnvConfig: new EnvConfig(task, tool_config)
    EnvConfig->>Environment: make()
    Environment->>Task: setup()
    Task-->>Environment: (observation, info)

    %% Create TaskSession with MCP server
    Benchmark->>TaskSession: new TaskSession(env, mcp_server)
    TaskSession-->>Benchmark: session_id
    Benchmark-->>User: SpawnResponse

    %% Task Execution Loop
    Agent->>TaskSession: list_tools()
    TaskSession->>MCPServer: await list_tools()
    MCPServer-->>TaskSession: List[MCPTool]
    TaskSession->>Task: filter_actions()
    Task-->>Agent: Filtered tools

    Agent->>TaskSession: call_tool(name, args)
    TaskSession->>MCPServer: await call_tool()
    MCPServer->>Task: Tool modifies task state (via closure)
    MCPServer-->>TaskSession: MCPCallToolResult
    TaskSession->>Task: validate_task()
    Task-->>TaskSession: (reward, info)
    TaskSession->>Task: finished()?
    Task-->>TaskSession: Boolean
    TaskSession-->>Agent: Tool result + evaluation

    %% Cleanup
    Agent->>TaskSession: close()
    TaskSession->>Environment: close()
    Environment->>Task: teardown()
    TaskSession-->>Agent: CloseResponse
```

## Data Flow: Task Execution Step

```mermaid
flowchart LR
    A[Agent calls tool] -->|MCPCallToolRequest| B[TaskSession.call_tool]
    B -->|await| C[MCP Server]
    C -->|Closure modifies| D[Task State]
    C -->|MCPCallToolResult| B
    B -->|validate_task| E[Task.validate_task]
    E -->|reward, info| B
    B -->|finished?| F[Task.finished]
    F -->|Boolean| B
    B -->|obs_postprocess| G[Task.obs_postprocess]
    G -->|Processed obs| B
    B -->|Result| H[Agent receives result]

    style A fill:#e3f2fd
    style H fill:#e3f2fd
    style D fill:#fff3e0
    style C fill:#f3e5f5
```

## Class Relationship Diagram

```mermaid
classDiagram
    class Benchmark {
        +BenchmarkMetadata metadata
        +ToolConfig tool_config
        -List~Task~ _task_list
        -SessionManager _session_manager
        +setup_benchmark_resources()
        +load_tasks() List~Task~
        +spawn(request) SpawnResponse
        +list_tasks() TaskListResponse
        +get_task_status() StatusResponse
        +shutdown() ShutdownResponse
    }

    class Task {
        +TaskMetadata metadata
        +bool validate_per_step
        +setup(tool) tuple
        +teardown() None
        +validate_task(obs) tuple
        +filter_actions(actions) List
        +finished() bool
        +obs_postprocess(obs) Observation
    }

    class ToolConfig {
        <<abstract>>
        +create_mcp_server(task) FastMCP
    }

    class Environment {
        +Task task
        +reset() EnvironmentOutput
        +step(action) EnvironmentOutput
        +get_actions() List~MCPTool~
        +close() None
    }

    class TaskSession {
        +str session_id
        +str task_id
        +Environment env
        +FastMCP mcp_server
        +int step_count
        +float total_reward
        +async list_tools() MCPListToolsResult
        +async call_tool(request) MCPCallToolResult
        +evaluate() EnvironmentOutput
        +reset(request) ResetResponse
        +close() CloseResponse
    }

    class EnvConfig {
        +Task task
        +ToolConfig tool_config
        +make() Environment
    }

    Benchmark "1" --> "*" Task : loads
    Benchmark "1" --> "1" ToolConfig : has
    Benchmark --> EnvConfig : creates
    EnvConfig --> Environment : makes
    Environment "1" --> "1" Task : wraps
    TaskSession "1" --> "1" Environment : manages
    TaskSession "1" --> "1" FastMCP : uses
    ToolConfig --> FastMCP : creates
```

## Key Relationships Summary

| From | To | Relationship | Method |
|------|-----|--------------|--------|
| **Benchmark** | Task | Contains multiple | `load_tasks()` returns `List[Task]` |
| **Benchmark** | ToolConfig | Has one | `tool_config` field (optional) |
| **ToolConfig** | FastMCP | Creates | `create_mcp_server(task)` returns `FastMCP` |
| **Benchmark** | EnvConfig | Creates | `spawn()` creates `EnvConfig(task, tool_config)` |
| **EnvConfig** | Environment | Factory | `make()` returns `Environment(task)` |
| **Environment** | Task | Wraps | Constructor takes `Task` |
| **Environment** | Task | Delegates lifecycle | `reset()` → `task.setup()` |
| **Environment** | Task | Delegates cleanup | `close()` → `task.teardown()` |
| **TaskSession** | Environment | Manages | Constructor takes `Environment` |
| **TaskSession** | FastMCP | Uses for tools | Constructor takes `mcp_server` parameter |
| **TaskSession** | Task | Validates | `call_tool()` calls `task.validate_task()` |
| **TaskSession** | Task | Filters | `list_tools()` calls `task.filter_actions()` |
| **FastMCP** | Task | Accesses state | Tools defined with closure over task instance |

## Lifecycle Flow

1. **Benchmark Setup**:
   - User calls `benchmark.setup()` or `benchmark.setup_benchmark_resources()` to initialize
   - Sets `benchmark.tool_config` field with a ToolConfig instance

2. **Task Loading**:
   - Benchmark loads tasks via `load_tasks()`, caches in `_task_list`

3. **Session Spawn**:
   - User calls `benchmark.spawn(task_id, seed)`
   - Benchmark finds task from loaded list
   - **ToolConfig creates MCP server**: `tool_config.create_mcp_server(task)` returns FastMCP instance
     - Tools are defined with closures that access task state directly
   - Creates `EnvConfig(task, tool_config)` → `Environment(task)`
   - Calls `env.reset()` which calls `task.setup()`
   - Creates `TaskSession(env, mcp_server)` with MCP server reference

4. **Task Execution**:
   - Agent lists tools via `await session.list_tools()`
     - TaskSession calls `await mcp_server.list_tools()`
     - Filters through `task.filter_actions()`
   - Agent calls tools via `await session.call_tool()`
     - TaskSession calls `await mcp_server.call_tool()`
     - MCP tool function executes, modifying task state via closure
     - Task validates via `task.validate_task()` (returns reward)
     - Task checks completion via `task.finished()`

5. **Cleanup**:
   - Agent calls `session.close()`
   - Calls `env.close()` → `task.teardown()`
   - Returns profiling data

## Key Architecture Decisions

### ToolConfig as Single Source of Truth

The architecture uses **ToolConfig** as the single source of truth for defining task action spaces:

- **Before**: Tasks had `register_mcp_tools()` method that was called during session creation
- **After**: ToolConfig's `create_mcp_server(task)` creates the MCP server with all tools defined
- **Benefit**: Enables research flexibility - different ToolConfigs can expose different tools for the same task

### In-Memory MCP Server

MCP servers are **in-memory FastMCP instances**, not subprocesses:

- TaskSession holds a reference to the MCP server instance
- `call_tool()` and `list_tools()` are async methods that await the MCP server
- Tools defined in ToolConfig use closures to access task state directly
- No subprocess overhead, simpler architecture

### Async Task Session Methods

TaskSession methods are async for MCP server interaction:

- `async list_tools()` - awaits MCP server's async list_tools()
- `async call_tool()` - awaits MCP server's async tool execution
- FastAPI endpoints are already async, so no changes needed for HTTP clients
- Python mode callers must use `await` when calling these methods

### Tool State Access via Closure

Tools access task state through closure, not through parameters:

```python
def create_mcp_server(self, task: Task) -> FastMCP:
    mcp = FastMCP(f"Counter Task: {task.metadata.id}")

    @mcp.tool()
    def increment() -> str:
        # Direct access to task state via closure
        task.counter += 1
        return f"Counter is now {task.counter}"

    return mcp
```

### No register_mcp_tools()

The `Task.register_mcp_tools()` method was **removed**:

- Tasks no longer define their own tools
- ToolConfig is responsible for creating the MCP server with tools
- Cleaner separation: Task defines logic, ToolConfig defines interface
