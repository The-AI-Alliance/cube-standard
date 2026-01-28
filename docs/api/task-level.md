---
layout: default
title: Task-Level API
parent: API Reference
nav_order: 1
---

# Task-Level API

The Task-Level API defines how agents interact with individual task instances. It combines the Model Context Protocol (MCP) for action execution with CUBE extensions for evaluation semantics.

{: .note }
> This API describes a single task instance. For managing multiple tasks and shared infrastructure, see the [Benchmark-Level API]({{site.baseurl}}/api/benchmark-level).

## Overview

A CUBE task provides three capabilities:

1. **Action execution** (via MCP `tools/*` methods)
2. **State observation** (via MCP `resources/*` methods)
3. **Evaluation** (via CUBE `cube/*` methods)

This design separates concerns:
- **MCP handles the "body"** - What the agent can do and see
- **CUBE handles the "goals"** - Whether the agent succeeded and how well

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    CUBE Task Instance                   │
├─────────────────────────────────────────────────────────┤
│  MCP Protocol (Tools & Resources)                       │
│  ├─ tools/list     → Discover available actions         │
│  ├─ tools/call     → Execute an action                  │
│  ├─ resources/list → Discover available resources       │
│  └─ resources/read → Read observation/task description  │
├─────────────────────────────────────────────────────────┤
│  CUBE Extensions (Evaluation)                           │
│  ├─ cube/evaluation → Get reward, done, info            │
│  ├─ cube/reset      → Reset to initial state            │
│  └─ cube/close      → Cleanup resources                 │
└─────────────────────────────────────────────────────────┘
         ↑                              ↑
    Agent actions              Evaluation harness
```

## MCP Protocol Methods

CUBE adopts the Model Context Protocol for action and observation. These methods are standard MCP - refer to the [MCP specification](https://modelcontextprotocol.io/docs) for complete details.

### `tools/list`

Discover the available actions (tools) for this task.

**Request:**
```json
{
  "method": "tools/list",
  "params": {}
}
```

**Response:**
```json
{
  "tools": [
    {
      "name": "click",
      "description": "Click at screen coordinates",
      "inputSchema": {
        "type": "object",
        "properties": {
          "x": {"type": "number", "description": "X coordinate"},
          "y": {"type": "number", "description": "Y coordinate"}
        },
        "required": ["x", "y"]
      }
    },
    {
      "name": "type_text",
      "description": "Type text into the focused element",
      "inputSchema": {
        "type": "object",
        "properties": {
          "text": {"type": "string", "description": "Text to type"}
        },
        "required": ["text"]
      }
    }
  ]
}
```

**Python interface:**
```python
tools = task.list_tools()
for tool in tools:
    print(f"{tool.name}: {tool.description}")
    print(f"  Parameters: {tool.inputSchema}")
```

### `tools/call`

Execute an action using a discovered tool.

**Request:**
```json
{
  "method": "tools/call",
  "params": {
    "name": "click",
    "arguments": {
      "x": 150,
      "y": 200
    }
  }
}
```

**Response:**
```json
{
  "content": [
    {
      "type": "text",
      "text": "Clicked element: <button id='submit'>Submit</button>"
    }
  ],
  "isError": false
}
```

**Python interface:**
```python
result = task.call_tool(
    name="click",
    arguments={"x": 150, "y": 200}
)

if result.isError:
    print(f"Action failed: {result.content}")
else:
    print(f"Success: {result.content}")
```

### `resources/list`

Discover available resources (observations, task descriptions, etc.).

**Request:**
```json
{
  "method": "resources/list",
  "params": {}
}
```

**Response:**
```json
{
  "resources": [
    {
      "uri": "task://description",
      "name": "Task Description",
      "description": "The goal of this task",
      "mimeType": "text/plain"
    },
    {
      "uri": "obs://current",
      "name": "Current Observation",
      "description": "Current state of the environment",
      "mimeType": "application/json"
    },
    {
      "uri": "obs://screenshot",
      "name": "Screenshot",
      "description": "Visual screenshot of current state",
      "mimeType": "image/png"
    }
  ]
}
```

**Python interface:**
```python
resources = task.list_resources()
for resource in resources:
    print(f"{resource.uri}: {resource.name}")
```

### `resources/read`

Read a specific resource by URI.

**Request:**
```json
{
  "method": "resources/read",
  "params": {
    "uri": "task://description"
  }
}
```

**Response:**
```json
{
  "contents": [
    {
      "uri": "task://description",
      "mimeType": "text/plain",
      "text": "Click the 'Submit' button to complete the form."
    }
  ]
}
```

**Python interface:**
```python
# Read task description
task_desc = task.read_resource("task://description")
print(task_desc.text)

# Read current observation
obs = task.read_resource("obs://current")
print(obs.json())

# Read screenshot (binary data)
screenshot = task.read_resource("obs://screenshot")
with open("screenshot.png", "wb") as f:
    f.write(screenshot.blob)
```

## CUBE Extension Methods

CUBE adds three methods to provide evaluation semantics similar to Gymnasium.

### `cube/evaluation`

Get the current evaluation state: observation, reward, termination status, and metadata.

**Request:**
```json
{
  "method": "cube/evaluation",
  "params": {}
}
```

**Response:**
```json
{
  "response": {
    "obs": {
      "contents": [
        {
          "type": "text",
          "data": "<html>...</html>"
        }
      ]
    },
    "reward": 0.0,
    "terminated": false,
    "truncated": false,
    "step": 5,
    "info": {
      "tokens_used": 1250,
      "time_elapsed": 12.3
    }
  }
}
```

**Fields:**

The response contains an `EnvironmentOutput` object with:

- `obs` (Observation): Current state observation containing a list of Content objects
  - `contents` (list): List of content items, each with `type`, `data`, and optional `tool_call_id` and `name`
- `reward` (float, default: 0.0): Reward received since last evaluation. Range typically [0, 1] but benchmark-specific.
- `terminated` (bool, default: false): Whether the task reached a terminal state (success or failure)
- `truncated` (bool, default: false): Whether the task was truncated due to time/step limits
- `step` (int, default: 0): Current step count
- `info` (object, default: {}): Additional metadata (performance metrics, debug info, etc.)

**Python interface:**
```python
result = task.evaluate()
state = result.response

print(f"Observation: {state.obs}")
print(f"Reward: {state.reward}")
print(f"Step: {state.step}")
print(f"Done: {state.terminated or state.truncated}")

if state.terminated:
    success = state.info.get("success", False)
    print(f"Task completed: {'Success' if success else 'Failure'}")
```

### `cube/step`

Execute an action and immediately get the evaluation state. This is a convenience method that combines `tools/call` and `cube/evaluation` in a single RPC call, reducing latency and simplifying agent code.

{: .note }
> This endpoint is automatically available whenever a task implements both `tools/call` and `cube/evaluation`. Benchmark developers don't need to implement it separately.

**Request:**
```json
{
  "method": "cube/step",
  "params": {
    "name": "click",
    "arguments": {
      "x": 150,
      "y": 200
    }
  }
}
```

**Response:**
```json
{
  "response": {
    "obs": {
      "contents": [
        {
          "type": "text",
          "tool_call_id": "call_123",
          "data": "Clicked element: <button id='submit'>Submit</button>"
        }
      ]
    },
    "reward": 1.0,
    "terminated": true,
    "truncated": false,
    "step": 6,
    "info": {
      "success": true,
      "tokens_used": 1350,
      "time_elapsed": 13.5
    }
  }
}
```

**Fields:**

The response contains an `EnvironmentOutput` object with:

- `obs` (Observation): Current state observation containing a list of Content objects
  - `contents` (list): List of content items, including tool results with `tool_call_id` field
- `reward` (float, default: 0.0): Reward received from this action
- `terminated` (bool, default: false): Whether the task reached a terminal state
- `truncated` (bool, default: false): Whether the task was truncated
- `step` (int, default: 0): Current step count after the action
- `info` (object, default: {}): Additional metadata

**Python interface:**
```python
# Instead of two separate calls:
# result = task.call_tool("click", {"x": 150, "y": 200})
# state = task.evaluate()

# Use cube/step for a single call:
result = task.step(
    name="click",
    arguments={"x": 150, "y": 200}
)

state = result.response
print(f"Observation: {state.obs}")
print(f"Reward: {state.reward}")
print(f"Step: {state.step}")
print(f"Done: {state.terminated or state.truncated}")

if state.terminated:
    success = state.info.get("success", False)
    print(f"Task completed: {'Success' if success else 'Failure'}")
```

**Benefits:**

- **Reduced latency**: Eliminates one round-trip for remote tasks (RPC overhead)
- **Simpler agent code**: Single call instead of two sequential calls
- **Atomic operation**: Guarantees evaluation happens immediately after action
- **Better performance**: Especially important for high-latency network connections

**When to use:**

- ✅ Use `cube/step` when you need both the tool result and evaluation state
- ✅ Use for typical agent control loops (act → observe → evaluate)
- ✅ Use for remote RPC tasks to minimize network overhead
- ❌ Use `tools/call` alone if you don't need immediate evaluation
- ❌ Use `cube/evaluation` alone if you want to check state without acting

### `cube/reset`

Reset the task to its initial state. Optionally accepts a seed for reproducibility.

**Request:**
```json
{
  "method": "cube/reset",
  "params": {
    "seed": 42
  }
}
```

**Response:**
```json
{
  "response": {
    "obs": {
      "contents": [
        {
          "type": "text",
          "data": "<html>...</html>"
        }
      ]
    },
    "reward": 0.0,
    "terminated": false,
    "truncated": false,
    "step": 0,
    "info": {
      "task_id": "click-button-v0",
      "task_description": "Click the Submit button",
      "seed": 42
    }
  }
}
```

**Python interface:**
```python
# Reset with specific seed for reproducibility
result = task.reset(seed=42)
state = result.response
print(f"Initial observation: {state.obs}")
print(f"Initial step: {state.step}")

# Reset with random seed
result = task.reset()
state = result.response
```

{: .note }
> Some benchmarks have deterministic initial states and ignore the seed parameter. Others generate random variations (e.g., randomized layouts, different data) and require seeds for reproducibility.

### `cube/close`

Cleanup task resources and shutdown the instance.

**Request:**
```json
{
  "method": "cube/close",
  "params": {}
}
```

**Response:**
```json
{
  "success": true,
  "profiling": {
    "total_steps": 15,
    "total_time": 45.2,
    "total_tokens": 5000
  }
}
```

**Python interface:**
```python
# Cleanup when done
task.close()

# Or use context manager (recommended)
with benchmark.spawn(task_id="example") as task:
    # Use task
    reset_result = task.reset()
    state = reset_result.response
    tool_result = task.call_tool("click", {"x": 100, "y": 100})
    eval_result = task.evaluate()
    eval_state = eval_result.response
    # Automatic cleanup on exit
```

## Standard Resource URIs

CUBE defines standard resource URIs that benchmarks should provide when applicable:

| URI | MIME Type | Description |
|-----|-----------|-------------|
| `task://description` | `text/plain` | Human-readable task goal |
| `task://instructions` | `text/plain` or `text/markdown` | Detailed instructions for the agent |
| `obs://current` | Varies | Current observation (structured) |
| `obs://screenshot` | `image/png` | Visual screenshot (if applicable) |
| `obs://accessibility-tree` | `application/json` | Accessibility tree (for GUI tasks) |
| `obs://html` | `text/html` | Current HTML (for web tasks) |
| `obs://text` | `text/plain` | Text-only observation |

Benchmarks can define additional URIs as needed, but should document them clearly.

## Evaluation Loop Pattern

The standard agent-task interaction loop using `cube/step`:

```python
from cube import LocalRunner

benchmark = LocalRunner("cube-benchmark-example")
task = benchmark.spawn(task_id="example-task", seed=42)

# Reset to initial state
reset_result = task.reset()
state = reset_result.response
total_reward = 0
done = False

while not done:
    # Agent observes
    task_desc = task.read_resource("task://description")
    current_obs = state.obs

    # Agent decides (your logic here)
    tools = task.list_tools()
    selected_tool = agent_policy(task_desc, current_obs, tools)

    # Agent acts and evaluates (single call)
    result = task.step(
        name=selected_tool.name,
        arguments=selected_tool.args
    )

    # Process results
    state = result.response
    total_reward += state.reward
    done = state.terminated or state.truncated

    if done:
        print(f"Task completed!")
        print(f"Total reward: {total_reward}")
        print(f"Success: {state.info.get('success', False)}")

task.close()
```

**Alternative: Using separate calls**

If you need fine-grained control or don't need evaluation after every action:

```python
# Agent acts
result = task.call_tool(
    name=selected_tool.name,
    arguments=selected_tool.args
)

# Check tool result before evaluating
if result.isError:
    print(f"Tool failed: {result.content}")
    # Handle error...

# Evaluate when needed
eval_result = task.evaluate()
state = eval_result.response
total_reward += state.reward
```

## Tool Reconfiguration

While benchmarks ship with default tools, some allow tool reconfiguration for research purposes:

```python
# Default tools
task = benchmark.spawn(task_id="web-task")

# Custom tools (if benchmark supports it)
task = benchmark.spawn(
    task_id="web-task",
    tool_config={
        "browser_driver": "vision-based",  # Instead of default HTML-based
        "tools": ["click_visual", "type_visual", "scroll"]
    }
)
```

Tool reconfiguration is **optional** and **benchmark-specific**. Check the benchmark documentation to see if it's supported.

## Error Handling

Task-level operations can fail for various reasons. Handle errors appropriately:

```python
from cube import CubeError, ToolExecutionError, ResourceNotFoundError

try:
    result = task.call_tool("click", {"x": 100, "y": 100})
except ToolExecutionError as e:
    print(f"Tool execution failed: {e.message}")
    print(f"Details: {e.details}")
except ResourceNotFoundError as e:
    print(f"Resource not found: {e.uri}")
except CubeError as e:
    print(f"CUBE error: {e}")
```

Common errors:

- `ToolExecutionError` - Tool/action execution failed
- `ResourceNotFoundError` - Requested resource doesn't exist
- `InvalidParameterError` - Invalid arguments to tool
- `SessionExpiredError` - Task instance was closed
- `TimeoutError` - Operation exceeded timeout

## Implementation Guide for Benchmark Authors

### Python Class Implementation

Implement a Python class with the required methods:

{: .note }
> **Bonus**: When you implement `call_tool()` and `evaluate()`, you automatically get `cube/step()` for free. The CUBE runtime provides this convenience method by calling your two methods sequentially, so benchmark developers only need to implement the core functionality.

```python
from typing import Any, Optional
from mcp.types import Tool, Resource, CallToolResult, ListToolsResult, ListResourcesResult, ReadResourceResult, TextResourceContents, BlobResourceContents

class MyBenchmarkTask:
    """CUBE-compliant task implementation."""

    def __init__(self, task_id: str, seed: Optional[int] = None):
        self.task_id = task_id
        self.seed = seed
        # Initialize your environment here
        self._env = self._create_environment()

    # MCP Methods
    def list_tools(self) -> ListToolsResult:
        """Return available tools/actions."""
        return ListToolsResult(
            tools=[
                Tool(
                    name="click",
                    description="Click at coordinates",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"}
                        },
                        "required": ["x", "y"]
                    }
                )
            ]
        )

    def call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        """Execute a tool/action."""
        if name == "click":
            x, y = arguments["x"], arguments["y"]
            result = self._env.click(x, y)
            return CallToolResult(
                content=[{"type": "text", "text": result}],
                isError=False
            )
        else:
            return CallToolResult(
                content=[{"type": "text", "text": f"Unknown tool: {name}"}],
                isError=True
            )

    def list_resources(self) -> ListResourcesResult:
        """Return available resources."""
        return ListResourcesResult(
            resources=[
                Resource(
                    uri="task://description",
                    name="Task Description",
                    description="The goal of this task",
                    mimeType="text/plain"
                ),
                Resource(
                    uri="obs://current",
                    name="Current Observation",
                    description="Current environment state",
                    mimeType="application/json"
                )
            ]
        )

    def read_resource(self, uri: str) -> ReadResourceResult:
        """Read a specific resource."""
        if uri == "task://description":
            return ReadResourceResult(
                contents=[
                    TextResourceContents(
                        uri=uri,
                        mimeType="text/plain",
                        text=self._env.get_task_description()
                    )
                ]
            )
        elif uri == "obs://current":
            obs_data = self._env.get_observation()
            return ReadResourceResult(
                contents=[
                    TextResourceContents(
                        uri=uri,
                        mimeType="application/json",
                        text=str(obs_data)  # Serialize to JSON string
                    )
                ]
            )
        else:
            raise ResourceNotFoundError(f"Resource not found: {uri}")

    # CUBE Methods
    def evaluate(self) -> dict[str, Any]:
        """Get current evaluation state."""
        obs = self._env.get_observation()
        reward = self._env.get_reward()
        terminated = self._env.is_done()
        truncated = self._env.is_truncated()

        return {
            "response": {
                "obs": {"contents": [{"type": "text", "data": obs}]},
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "step": self._env.step_count,
                "info": {
                    "success": self._env.is_successful() if terminated else False
                }
            }
        }

    def reset(self, seed: Optional[int] = None) -> dict[str, Any]:
        """Reset task to initial state."""
        if seed is not None:
            self.seed = seed
        self._env.reset(seed=self.seed)

        obs = self._env.get_observation()
        return {
            "response": {
                "obs": {"contents": [{"type": "text", "data": obs}]},
                "reward": 0.0,
                "terminated": False,
                "truncated": False,
                "step": 0,
                "info": {
                    "task_id": self.task_id,
                    "seed": self.seed
                }
            }
        }

    def close(self) -> dict[str, Any]:
        """Cleanup resources."""
        profiling = self._env.get_profiling_data()
        self._env.cleanup()

        return {
            "success": True,
            "profiling": profiling
        }

    def _create_environment(self):
        """Initialize your actual environment."""
        # Your environment initialization logic
        pass
```

### Adding RPC Layer

For remote access, expose the same methods via HTTP/JSON-RPC. Use the provided CUBE server wrapper:

```python
from cube.server import CubeTaskServer

# Your task class from above
task = MyBenchmarkTask(task_id="example", seed=42)

# Wrap in RPC server
server = CubeTaskServer(task, host="0.0.0.0", port=8001)
server.start()
```

The server automatically exposes all methods as RPC endpoints with the same signatures.

## Next Steps

- **[Benchmark-Level API]({{site.baseurl}}/api/benchmark-level)**: Learn how to manage multiple tasks
- **[Benchmark Author Guide]({{site.baseurl}}/guides/benchmark-authors)**: Complete tutorial on wrapping a benchmark
- **[Examples](../examples/)**: See complete implementation examples

