---
layout: default
title: Benchmark Author Guide
parent: Guides
nav_order: 1
---

# Benchmark Author Guide

Learn how to wrap your benchmark to be CUBE-compliant in a step-by-step tutorial.

{: .note }
> **Time estimate**: 1-2 hours for a simple benchmark, 4-6 hours for complex benchmarks with shared infrastructure

## What You'll Build

By the end of this guide, you'll have:

- ✅ A CUBE-compliant wrapper for your benchmark
- ✅ Task-Level API implementation (agent-task interaction)
- ✅ Benchmark-Level API implementation (task orchestration)
- ✅ Local Python execution support
- ✅ Remote RPC execution support
- ✅ Registry metadata for discoverability
- ✅ A distributable Python package

## Prerequisites

Before starting, ensure you have:

- A working benchmark (with tasks, environments, and evaluation logic)
- Python 3.9 or higher
- Basic familiarity with Python classes and APIs
- Your benchmark's infrastructure setup (Docker, etc.)

## Tutorial Overview

We'll wrap a hypothetical benchmark called "SimpleClick" - a minimal web interaction benchmark where agents must click specific buttons.

```
Our benchmark before CUBE:
- 10 HTML tasks with different button layouts
- Custom Python API: click(x, y), get_screenshot(), check_success()
- No standard interface, hard to integrate

Our benchmark after CUBE:
- Same 10 tasks, same functionality
- Standard CUBE API that works with any platform
- Discoverable via CUBE registry
- One-line installation: pip install cube-benchmark-simpleclick
```

## Step 1: Project Setup

### Create Package Structure

```bash
mkdir cube-benchmark-simpleclick
cd cube-benchmark-simpleclick

# Create package structure
mkdir -p simpleclick/{benchmark,tasks,server}
touch simpleclick/__init__.py
touch simpleclick/benchmark/__init__.py
touch simpleclick/tasks/__init__.py
touch simpleclick/server/__init__.py
```

Your structure should look like:
```
cube-benchmark-simpleclick/
├── simpleclick/
│   ├── __init__.py
│   ├── benchmark/          # Existing benchmark code
│   ├── tasks/              # Task implementations
│   └── server/             # CUBE server wrappers
├── setup.py
├── README.md
└── pyproject.toml
```

### Install Dependencies

Create `pyproject.toml`:

```toml
[project]
name = "cube-benchmark-simpleclick"
version = "1.0.0"
description = "SimpleClick benchmark wrapped for CUBE"
requires-python = ">=3.9"
dependencies = [
    "cube-standard>=0.1.0",
    "playwright>=1.40.0",  # Your existing dependencies
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "black>=23.0",
]
```

Install:
```bash
pip install -e ".[dev]"
```

## Step 2: Implement Task-Level API

The Task-Level API defines how agents interact with a single task instance. We'll implement this as a Python class.

### Create Task Class

Create `simpleclick/tasks/cube_task.py`:

```python
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import base64

@dataclass
class Tool:
    """MCP tool schema."""
    name: str
    description: str
    input_schema: Dict[str, Any]

@dataclass
class Resource:
    """MCP resource schema."""
    uri: str
    name: str
    description: str
    mime_type: str

class SimpleClickTask:
    """CUBE-compliant task implementation."""

    def __init__(self, task_id: str, seed: Optional[int] = None):
        """
        Initialize a task instance.

        Args:
            task_id: ID of the task to load
            seed: Random seed for reproducibility
        """
        self.task_id = task_id
        self.seed = seed

        # Load your existing task implementation
        from ..benchmark import load_task
        self._task = load_task(task_id, seed=seed)

        # Track state
        self._step_count = 0
        self._terminated = False

    # ===== MCP Protocol: Tools (Actions) =====

    def list_tools(self) -> List[Tool]:
        """
        Return available tools/actions for this task.
        This defines what the agent can DO.
        """
        return [
            Tool(
                name="click",
                description="Click at screen coordinates",
                input_schema={
                    "type": "object",
                    "properties": {
                        "x": {
                            "type": "number",
                            "description": "X coordinate in pixels"
                        },
                        "y": {
                            "type": "number",
                            "description": "Y coordinate in pixels"
                        }
                    },
                    "required": ["x", "y"]
                }
            ),
            Tool(
                name="wait",
                description="Wait for a specified duration",
                input_schema={
                    "type": "object",
                    "properties": {
                        "seconds": {
                            "type": "number",
                            "description": "Seconds to wait"
                        }
                    },
                    "required": ["seconds"]
                }
            )
        ]

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool/action.

        Args:
            name: Tool name (from list_tools)
            arguments: Tool arguments (validated against input_schema)

        Returns:
            MCP tool result with content and error status
        """
        try:
            if name == "click":
                x = arguments["x"]
                y = arguments["y"]

                # Call your existing benchmark implementation
                result = self._task.click(x, y)
                self._step_count += 1

                # Check if task is complete
                self._terminated = self._task.is_complete()

                return {
                    "content": [{
                        "type": "text",
                        "text": f"Clicked at ({x}, {y}). Element: {result}"
                    }],
                    "isError": False
                }

            elif name == "wait":
                seconds = arguments["seconds"]
                import time
                time.sleep(seconds)

                return {
                    "content": [{
                        "type": "text",
                        "text": f"Waited {seconds} seconds"
                    }],
                    "isError": False
                }

            else:
                return {
                    "content": [{
                        "type": "text",
                        "text": f"Unknown tool: {name}"
                    }],
                    "isError": True
                }

        except Exception as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Tool execution failed: {str(e)}"
                }],
                "isError": True
            }

    # ===== MCP Protocol: Resources (Observations) =====

    def list_resources(self) -> List[Resource]:
        """
        Return available resources (observations, task info).
        This defines what the agent can SEE.
        """
        return [
            Resource(
                uri="task://description",
                name="Task Description",
                description="The goal of this task",
                mime_type="text/plain"
            ),
            Resource(
                uri="obs://screenshot",
                name="Screenshot",
                description="Current screenshot of the webpage",
                mime_type="image/png"
            ),
            Resource(
                uri="obs://html",
                name="HTML Source",
                description="Current HTML of the webpage",
                mime_type="text/html"
            )
        ]

    def read_resource(self, uri: str) -> Dict[str, Any]:
        """
        Read a specific resource by URI.

        Args:
            uri: Resource URI (from list_resources)

        Returns:
            MCP resource contents
        """
        if uri == "task://description":
            description = self._task.get_description()
            return {
                "contents": [{
                    "uri": uri,
                    "mimeType": "text/plain",
                    "text": description
                }]
            }

        elif uri == "obs://screenshot":
            # Get screenshot from your benchmark
            screenshot_bytes = self._task.get_screenshot()
            screenshot_b64 = base64.b64encode(screenshot_bytes).decode()

            return {
                "contents": [{
                    "uri": uri,
                    "mimeType": "image/png",
                    "blob": screenshot_b64  # Base64 encoded
                }]
            }

        elif uri == "obs://html":
            html = self._task.get_html()
            return {
                "contents": [{
                    "uri": uri,
                    "mimeType": "text/html",
                    "text": html
                }]
            }

        else:
            raise ValueError(f"Unknown resource URI: {uri}")

    # ===== CUBE Extensions: Evaluation =====

    def evaluate(self) -> Dict[str, Any]:
        """
        Get current evaluation state (Gym-style).

        Returns observation, reward, terminated, truncated, and info.
        """
        # Get observation (screenshot + HTML)
        screenshot = self.read_resource("obs://screenshot")
        html = self.read_resource("obs://html")

        observation = {
            "screenshot": screenshot["contents"][0]["blob"],
            "html": html["contents"][0]["text"]
        }

        # Compute reward (your evaluation logic)
        if self._task.is_successful():
            reward = 1.0
        elif self._terminated:
            reward = 0.0
        else:
            reward = 0.0

        # Check termination
        terminated = self._terminated
        truncated = self._step_count >= 20  # Max 20 steps

        # Additional info
        info = {
            "step_count": self._step_count,
            "success": self._task.is_successful() if terminated else False,
            "task_id": self.task_id
        }

        return {
            "observation": observation,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info
        }

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Reset task to initial state.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Initial observation and info
        """
        if seed is not None:
            self.seed = seed

        # Reset your benchmark task
        self._task.reset(seed=self.seed)
        self._step_count = 0
        self._terminated = False

        # Get initial observation
        eval_state = self.evaluate()

        return {
            "observation": eval_state["observation"],
            "info": {
                "task_id": self.task_id,
                "seed": self.seed,
                "description": self._task.get_description()
            }
        }

    def close(self):
        """Cleanup resources."""
        if hasattr(self._task, 'cleanup'):
            self._task.cleanup()
```

### Test Your Task Implementation

Create `test_task.py`:

```python
from simpleclick.tasks.cube_task import SimpleClickTask

def test_task_basic():
    """Test basic task functionality."""
    task = SimpleClickTask(task_id="task-001", seed=42)

    # Test reset
    state = task.reset()
    assert "observation" in state
    assert state["info"]["task_id"] == "task-001"

    # Test tools
    tools = task.list_tools()
    assert len(tools) > 0
    assert any(t.name == "click" for t in tools)

    # Test action
    result = task.call_tool("click", {"x": 100, "y": 100})
    assert not result["isError"]

    # Test evaluation
    eval_state = task.evaluate()
    assert "reward" in eval_state
    assert "terminated" in eval_state

    # Cleanup
    task.close()

if __name__ == "__main__":
    test_task_basic()
    print("✓ Task-Level API working!")
```

Run the test:
```bash
python test_task.py
```

## Step 3: Implement Benchmark-Level API

The Benchmark-Level API manages multiple tasks and shared infrastructure.

Create `simpleclick/server/benchmark_server.py`:

```python
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import uuid
from datetime import datetime

@dataclass
class TaskInfo:
    id: str
    description: str
    difficulty: str
    tags: List[str]

class SimpleClickBenchmark:
    """CUBE-compliant benchmark server."""

    def __init__(self):
        """Initialize benchmark with shared infrastructure."""
        # Start shared services if needed
        # For SimpleClick, we don't have shared infrastructure,
        # but benchmarks like WebArena would start Docker containers here

        self._active_sessions = {}
        self._tasks_catalog = self._load_tasks_catalog()

    def info(self) -> Dict[str, Any]:
        """Return benchmark metadata."""
        return {
            "id": "simpleclick-v1",
            "name": "SimpleClick Benchmark",
            "version": "1.0.0",
            "description": "Simple web clicking tasks for agent evaluation",
            "task_count": len(self._tasks_catalog),
            "authors": ["Your Name"],
            "homepage_url": "https://github.com/yourname/simpleclick",
            "license": "MIT",
            "capabilities": {
                "tool_reconfiguration": False,
                "parallel_tasks": 10,
                "supports_seeds": True,
                "deterministic": False
            },
            "hardware_requirements": {
                "ram_gb": 2,
                "disk_gb": 1,
                "gpu": False
            },
            "estimated_cost": {
                "avg_tokens_per_task": 5000,
                "avg_time_minutes": 5
            }
        }

    def list_tasks(
        self,
        offset: int = 0,
        limit: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """List available tasks with filtering and pagination."""
        tasks = self._tasks_catalog.copy()

        # Apply filters
        if filter:
            if "difficulty" in filter:
                tasks = [t for t in tasks if t.difficulty == filter["difficulty"]]
            if "tags" in filter:
                filter_tags = set(filter["tags"])
                tasks = [t for t in tasks if filter_tags.intersection(t.tags)]

        # Pagination
        total = len(tasks)
        if limit:
            tasks = tasks[offset:offset + limit]
        else:
            tasks = tasks[offset:]

        return {
            "tasks": [
                {
                    "id": t.id,
                    "description": t.description,
                    "difficulty": t.difficulty,
                    "tags": t.tags
                }
                for t in tasks
            ],
            "total": total,
            "offset": offset,
            "limit": limit
        }

    def spawn(
        self,
        task_id: str,
        seed: Optional[int] = None,
        tool_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Spawn a new task instance."""
        # Validate task exists
        if not any(t.id == task_id for t in self._tasks_catalog):
            raise ValueError(f"Task not found: {task_id}")

        # Generate session ID
        session_id = f"session-{uuid.uuid4().hex[:12]}"

        # Create task instance
        from ..tasks.cube_task import SimpleClickTask
        task = SimpleClickTask(task_id=task_id, seed=seed)

        # For local execution, we can just store the task object
        # For remote execution, we'd start an RPC server here
        port = self._allocate_port()

        # Start RPC server (we'll implement this next)
        from cube.server import CubeTaskServer
        server = CubeTaskServer(task, host="localhost", port=port)
        server.start_async()  # Non-blocking start

        # Track session
        self._active_sessions[session_id] = {
            "task": task,
            "server": server,
            "task_id": task_id,
            "port": port,
            "spawned_at": datetime.now()
        }

        return {
            "session_id": session_id,
            "url": f"http://localhost:{port}",
            "info": {
                "task_id": task_id,
                "seed": seed,
                "spawned_at": datetime.now().isoformat()
            }
        }

    def status(self) -> Dict[str, Any]:
        """Get status of running tasks."""
        tasks_status = []
        for session_id, session in self._active_sessions.items():
            uptime = (datetime.now() - session["spawned_at"]).total_seconds()
            tasks_status.append({
                "session_id": session_id,
                "task_id": session["task_id"],
                "status": "running",
                "uptime_seconds": uptime
            })

        return {
            "tasks": tasks_status,
            "benchmark_status": {
                "shared_services_healthy": True,
                "available_task_slots": 10 - len(self._active_sessions)
            }
        }

    def shutdown(self, session_id: Optional[str] = None):
        """Shutdown task instances."""
        if session_id:
            if session_id in self._active_sessions:
                session = self._active_sessions[session_id]
                session["task"].close()
                session["server"].stop()
                del self._active_sessions[session_id]
        else:
            # Shutdown all
            for sid in list(self._active_sessions.keys()):
                self.shutdown(session_id=sid)

    def _load_tasks_catalog(self) -> List[TaskInfo]:
        """Load all available tasks."""
        # In reality, you'd load this from files or database
        return [
            TaskInfo(
                id="task-001",
                description="Click the red Submit button",
                difficulty="easy",
                tags=["button", "single-click"]
            ),
            TaskInfo(
                id="task-002",
                description="Click the Login button in the header",
                difficulty="easy",
                tags=["button", "navigation"]
            ),
            # ... more tasks ...
        ]

    def _allocate_port(self) -> int:
        """Allocate an available port."""
        import socket
        with socket.socket() as s:
            s.bind(('', 0))
            return s.getsockname()[1]
```

## Step 4: Add RPC Server Support

Create `simpleclick/server/rpc_server.py`:

```python
from flask import Flask, request, jsonify
from typing import Any

class CubeTaskServer:
    """RPC server that exposes Task-Level API via HTTP."""

    def __init__(self, task, host="localhost", port=8000):
        self.task = task
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self._setup_routes()

    def _setup_routes(self):
        """Setup HTTP routes for MCP and CUBE methods."""

        @self.app.route("/mcp/tools/list", methods=["POST"])
        def tools_list():
            tools = self.task.list_tools()
            return jsonify({"tools": [t.__dict__ for t in tools]})

        @self.app.route("/mcp/tools/call", methods=["POST"])
        def tools_call():
            data = request.json
            result = self.task.call_tool(
                name=data["name"],
                arguments=data["arguments"]
            )
            return jsonify(result)

        @self.app.route("/mcp/resources/list", methods=["POST"])
        def resources_list():
            resources = self.task.list_resources()
            return jsonify({"resources": [r.__dict__ for r in resources]})

        @self.app.route("/mcp/resources/read", methods=["POST"])
        def resources_read():
            data = request.json
            result = self.task.read_resource(uri=data["uri"])
            return jsonify(result)

        @self.app.route("/cube/evaluation", methods=["POST"])
        def evaluation():
            result = self.task.evaluate()
            return jsonify(result)

        @self.app.route("/cube/reset", methods=["POST"])
        def reset():
            data = request.json or {}
            result = self.task.reset(seed=data.get("seed"))
            return jsonify(result)

        @self.app.route("/cube/close", methods=["POST"])
        def close():
            self.task.close()
            return jsonify({})

    def start(self):
        """Start server (blocking)."""
        self.app.run(host=self.host, port=self.port)

    def start_async(self):
        """Start server in background thread."""
        import threading
        thread = threading.Thread(target=self.start, daemon=True)
        thread.start()

    def stop(self):
        """Stop the server."""
        # Flask doesn't have built-in shutdown, use werkzeug
        pass
```

## Step 5: Create Python Client Helpers

Create `simpleclick/__init__.py`:

```python
"""CUBE SimpleClick Benchmark."""

from .tasks.cube_task import SimpleClickTask
from .server.benchmark_server import SimpleClickBenchmark

__version__ = "1.0.0"

# For local execution
def load_task(task_id: str, seed=None):
    """Load a task for local execution."""
    return SimpleClickTask(task_id=task_id, seed=seed)

# For benchmark server
def create_benchmark():
    """Create benchmark server instance."""
    return SimpleClickBenchmark()
```

## Step 6: Register with CUBE Registry

Create `registry_metadata.json`:

```json
{
  "id": "simpleclick-v1",
  "name": "SimpleClick Benchmark",
  "version": "1.0.0",
  "authors": ["Your Name"],
  "paper": null,
  "package": "cube-benchmark-simpleclick",
  "parent": null,
  "package_license": "MIT",
  "benchmark_license": "MIT",
  "content_notice": null,
  "compliance": ["no-docker-root", "task-isolated"],
  "runtime": "local",
  "hardware": {
    "ram_gb": 2,
    "gpu": false,
    "disk_gb": 1
  },
  "task_count": 10,
  "estimated_tokens": 50000
}
```

Submit to registry:
```bash
cube-registry submit registry_metadata.json
```

## Step 7: Package and Distribute

Update `setup.py`:

```python
from setuptools import setup, find_packages

setup(
    name="cube-benchmark-simpleclick",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "cube-standard>=0.1.0",
        "playwright>=1.40.0",
        "flask>=2.0.0",
    ],
    entry_points={
        "console_scripts": [
            "simpleclick-server=simpleclick.server:main",
        ],
    },
)
```

Build and publish:
```bash
python -m build
python -m twine upload dist/*
```

## Step 8: Test End-to-End

Create `test_e2e.py`:

```python
from simpleclick import load_task, create_benchmark

def test_local_execution():
    """Test local Python execution."""
    task = load_task("task-001", seed=42)
    state = task.reset()

    tools = task.list_tools()
    result = task.call_tool("click", {"x": 100, "y": 100})
    eval_state = task.evaluate()

    assert eval_state["step_count"] == 1
    task.close()
    print("✓ Local execution works!")

def test_benchmark_api():
    """Test benchmark-level API."""
    benchmark = create_benchmark()

    info = benchmark.info()
    assert info["task_count"] == 10

    tasks = benchmark.list_tasks(limit=5)
    assert len(tasks["tasks"]) == 5

    session = benchmark.spawn(task_id="task-001", seed=42)
    assert "session_id" in session
    assert "url" in session

    benchmark.shutdown(session_id=session["session_id"])
    print("✓ Benchmark API works!")

if __name__ == "__main__":
    test_local_execution()
    test_benchmark_api()
    print("\n✅ All tests passed! Your benchmark is CUBE-compliant!")
```

## Next Steps

Congratulations! You've wrapped your benchmark for CUBE. Now:

1. **Test with real agents**: Use your benchmark with different agent frameworks
2. **Add to examples**: Contribute example agents for your benchmark
3. **Improve documentation**: Write task-specific docs
4. **Join the community**: Share your benchmark in discussions
5. **Iterate**: Gather feedback and improve the wrapper

## Common Patterns for Different Benchmark Types

### Pattern: Shared Docker Infrastructure (like WebArena)

```python
class MyBenchmark:
    def __init__(self):
        # Start shared containers once
        self._docker_client = docker.from_env()
        self._containers = self._start_shared_containers()

    def spawn(self, task_id, seed):
        # Create task using shared containers
        # Use snapshots or copy-on-write for isolation
        pass
```

### Pattern: Per-Task Containers (like SWE-bench)

```python
class MyBenchmark:
    def spawn(self, task_id, seed):
        # Start fresh container for this task
        container = docker_client.containers.run(
            image=f"task-{task_id}:latest",
            detach=True
        )
        # Return container endpoint
        pass
```

### Pattern: Live Services (like GAIA)

```python
class MyTask:
    def call_tool(self, name, arguments):
        if name == "web_search":
            # Call real search API
            results = requests.get(f"https://api.search.com?q={arguments['query']}")
            return {"content": [{"type": "text", "text": results.text}]}
```

## Troubleshooting

### Issue: Port conflicts when spawning multiple tasks

**Solution**: Use dynamic port allocation:
```python
import socket

def _allocate_port():
    with socket.socket() as s:
        s.bind(('', 0))
        return s.getsockname()[1]
```

### Issue: Tasks not cleaning up properly

**Solution**: Use context managers:
```python
def __enter__(self):
    return self

def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()
```

### Issue: Large observations slow down evaluation

**Solution**: Lazy loading with resources:
```python
def read_resource(self, uri):
    if uri == "obs://screenshot":
        # Only generate screenshot when requested
        screenshot = self._task.capture_screenshot()
        return {"contents": [{"blob": screenshot}]}
```

## Getting Help

- **API Questions**: [API Reference](../api/)
- **Examples**: Check [existing benchmarks](https://github.com/cube-benchmarks)
- **Community**: [GitHub Discussions](https://github.com/The-AI-Alliance/cube-standard/discussions)
- **Issues**: [Report bugs](https://github.com/The-AI-Alliance/cube-standard/issues)

---

**Congratulations!** You've made your benchmark CUBE-compliant and accessible to the entire community.
