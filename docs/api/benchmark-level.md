---
layout: default
title: Benchmark-Level API
parent: API Reference
nav_order: 2
---

# Benchmark-Level API

The Benchmark-Level API defines how evaluation harnesses discover, spawn, and manage task instances. This layer handles shared infrastructure, resource allocation, and task lifecycle management.

{: .note }
> This API describes benchmark orchestration. For agent-task interaction, see the [Task-Level API]({{site.baseurl}}/api/task-level).

## Overview

The Benchmark-Level API provides five core capabilities:

1. **Discovery** - Get benchmark metadata and capabilities
2. **Task Listing** - Browse available tasks with filtering/pagination
3. **Task Spawning** - Create new task instances
4. **Health Monitoring** - Check status of running tasks
5. **Lifecycle Management** - Shutdown tasks and cleanup resources

## Python Interface

CUBE packages provide a Python interface for instantiation and usage:

```python
from cube_webarena import WebArenaBenchmark, WebArenaToolConfig

# Instantiate the benchmark
benchmark = WebArenaBenchmark()
tool_config = WebArenaToolConfig()

# Deploy the benchmark on local server
endpoint = benchmark.setup(available_ports=[8000, 8001, 8002], tool_config=tool_config)

# From this point on, all benchmark-level endpoints are available:
# 1. Through Python API: benchmark.spawn(...)
# 2. Through HTTP API: POST to endpoint/cube/spawn
```

CUBE benchmarks inherit from the abstract `Benchmark` class and implement required methods:

```python
from cube import Benchmark
from cube.apis.benchmark import (
    BenchmarkMetadata, SpawnRequest, SpawnResponse,
    StatusRequest, StatusResponse, ShutdownRequest, ShutdownResponse
)
from cube.tool import ToolConfig

class WebArenaToolConfig(ToolConfig):
    def make(self):
      # TODO
      pass

class WebArenaBenchmark(Benchmark):
    """WebArena benchmark implementation."""

    metadata = BenchmarkMetadata(
        name="WebArena",
        version="1.0.0",
        description="Web navigation benchmark",
        authors=["Authors"],
        license="MIT",
        requirements={"ram_gb": 16},
        num_tasks=812,
        tags=["web", "navigation"],
        other={}
    )

    def setup(self, available_ports: list[int], tool_config: ToolConfig) -> str:
        """Initialize shared infrastructure and start benchmark server."""
        # Initialize shared infrastructure
        # Start benchmark server on available port
        # Return endpoint URL
        self.tool_config = tool_config
        pass

    def load_tasks(self) -> list[Task]:
        """Return list of Task objects."""
        pass

    def spawn(self, request: SpawnRequest) -> SpawnResponse:
        """Spawn task instance."""
        pass

    def get_task_status(self, request: StatusRequest) -> StatusResponse:
        """Return status of running tasks."""
        pass

    def shutdown(self, request: ShutdownRequest) -> ShutdownResponse:
        """Shutdown task sessions and cleanup."""
        pass

    def close(self):
        """Cleanup shared resources."""
        pass
```

Benchmarks expose both Python and HTTP/JSON-RPC interfaces for all operations.

## Why a Separate Benchmark Layer?

Many benchmarks require **shared infrastructure** that spans multiple tasks:

- **WebArena**: A persistent "micro-internet" of websites (GitLab, Reddit, e-commerce)
- **OSWorld**: A full desktop operating system with installed applications
- **SWE-bench**: Shared Docker image layers and repository caches

Starting this infrastructure for every single task would be prohibitively expensive. The Benchmark-Level API allows benchmarks to:

- Maintain persistent shared resources
- Spawn lightweight task instances that share infrastructure
- Use copy-on-write or snapshotting for efficient state isolation
- Manage resource allocation across concurrent evaluations

## API Methods

### `cube/info`

Get metadata about the benchmark including name, version, capabilities, and resource requirements.

**Request:**
```json
{
  "method": "cube/info",
  "params": {}
}
```

**Response:**
```json
{
  "name": "WebArena Verified",
  "version": "1.2.0",
  "description": "Realistic web navigation tasks with verifiable gold trajectories",
  "authors": ["Jane Researcher", "John Developer"],
  "license": "CC-BY-NC-4.0",
  "requirements": {
    "ram_gb": 16,
    "disk_gb": 50,
    "gpu": false
  },
  "num_tasks": 812,
  "tags": ["web", "navigation", "gui"],
  "other": {
    "paper_url": "https://arxiv.org/abs/...",
    "homepage_url": "https://webarena.dev",
    "capabilities": {
      "tool_reconfiguration": true,
      "parallel_tasks": 10,
      "supports_seeds": true,
      "deterministic": false
    },
    "estimated_cost": {
      "avg_tokens_per_task": 25000,
      "avg_time_minutes": 15
    }
  }
}
```

**Python interface:**
```python
from cube_webarena import WebArenaBenchmark, WebArenaToolConfig

# Instantiate the benchmark
benchmark = WebArenaBenchmark()

# Access info via Python API (no setup required for metadata)
info = benchmark.info()
print(f"Benchmark: {info.name} v{info.version}")
print(f"Tasks: {info.num_tasks}")
print(f"RAM required: {info.requirements.get('ram_gb', 0)}GB")
print(f"Capabilities: {info.other}")

# Or deploy benchmark and use HTTP endpoint
tool_config = WebArenaToolConfig()
endpoint = benchmark.setup(available_ports=[8000, 8001, 8002], tool_config=tool_config)

import requests
response = requests.post(f"{endpoint}/cube/info", json={"method": "cube/info", "params": {}})
info_data = response.json()
```

**Key fields:**

- `name` (string, required): Human-readable name
- `version` (string, required): Semantic version
- `description` (string, required): Benchmark description
- `authors` (list[string], default: []): List of benchmark author names
- `license` (string, default: ""): Benchmark license
- `requirements` (object, default: {}): Hardware requirements to install and run the benchmark
- `num_tasks` (int, default: 0): Total number of tasks
- `tags` (list[string], default: []): Benchmark tags
- `other` (object, default: {}): Additional metadata (capabilities, URLs, estimated costs, etc.)

### `cube/tasks`

List available tasks with optional filtering and pagination.

**Request:**
```json
{
  "method": "cube/tasks",
  "params": {
    "task_id": null,
    "offset": 0,
    "limit": 10,
    "filter": {
      "difficulty": "medium",
      "domain": "e-commerce",
      "tags": ["form-filling"]
    }
  }
}
```

**Response:**
```json
{
  "tasks": [
    {
      "id": "shopping-cart-123",
      "seed": null,
      "description": "Add items to cart and complete checkout",
      "tags": ["e-commerce", "form-filling", "multi-step"],
      "max_steps": 20,
      "difficulty": "medium",
      "domain": "e-commerce",
      "other": {
        "estimated_steps": 15,
        "requires_payment": false
      }
    },
    {
      "id": "product-search-456",
      "seed": null,
      "description": "Search for specific product and add to wishlist",
      "tags": ["e-commerce", "search"],
      "max_steps": 15,
      "difficulty": "medium",
      "domain": "e-commerce",
      "other": {
        "estimated_steps": 8
      }
    }
  ],
  "total": 156,
  "offset": 0,
  "limit": 10
}
```

**Python interface:**
```python
from cube_webarena import WebArenaBenchmark, WebArenaToolConfig
from cube.apis.benchmark import TaskRequest

# Instantiate the benchmark
benchmark = WebArenaBenchmark()

# Setup benchmark (required for loading tasks)
tool_config = WebArenaToolConfig()
endpoint = benchmark.setup(available_ports=[8000, 8001, 8002], tool_config=tool_config)

# List all tasks via Python API
request = TaskRequest()
result = benchmark.list_tasks(request)
print(f"Total tasks: {result.total}")

# Get specific task
request = TaskRequest(task_id="shopping-cart-123")
task_result = benchmark.list_tasks(request)

# Pagination
request = TaskRequest(offset=0, limit=20)
page1 = benchmark.list_tasks(request)

request = TaskRequest(offset=20, limit=20)
page2 = benchmark.list_tasks(request)

# Filtering (benchmark-specific)
request = TaskRequest(filter={"difficulty": "medium"})
medium_tasks = benchmark.list_tasks(request)

# Print task details
for task in medium_tasks.tasks:
    print(f"{task.id}: {task.description}")
    print(f"  Tags: {', '.join(task.tags)}")
    if 'difficulty' in task.other:
        print(f"  Difficulty: {task.other['difficulty']}")

# Or via HTTP endpoint
import requests
response = requests.post(
    f"{endpoint}/cube/tasks",
    json={
        "method": "cube/tasks",
        "params": {"offset": 0, "limit": 10}
    }
)
tasks_data = response.json()
```

**Parameters:**

- `task_id` (string, optional): Unique task identifier. If provided, fetches only that specific task (default: None)
- `offset` (int, optional): Skip first N tasks (default: 0)
- `limit` (int, optional): Return at most N tasks. Use -1 for no limit (default: -1)
- `filter` (object, optional): Filter criteria (benchmark-specific fields, default: {})

**Response fields:**

Each task in the response has:

- `id` (string, required): Unique task identifier
- `seed` (int | null, optional): Random seed for the task, if applicable
- `description` (string, default: ""): Task description
- `tags` (list[string], default: []): List of task tags
- `max_steps` (int | null, optional): Maximum number of steps allowed
- `difficulty` (string | null, optional): Task difficulty level
- `domain` (string | null, optional): Task domain (e.g., 'web', 'coding')
- `other` (object, default: {}): Additional task metadata

**Filter fields** are benchmark-specific but commonly include:
- `difficulty`: Task difficulty level
- `domain`: Task domain/category
- `tags`: List of task tags

### `cube/spawn`

Create a new task instance. Returns an endpoint URL where the task exposes the Task-Level API.

**Request:**
```json
{
  "method": "cube/spawn",
  "params": {
    "task_id": "shopping-cart-123",
    "seed": 42
  }
}
```

**Response:**
```json
{
  "url": "http://localhost:8001",
  "session_id": "task-abc123def456",
  "other": {
    "task_id": "shopping-cart-123",
    "seed": 42,
    "spawned_at": "2026-01-22T10:30:00Z",
    "expires_at": "2026-01-22T12:30:00Z"
  }
}
```

**Python interface:**
```python
from cube_webarena import WebArenaBenchmark, WebArenaToolConfig
from cube.apis.benchmark import SpawnRequest, ShutdownRequest

# Instantiate the benchmark
benchmark = WebArenaBenchmark()

# Setup benchmark (required for spawning tasks)
tool_config = WebArenaToolConfig()
endpoint = benchmark.setup(available_ports=[8000, 8001, 8002], tool_config=tool_config)

# Spawn a task via Python API
request = SpawnRequest(task_id="shopping-cart-123", seed=42)
session = benchmark.spawn(request)

print(f"Task endpoint: {session.url}")  # exposes Task-Level API
print(f"Session ID: {session.session_id}")


# Or via HTTP endpoint
response = requests.post(
    f"{endpoint}/cube/spawn",
    json={
        "method": "cube/spawn",
        "params": {"task_id": "shopping-cart-123", "seed": 42}
    }
)
session_data = response.json()
```

**Parameters:**

- `task_id` (string, required): ID of the task to spawn (from `cube/tasks`)
- `seed` (int, optional): Random seed for reproducibility (default: None)

**Returns:**

- `url` (string, required): Endpoint URL exposing the Task-Level API
- `session_id` (string, required): Unique identifier for this task instance
- `other` (object, default: {}): Additional session information (spawned_at, expires_at, etc.)

{: .note }
> The returned `url` is where the task instance runs. Connect to this URL to interact with the task using the Task-Level API.

### `cube/status`

Check the health and status of running task instances.

**Request:**
```json
{
  "method": "cube/status",
  "params": {
    "session_id": null,
    "offset": 0,
    "limit": -1,
    "filter": {}
  }
}
```

**Response:**
```json
{
  "tasks": [
    {
      "session_id": "task-abc123",
      "task_id": "shopping-cart-123",
      "status": "running",
      "created_at": "2026-01-22T10:30:00Z",
      "step_count": 5,
      "last_updated": "2026-01-22T10:32:25Z",
      "other": {
        "uptime_seconds": 145,
        "resource_usage": {
          "ram_mb": 512,
          "cpu_percent": 15.2
        }
      }
    },
    {
      "session_id": "task-def456",
      "task_id": "product-search-456",
      "status": "running",
      "created_at": "2026-01-22T09:15:00Z",
      "step_count": 12,
      "last_updated": "2026-01-22T09:29:52Z",
      "other": {
        "uptime_seconds": 892,
        "resource_usage": {
          "ram_mb": 480,
          "cpu_percent": 2.1
        }
      }
    }
  ]
}
```

**Python interface:**
```python
from cube_webarena import WebArenaBenchmark, WebArenaToolConfig
from cube.apis.benchmark import StatusRequest

# Instantiate the benchmark
benchmark = WebArenaBenchmark()

# Setup benchmark (required before spawning tasks)
tool_config = WebArenaToolConfig()
endpoint = benchmark.setup(available_ports=[8000, 8001, 8002], tool_config=tool_config)

# Get status of all tasks via Python API
request = StatusRequest()
status = benchmark.get_task_status(request)

# Get status of a specific session
request = StatusRequest(session_id="task-abc123")
status = benchmark.get_task_status(request)

# With pagination and filtering
request = StatusRequest(offset=0, limit=10, filter={"status": "running"})
status = benchmark.get_task_status(request)

# Check individual tasks
for task_status in status.tasks:
    print(f"Task {task_status.session_id}: {task_status.status}")
    print(f"  Created at: {task_status.created_at}")
    print(f"  Step count: {task_status.step_count}")
    if task_status.last_updated:
        print(f"  Last updated: {task_status.last_updated}")
    if 'uptime_seconds' in task_status.other:
        print(f"  Uptime: {task_status.other['uptime_seconds']}s")
    if 'resource_usage' in task_status.other:
        print(f"  RAM: {task_status.other['resource_usage']['ram_mb']}MB")

# Or via HTTP endpoint (benchmark.get_task_status() and POST to endpoint/cube/status do the same thing)
import requests
response = requests.post(
    f"{endpoint}/cube/status",
    json={"method": "cube/status", "params": {}}
)
status_data = response.json()
```

**Parameters:**

- `session_id` (string, optional): Unique task session identifier. If provided, fetches only that specific session (default: None)
- `offset` (int, optional): Skip first N tasks (default: 0)
- `limit` (int, optional): Return at most N tasks. Use -1 for no limit (default: -1)
- `filter` (object, optional): Filter criteria (benchmark-specific fields, default: {})

**Response fields:**

Each task status has:

- `session_id` (string, required): Session identifier
- `task_id` (string, required): Task identifier
- `status` (string, required): Task status (one of: "running", "stopped", "error")
- `created_at` (datetime, required): Session creation timestamp
- `step_count` (int, default: 0): Number of steps executed
- `last_updated` (datetime | null, optional): Last update timestamp
- `other` (object, default: {}): Additional status information (uptime_seconds, resource_usage, etc.)

**Status values:**

- `running` - Task is running
- `stopped` - Task has been stopped
- `error` - Task encountered an error

### `cube/shutdown`

Shutdown task instances and cleanup resources.

**Request (shutdown specific task):**
```json
{
  "method": "cube/shutdown",
  "params": {
    "session_id": "task-abc123"
  }
}
```

**Request (shutdown all tasks):**
```json
{
  "method": "cube/shutdown",
  "params": {}
}
```

**Response:**
```json
{
  "success": true,
  "cleaned": ["task-abc123"]
}
```

**Python interface:**
```python
from cube_webarena import WebArenaBenchmark, WebArenaToolConfig
from cube.apis.benchmark import ShutdownRequest, SpawnRequest

# Instantiate the benchmark
benchmark = WebArenaBenchmark()

# Setup benchmark
tool_config = WebArenaToolConfig()
endpoint = benchmark.setup(available_ports=[8000, 8001, 8002], tool_config=tool_config)

# Shutdown a specific task via Python API
request = ShutdownRequest(session_id="task-abc123")
result = benchmark.shutdown(request)

# Shutdown all running tasks
request = ShutdownRequest()
result = benchmark.shutdown(request)
print(f"Successfully cleaned up {len(result.cleaned)} sessions")


# Or via HTTP endpoint (benchmark.shutdown() and POST to endpoint/cube/shutdown do the same thing)
import requests
response = requests.post(
    f"{endpoint}/cube/shutdown",
    json={"method": "cube/shutdown", "params": {"session_id": "task-abc123"}}
)
result = response.json()
```

**Returns:**

- `success` (bool, required): Whether shutdown was successful
- `cleaned` (list[string], required): List of session IDs that were cleaned up

## Implementation Guide for Benchmark Authors

### Python Class Implementation

CUBE benchmarks must inherit from the abstract `Benchmark` class and implement required methods:

```python
from cube import Benchmark, Task
from cube.apis.benchmark import (
    BenchmarkMetadata, TaskRequest, TaskListResponse,
    SpawnRequest, SpawnResponse,
    StatusRequest, StatusResponse,
    ShutdownRequest, ShutdownResponse
)
from cube.tool import ToolConfig
import uuid

class MyBenchmark(Benchmark):
    """CUBE-compliant benchmark implementation."""

    # Define benchmark metadata
    metadata = BenchmarkMetadata(
        name="My Benchmark",
        version="1.0.0",
        description="A CUBE-compliant benchmark implementation",
        authors=["Benchmark Author"],
        license="MIT",
        requirements={
            "ram_gb": 8,
            "disk_gb": 20,
            "gpu": False
        },
        num_tasks=100,  # Update with actual count
        tags=["example", "tutorial"],
        other={
            "capabilities": {
                "parallel_tasks": 5,
                "supports_seeds": True
            }
        }
    )

    def __init__(self):
        super().__init__()
        self._shared_services = None
        self._active_sessions = {}
        self._tasks = []
        self._benchmark_server = None

    def setup(self, available_ports: list[int], tool_config: ToolConfig) -> str:
        """Initialize shared infrastructure and deploy benchmark server."""
        # Store tool config
        self.tool_config = tool_config

        # Start Docker containers, VMs, databases, etc.
        self._shared_services = self._start_shared_services()

        # Start benchmark server on an available port
        from cube.server import CubeBenchmarkServer
        benchmark_port = available_ports[0]
        self._benchmark_server = CubeBenchmarkServer(self, host="localhost", port=benchmark_port)
        self._benchmark_server.start_async()

        # Return endpoint URL
        endpoint = f"http://localhost:{benchmark_port}"
        return endpoint

    def close(self):
        """Clean up shared resources."""
        # Stop benchmark server
        if self._benchmark_server:
            self._benchmark_server.stop()

        # Stop shared services
        if self._shared_services:
            self._stop_shared_services()

    def load_tasks(self) -> list[Task]:
        """Load and return the list of tasks for this benchmark."""
        if not self._tasks:
            # Load tasks from disk, database, etc.
            self._tasks = self._load_tasks_from_source()
        return self._tasks

    def spawn(self, request: SpawnRequest) -> SpawnResponse:
        """Spawn a new task instance."""
        # Generate unique session ID
        session_id = f"task-{uuid.uuid4().hex}"

        # Find the task
        tasks = self.load_tasks()
        task = next((t for t in tasks if t.id == request.task_id), None)
        if not task:
            raise ValueError(f"Task {request.task_id} not found")

        # Start task server on a new port
        from cube.server import CubeTaskServer
        port = self._allocate_port()
        server = CubeTaskServer(task, host="localhost", port=port)
        server.start_async()

        # Track session
        from datetime import datetime
        self._active_sessions[session_id] = {
            "task": task,
            "server": server,
            "task_id": request.task_id,
            "port": port,
            "created_at": datetime.now()
        }

        return SpawnResponse(
            url=f"http://localhost:{port}",
            session_id=session_id,
            other={
                "task_id": request.task_id,
                "seed": request.seed,
                "spawned_at": datetime.now().isoformat()
            }
        )

    def get_task_status(self, request: StatusRequest) -> StatusResponse:
        """Get status of running tasks."""
        from cube.apis.benchmark import TaskStatus, TaskStatusEnum

        tasks_status = []
        sessions = self._active_sessions

        # Filter to specific session if requested
        if request.session_id:
            sessions = {request.session_id: self._active_sessions[request.session_id]} if request.session_id in self._active_sessions else {}

        # Build status list
        for sid, session in sessions.items():
            tasks_status.append(TaskStatus(
                session_id=sid,
                task_id=session["task_id"],
                status=TaskStatusEnum.running,
                created_at=session["created_at"],
                step_count=session.get("step_count", 0),
                last_updated=session.get("last_updated"),
                other={
                    "uptime_seconds": (datetime.now() - session["created_at"]).total_seconds()
                }
            ))

        # Apply pagination
        if request.limit == -1:
            paginated_tasks = tasks_status[request.offset:]
        else:
            paginated_tasks = tasks_status[request.offset:request.offset+request.limit]

        return StatusResponse(tasks=paginated_tasks)

    def shutdown(self, request: ShutdownRequest) -> ShutdownResponse:
        """Shutdown task instances."""
        cleaned = []

        if request.session_id:
            # Shutdown specific task
            if request.session_id in self._active_sessions:
                session = self._active_sessions[request.session_id]
                session["server"].stop()
                del self._active_sessions[request.session_id]
                cleaned.append(request.session_id)
        else:
            # Shutdown all tasks
            for sid in list(self._active_sessions.keys()):
                sub_request = ShutdownRequest(session_id=sid)
                result = self.shutdown(sub_request)
                cleaned.extend(result.cleaned)

        return ShutdownResponse(success=True, cleaned=cleaned)

    # Helper methods
    def _load_tasks_from_source(self) -> list[Task]:
        """Load tasks from your data source."""
        # Your logic to load tasks
        pass

    def _start_shared_services(self):
        """Initialize shared infrastructure."""
        # Start Docker containers, VMs, databases, etc.
        pass

    def _stop_shared_services(self):
        """Stop shared infrastructure."""
        # Stop Docker containers, VMs, databases, etc.
        pass

    def _allocate_port(self) -> int:
        """Allocate an available port for task instance."""
        import socket
        with socket.socket() as s:
            s.bind(('', 0))
            return s.getsockname()[1]
```

### Usage Example

Once implemented, users can use your benchmark like this:

```python
from my_cube import MyBenchmark
from cube.apis.benchmark import SpawnRequest, ShutdownRequest
from cube.tool import ToolConfig
import requests

# Instantiate the benchmark
benchmark = MyBenchmark()

# Deploy the benchmark on local server
tool_config = ToolConfig()
endpoint = benchmark.setup(available_ports=[8000, 8001, 8002], tool_config=tool_config)

print(f"Benchmark server running at: {endpoint}")

# Use Python API
spawn_request = SpawnRequest(task_id="task-1", seed=42)
session = benchmark.spawn(spawn_request)
print(f"Task running at: {session.url}")

# Or use HTTP API (both do the same thing)
response = requests.post(
    f"{endpoint}/cube/spawn",
    json={
        "method": "cube/spawn",
        "params": {"task_id": "task-1", "seed": 42}
    }
)
session_data = response.json()
print(f"Task running at: {session_data['url']}")

# Cleanup
shutdown_request = ShutdownRequest(session_id=session.session_id)
benchmark.shutdown(shutdown_request)
benchmark.close()
```

## Next Steps

- **[Task-Level API]({{site.baseurl}}/api/task-level)**: Learn the agent-task interaction protocol
- **[Package-Level Standard]({{site.baseurl}}/api/package-level)**: Learn deployment requirements
- **[Benchmark Author Guide]({{site.baseurl}}/guides/benchmark-authors)**: Complete implementation tutorial
