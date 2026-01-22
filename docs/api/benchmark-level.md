---
layout: default
title: Benchmark-Level API
parent: API Reference
nav_order: 2
---

# Benchmark-Level API

The Benchmark-Level API defines how evaluation harnesses discover, spawn, and manage task instances. This layer handles shared infrastructure, resource allocation, and task lifecycle management.

{: .note }
> This API describes benchmark orchestration. For agent-task interaction, see the [Task-Level API](task-level.html).

## Overview

The Benchmark-Level API provides five core capabilities:

1. **Discovery** - Get benchmark metadata and capabilities
2. **Task Listing** - Browse available tasks with filtering/pagination
3. **Task Spawning** - Create new task instances
4. **Health Monitoring** - Check status of running tasks
5. **Lifecycle Management** - Shutdown tasks and cleanup resources

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
  "id": "webarena-verified",
  "name": "WebArena Verified",
  "version": "1.2.0",
  "description": "Realistic web navigation tasks with verifiable gold trajectories",
  "task_count": 812,
  "authors": ["Jane Researcher", "John Developer"],
  "paper_url": "https://arxiv.org/abs/...",
  "homepage_url": "https://webarena.dev",
  "license": "CC-BY-NC-4.0",
  "capabilities": {
    "tool_reconfiguration": true,
    "parallel_tasks": 10,
    "supports_seeds": true,
    "deterministic": false
  },
  "hardware_requirements": {
    "ram_gb": 16,
    "disk_gb": 50,
    "gpu": false
  },
  "estimated_cost": {
    "avg_tokens_per_task": 25000,
    "avg_time_minutes": 15
  }
}
```

**Python interface:**
```python
from cube import LocalRunner

benchmark = LocalRunner("cube-benchmark-webarena")
info = benchmark.info()

print(f"Benchmark: {info.name} v{info.version}")
print(f"Tasks: {info.task_count}")
print(f"RAM required: {info.hardware_requirements.ram_gb}GB")
print(f"Supports parallel execution: {info.capabilities.parallel_tasks} tasks")
```

**Key fields:**

- `id` (string): Unique identifier matching registry
- `name` (string): Human-readable name
- `version` (string): Semantic version
- `task_count` (int): Total number of tasks
- `capabilities` (object): Feature flags (tool reconfiguration, parallelization, etc.)
- `hardware_requirements` (object): Minimum resource requirements
- `estimated_cost` (object): Expected token usage and time per task

### `cube/tasks`

List available tasks with optional filtering and pagination.

**Request:**
```json
{
  "method": "cube/tasks",
  "params": {
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
      "description": "Add items to cart and complete checkout",
      "difficulty": "medium",
      "tags": ["e-commerce", "form-filling", "multi-step"],
      "estimated_steps": 15,
      "metadata": {
        "domain": "e-commerce",
        "requires_payment": false
      }
    },
    {
      "id": "product-search-456",
      "description": "Search for specific product and add to wishlist",
      "difficulty": "medium",
      "tags": ["e-commerce", "search"],
      "estimated_steps": 8,
      "metadata": {
        "domain": "e-commerce"
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
# List all tasks
tasks = benchmark.list_tasks()
print(f"Total tasks: {len(tasks)}")

# Pagination
page1 = benchmark.list_tasks(offset=0, limit=20)
page2 = benchmark.list_tasks(offset=20, limit=20)

# Filtering
medium_tasks = benchmark.list_tasks(
    filter={"difficulty": "medium"}
)

ecommerce_tasks = benchmark.list_tasks(
    filter={"domain": "e-commerce", "tags": ["form-filling"]}
)

# Print task details
for task in medium_tasks:
    print(f"{task.id}: {task.description}")
    print(f"  Difficulty: {task.difficulty}")
    print(f"  Tags: {', '.join(task.tags)}")
```

**Parameters:**

- `offset` (int, optional): Skip first N tasks (default: 0)
- `limit` (int, optional): Return at most N tasks (default: all tasks)
- `filter` (object, optional): Filter criteria (benchmark-specific fields)

**Filter fields** are benchmark-specific but commonly include:
- `difficulty`: Task difficulty level
- `domain`: Task domain/category
- `tags`: List of task tags
- `estimated_steps`: Approximate number of agent actions needed

### `cube/spawn`

Create a new task instance. Returns an endpoint URL where the task exposes the Task-Level API.

**Request:**
```json
{
  "method": "cube/spawn",
  "params": {
    "task_id": "shopping-cart-123",
    "seed": 42,
    "tool_config": {
      "browser_mode": "vision-based"
    }
  }
}
```

**Response:**
```json
{
  "session_id": "task-abc123def456",
  "url": "http://localhost:8001",
  "info": {
    "task_id": "shopping-cart-123",
    "seed": 42,
    "spawned_at": "2026-01-22T10:30:00Z",
    "expires_at": "2026-01-22T12:30:00Z"
  }
}
```

**Python interface:**
```python
# Spawn a task
session = benchmark.spawn(
    task_id="shopping-cart-123",
    seed=42
)

print(f"Task endpoint: {session.url}")
print(f"Session ID: {session.id}")

# Connect to the task instance (exposes Task-Level API)
from cube import RemoteRunner
task = RemoteRunner(session.url)

# Now use Task-Level API
state = task.reset()
tools = task.list_tools()
result = task.call_tool("click", {"x": 100, "y": 100})
eval_state = task.evaluate()

# Cleanup
task.close()

# Or spawn multiple tasks in parallel
sessions = []
for task_id in ["task-1", "task-2", "task-3"]:
    session = benchmark.spawn(task_id=task_id, seed=42)
    sessions.append(session)

# Evaluate them in parallel
# ... your parallel execution logic ...

# Cleanup all
for session in sessions:
    benchmark.shutdown(session_id=session.id)
```

**Parameters:**

- `task_id` (string, required): ID of the task to spawn (from `cube/tasks`)
- `seed` (int, optional): Random seed for reproducibility
- `tool_config` (object, optional): Tool configuration overrides (if benchmark supports it)

**Returns:**

- `session_id` (string): Unique identifier for this task instance
- `url` (string): Endpoint URL exposing the Task-Level API
- `info` (object): Metadata about the spawned task

{: .note }
> The returned `url` is where the task instance runs. Connect to this URL to interact with the task using the Task-Level API.

### `cube/status`

Check the health and status of running task instances.

**Request:**
```json
{
  "method": "cube/status",
  "params": {}
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
      "uptime_seconds": 145,
      "resource_usage": {
        "ram_mb": 512,
        "cpu_percent": 15.2
      }
    },
    {
      "session_id": "task-def456",
      "task_id": "product-search-456",
      "status": "idle",
      "uptime_seconds": 892,
      "resource_usage": {
        "ram_mb": 480,
        "cpu_percent": 2.1
      }
    }
  ],
  "benchmark_status": {
    "shared_services_healthy": true,
    "available_task_slots": 7,
    "total_ram_usage_mb": 4096
  }
}
```

**Python interface:**
```python
status = benchmark.status()

# Check individual tasks
for task_status in status.tasks:
    print(f"Task {task_status.session_id}: {task_status.status}")
    print(f"  Uptime: {task_status.uptime_seconds}s")
    print(f"  RAM: {task_status.resource_usage.ram_mb}MB")

# Check benchmark health
if status.benchmark_status.shared_services_healthy:
    print("✓ Benchmark infrastructure healthy")
    print(f"Available slots: {status.benchmark_status.available_task_slots}")
```

**Status values:**
- `running` - Task is actively being used
- `idle` - Task is spawned but not currently in use
- `error` - Task encountered an error
- `shutting_down` - Task is cleaning up

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
  "shutdown": [
    {
      "session_id": "task-abc123",
      "status": "success"
    }
  ]
}
```

**Python interface:**
```python
# Shutdown a specific task
benchmark.shutdown(session_id="task-abc123")

# Shutdown all running tasks
benchmark.shutdown()

# Common pattern: cleanup in a finally block
session = None
try:
    session = benchmark.spawn(task_id="example")
    task = RemoteRunner(session.url)
    # ... evaluation logic ...
finally:
    if session:
        benchmark.shutdown(session_id=session.id)

# Or use context manager (recommended)
with benchmark.spawn_context(task_id="example") as task:
    # task is already connected and ready to use
    state = task.reset()
    # ... evaluation logic ...
    # Automatic cleanup on exit
```

## Resource Management Patterns

### Pattern 1: Single Task Sequential Execution

Simplest pattern - spawn one task at a time:

```python
benchmark = LocalRunner("cube-benchmark-example")
tasks = benchmark.list_tasks(limit=10)

for task_info in tasks:
    session = benchmark.spawn(task_id=task_info.id, seed=42)
    task = RemoteRunner(session.url)

    try:
        # Evaluate
        state = task.reset()
        # ... agent logic ...
        result = task.evaluate()
    finally:
        task.close()
        benchmark.shutdown(session_id=session.id)
```

### Pattern 2: Parallel Task Execution

Spawn multiple tasks and evaluate in parallel:

```python
import concurrent.futures

def evaluate_task(benchmark, task_id, agent_fn):
    session = benchmark.spawn(task_id=task_id, seed=42)
    task = RemoteRunner(session.url)

    try:
        return agent_fn(task)
    finally:
        task.close()
        benchmark.shutdown(session_id=session.id)

benchmark = LocalRunner("cube-benchmark-example")
tasks = benchmark.list_tasks(limit=50)

# Check max parallelism
info = benchmark.info()
max_parallel = info.capabilities.parallel_tasks

# Evaluate in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
    futures = [
        executor.submit(evaluate_task, benchmark, task.id, my_agent)
        for task in tasks
    ]

    results = [f.result() for f in concurrent.futures.as_completed(futures)]
```

### Pattern 3: Long-Running Persistent Tasks

Keep tasks alive for multiple evaluation episodes:

```python
# Spawn once
session = benchmark.spawn(task_id="persistent-task")
task = RemoteRunner(session.url)

# Evaluate multiple times with different parameters
for trial in range(100):
    state = task.reset(seed=trial)
    # ... evaluation ...
    result = task.evaluate()

    # Don't close - reuse the same instance

# Cleanup after all trials
task.close()
benchmark.shutdown(session_id=session.id)
```

### Pattern 4: Resource-Aware Batching

Batch task execution based on available resources:

```python
benchmark = LocalRunner("cube-benchmark-example")
info = benchmark.info()

# Calculate how many tasks we can run in parallel
available_ram_gb = 32  # Your machine's RAM
task_ram_gb = info.hardware_requirements.ram_gb
batch_size = min(
    available_ram_gb // task_ram_gb,
    info.capabilities.parallel_tasks
)

print(f"Running {batch_size} tasks in parallel")

# Process in batches
tasks = benchmark.list_tasks()
for i in range(0, len(tasks), batch_size):
    batch = tasks[i:i+batch_size]

    # Spawn batch
    sessions = [benchmark.spawn(task_id=t.id) for t in batch]

    # Evaluate batch in parallel
    # ... parallel execution ...

    # Cleanup batch
    for session in sessions:
        benchmark.shutdown(session_id=session.id)
```

## Implementation Guide for Benchmark Authors

### Python Class Implementation

Implement a benchmark server class:

```python
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import uuid

@dataclass
class TaskInfo:
    id: str
    description: str
    difficulty: str
    tags: List[str]

class MyBenchmarkServer:
    """CUBE-compliant benchmark implementation."""

    def __init__(self):
        # Initialize shared infrastructure
        self._shared_services = self._start_shared_services()
        self._active_sessions = {}

    def info(self) -> Dict[str, Any]:
        """Return benchmark metadata."""
        return {
            "id": "my-benchmark",
            "name": "My Benchmark",
            "version": "1.0.0",
            "task_count": len(self._get_all_tasks()),
            "capabilities": {
                "tool_reconfiguration": False,
                "parallel_tasks": 5,
                "supports_seeds": True,
                "deterministic": False
            },
            "hardware_requirements": {
                "ram_gb": 8,
                "disk_gb": 20,
                "gpu": False
            }
        }

    def list_tasks(
        self,
        offset: int = 0,
        limit: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """List available tasks."""
        all_tasks = self._get_all_tasks()

        # Apply filters
        if filter:
            all_tasks = self._filter_tasks(all_tasks, filter)

        # Apply pagination
        total = len(all_tasks)
        tasks = all_tasks[offset:offset+limit if limit else None]

        return {
            "tasks": [t.__dict__ for t in tasks],
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
        # Generate unique session ID
        session_id = f"task-{uuid.uuid4().hex}"

        # Create task instance
        from .task import MyBenchmarkTask
        task = MyBenchmarkTask(
            task_id=task_id,
            seed=seed,
            tool_config=tool_config,
            shared_services=self._shared_services
        )

        # Start RPC server for this task
        from cube.server import CubeTaskServer
        port = self._allocate_port()
        server = CubeTaskServer(task, host="localhost", port=port)
        server.start_async()

        # Track session
        self._active_sessions[session_id] = {
            "task": task,
            "server": server,
            "task_id": task_id,
            "port": port
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
            tasks_status.append({
                "session_id": session_id,
                "task_id": session["task_id"],
                "status": "running",
                "uptime_seconds": session["server"].uptime()
            })

        return {
            "tasks": tasks_status,
            "benchmark_status": {
                "shared_services_healthy": self._check_health(),
                "available_task_slots": 5 - len(self._active_sessions)
            }
        }

    def shutdown(self, session_id: Optional[str] = None):
        """Shutdown task instances."""
        if session_id:
            # Shutdown specific task
            if session_id in self._active_sessions:
                session = self._active_sessions[session_id]
                session["task"].close()
                session["server"].stop()
                del self._active_sessions[session_id]
        else:
            # Shutdown all tasks
            for sid in list(self._active_sessions.keys()):
                self.shutdown(session_id=sid)

    def _get_all_tasks(self) -> List[TaskInfo]:
        """Get all available tasks."""
        # Your logic to enumerate tasks
        pass

    def _filter_tasks(self, tasks: List[TaskInfo], filter: Dict) -> List[TaskInfo]:
        """Apply filter criteria to tasks."""
        # Your filtering logic
        pass

    def _start_shared_services(self):
        """Initialize shared infrastructure."""
        # Start Docker containers, VMs, databases, etc.
        pass

    def _check_health(self) -> bool:
        """Check if shared services are healthy."""
        # Health check logic
        pass

    def _allocate_port(self) -> int:
        """Allocate an available port for task instance."""
        # Port allocation logic
        pass
```

### Exposing as RPC Server

```python
from cube.server import CubeBenchmarkServer

benchmark = MyBenchmarkServer()
server = CubeBenchmarkServer(benchmark, host="0.0.0.0", port=8000)
server.start()
```

## Next Steps

- **[Task-Level API](task-level.html)**: Learn the agent-task interaction protocol
- **[Package-Level Standard](package-level.html)**: Learn deployment requirements
- **[Benchmark Author Guide](../guides/benchmark-authors.html)**: Complete implementation tutorial
