# APIs


## Benchmark-Level API

The Benchmark-Level API defines how evaluation harnesses discover, spawn, and manage task instances. This layer handles shared infrastructure, resource allocation, and task lifecycle management.

## `cube/info`

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
  "name": "...",
  "version": "...",
  "description": "...",
  "authors": ["Jane Researcher", "John Developer"],
  "license": "CC-BY-NC-4.0",
  "requirements": {
    "ram_gb": 16,
    "disk_gb": 50,
    "gpu": false
  },
  "num_tasks": 10,
  "other": {
    "...": "..."
  }
}
```

## `cube/tasks`

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
      "id": "...",
      "description": "...",
      "tags": [],
      "other": {
        "domain": "...",
        "difficulty": "...",
        "estimated_steps": 15,
      }
    },
  ],
  "total": 1,
  "offset": 0,
  "limit": -1
}
```

## `cube/spawn`

Create a new task instance. Returns an endpoint URL where the task exposes the Task-Level API.

**Request:**
```json
{
  "method": "cube/spawn",
  "params": {
    "task_id": "...",
    "seed": 0,
  }
}
```

NOTE: tool_config will be defined in the benchmark.start() step. This will spin up the RCP server hosting the benchmark with the tools provided.

**Response:**
```json
{
  "url": "http://localhost:8001",
  "session_id": "task-abc123def456",
  "other": {
    "task_id": "...",
    "seed": 0,
    "spawned_at": "...",
    "expires_at": "..."
  }
}
```

TODO: Aman check the ones below

## `cube/status`

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

## `cube/shutdown`

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

## Task-Level API

The Task-Level API defines how agents interact with individual task instances. It combines the Model Context Protocol (MCP) for action execution with CUBE extensions for evaluation semantics.

TODO: Aman copy paste the ones from
https://github.com/The-AI-Alliance/cube-standard/blob/1eaeda50f59c31ba728614bb79eb9990017698ed/docs/api/task-level.md

