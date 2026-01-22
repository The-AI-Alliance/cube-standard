---
layout: default
title: API Reference
nav_order: 40
has_children: true
---

# API Reference

The CUBE standard defines four layers of API specifications. Each layer serves a distinct purpose and can be implemented independently.

## API Layers Overview

| Layer | Purpose | Audience |
|-------|---------|----------|
| [Task Level](task-level.html) | Define how agents interact with individual task instances | Agent developers, benchmark authors |
| [Benchmark Level](benchmark-level.html) | Define task discovery, spawning, and lifecycle management | Platform developers, benchmark authors |
| [Package Level](package-level.html) | Define installation, deployment, and parallelization | Infrastructure engineers, benchmark authors |
| [Registry](registry.html) | Define metadata schema for benchmark discovery | All users, registry maintainers |

## Quick Reference

### Task-Level API

The agent-environment interaction layer. Combines MCP for actions and CUBE for evaluation.

**Key methods**:
- `tools/list` - Discover available actions
- `tools/call` - Execute an action
- `resources/read` - Read observations and task descriptions
- `cube/evaluation` - Get reward, termination, and info
- `cube/reset` - Reset task to initial state
- `cube/close` - Cleanup task resources

**[Full Task-Level Specification →](task-level.html)**

### Benchmark-Level API

The task orchestration layer. Manages shared infrastructure and task spawning.

**Key methods**:
- `cube/info` - Get benchmark metadata
- `cube/tasks` - List available tasks with pagination/filtering
- `cube/spawn` - Create a new task instance
- `cube/status` - Check health of running tasks
- `cube/shutdown` - Cleanup task instances

**[Full Benchmark-Level Specification →](benchmark-level.html)**

### Package-Level Standard

The deployment and installation layer. Defines how benchmarks are packaged and distributed.

**Key elements**:
- Installation requirements (pip, Docker, etc.)
- Resource requirements (RAM, GPU, disk)
- Parallelization support
- Deployment models (local, containerized, remote)

**[Full Package-Level Specification →](package-level.html)**

### Registry Standard

The discovery and metadata layer. Enables filtering and automated installation.

**Key fields**:
- Identification (id, name, version)
- Licensing (package_license, benchmark_license, content_notice)
- Requirements (runtime, hardware)
- Economics (task_count, estimated_tokens)

**[Full Registry Specification →](registry.html)**

## Design Principles

### 1. Layered Architecture

Each API layer builds on the previous one but remains independently useful:

```
Registry Layer
    ↓
Package Layer
    ↓
Benchmark Layer
    ↓
Task Layer
```

A simple benchmark might only implement the Task Layer (single static task). A complex benchmark suite implements all layers.

### 2. Build on Standards

CUBE doesn't reinvent protocols:

- **Task actions**: [Model Context Protocol (MCP)](https://modelcontextprotocol.io/)
- **Environment stepping**: [Gymnasium API](https://gymnasium.farama.org/)
- **Packaging**: Standard Python packaging (PyPI, pip)

### 3. Python-First with RPC Fallback

All APIs have two representations:

1. **Python class interface** - For local, in-process execution
2. **RPC/HTTP interface** - For remote or sandboxed execution

The APIs are **1:1 equivalent**. Switching between local and remote requires no code changes beyond initial connection.

### 4. Progressive Disclosure

APIs are designed for gradual learning:

- **Basic usage**: Just call methods, sensible defaults
- **Advanced usage**: Full control over seeds, tool configs, resource limits
- **Expert usage**: Extend via custom protocols

## Implementation Patterns

### For Benchmark Authors

You implement CUBE by providing:

1. **A Python class** with Task-Level methods
2. **A benchmark server** (optional) with Benchmark-Level methods
3. **A package setup** following Package-Level requirements
4. **Registry metadata** for discovery

See the [Benchmark Author Guide](../guides/benchmark-authors.html) for step-by-step instructions.

### For Platform Developers

You integrate CUBE by:

1. **Connecting to benchmarks** via the Benchmark-Level API
2. **Spawning tasks** and getting their endpoints
3. **Interacting with tasks** via the Task-Level API
4. **Filtering benchmarks** via the Registry API

See the [Platform Developer Guide](../guides/platform-developers.html) for integration patterns.

### For End Users (Researchers)

You use CUBE by:

1. **Discovering benchmarks** via the Registry
2. **Installing benchmarks** via pip
3. **Running evaluations** using Task-Level methods

See the [Quick Start Guide](../quickstart.html) for examples.

## API Conventions

### Naming

- **Namespace separation**: `mcp/*` for MCP methods, `cube/*` for CUBE extensions
- **Resource URIs**: Use standard URI format (e.g., `task://description`, `obs://current`)
- **Kebab-case**: API endpoint names use kebab-case (e.g., `task-level-api`)
- **Snake_case**: Python parameters use snake_case (e.g., `task_id`, `tool_config`)

### Error Handling

All API methods should return structured errors:

```python
{
    "error": {
        "code": "TASK_NOT_FOUND",
        "message": "Task 'invalid-id' does not exist",
        "details": {
            "available_tasks": ["task-1", "task-2", "task-3"]
        }
    }
}
```

Common error codes:
- `TASK_NOT_FOUND` - Requested task doesn't exist
- `TOOL_EXECUTION_FAILED` - Action failed during execution
- `RESOURCE_UNAVAILABLE` - Insufficient resources to spawn task
- `INVALID_PARAMETER` - Invalid argument provided
- `SESSION_EXPIRED` - Task instance was shutdown

### Versioning

API versions are specified in:
- **Package version**: Semantic versioning (e.g., `1.2.3`)
- **API version header**: `CUBE-Version: 1.0`
- **Backward compatibility**: Minor versions are backward compatible

Breaking changes require major version bump and deprecation period.

### Type System

All APIs use standard JSON types with optional JSON Schema validation:

- **Primitives**: `string`, `number`, `boolean`, `null`
- **Collections**: `array`, `object`
- **Rich types**: ISO timestamps, URIs, base64-encoded binary

Python implementations should use type hints:

```python
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class TaskInfo:
    id: str
    description: str
    difficulty: Optional[float] = None
    tags: List[str] = None
```

## Next Steps

- **[Task-Level API](task-level.html)**: Learn the agent-environment interaction protocol
- **[Benchmark-Level API](benchmark-level.html)**: Learn task orchestration
- **[Package-Level Standard](package-level.html)**: Learn deployment requirements
- **[Registry Standard](registry.html)**: Learn metadata schema

Or jump to the guides:
- **[Benchmark Author Guide](../guides/benchmark-authors.html)**: Wrap your benchmark
- **[Platform Developer Guide](../guides/platform-developers.html)**: Integrate CUBE support
