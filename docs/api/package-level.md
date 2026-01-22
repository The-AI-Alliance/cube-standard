---
layout: default
title: Package-Level Standard
parent: API Reference
nav_order: 3
---

# Package-Level Standard

{: .warning }
> **Work in Progress**: This specification is under active development. The content below represents initial thinking and is subject to significant changes.

## Overview

The Package-Level Standard defines how CUBE-compliant benchmarks are packaged, distributed, installed, and deployed. This layer ensures that benchmarks can be easily integrated into diverse computing environments while declaring their resource requirements and deployment constraints.

## Purpose

The Package-Level Standard addresses:

1. **Installation**: How to install a benchmark (pip, conda, Docker, etc.)
2. **Dependencies**: How to declare and manage dependencies
3. **Resource Requirements**: How to specify RAM, GPU, disk, and other hardware needs
4. **Deployment Models**: How to support different deployment scenarios (local, container, VM, cloud)
5. **Parallelization**: How to enable concurrent task execution
6. **Configuration**: How to handle environment-specific settings

## Key Concepts

### Deployment Models

Benchmarks may support one or more deployment models:

- **Local Python**: Pure Python package, runs directly on the host
- **Docker**: Containerized, requires Docker daemon
- **Apptainer/Singularity**: HPC-compatible containerization
- **VM**: Requires virtual machine (VirtualBox, VMware, etc.)
- **Remote**: Runs on remote infrastructure, accessed via API
- **Hybrid**: Combination of the above

### Resource Declarations

Benchmarks must declare:

- Minimum RAM requirements (GB)
- GPU requirements (type, memory, CUDA version)
- Disk space requirements (GB)
- Network requirements (ports, bandwidth)
- Special hardware (webcam, microphone, etc.)

## Planned Specification

{: .note }
> This section will be expanded with complete specifications for:
>
> - Standard package structure (`setup.py`, `pyproject.toml`)
> - Dependency declaration formats
> - Resource requirement schema
> - Environment configuration patterns
> - Multi-deployment support patterns
> - Compliance levels and badges

## Initial Design Considerations

### Installation Requirements

```toml
# pyproject.toml example
[project]
name = "cube-benchmark-example"
version = "1.0.0"
requires-python = ">=3.9"
dependencies = [
    "cube-standard>=0.1.0",
    # benchmark-specific deps
]

[project.optional-dependencies]
docker = ["docker>=6.0.0"]
gpu = ["torch>=2.0.0"]

[tool.cube]
# CUBE-specific metadata
deployment_models = ["local", "docker"]
min_ram_gb = 8
gpu_required = false
```

### Deployment Variants

Benchmarks with multiple deployment options should provide clear installation paths:

```bash
# Lightweight local installation
pip install cube-benchmark-example

# With Docker support
pip install cube-benchmark-example[docker]

# With GPU support
pip install cube-benchmark-example[gpu]

# Full installation
pip install cube-benchmark-example[all]
```

### Resource Validation

Benchmarks should validate resources at runtime:

```python
from cube.utils import check_resources

class MyBenchmark:
    def __init__(self):
        # Validate before starting
        check_resources(
            min_ram_gb=8,
            min_disk_gb=20,
            required_ports=[8080, 8081],
            gpu_required=False
        )
        # Continue initialization...
```

### Parallelization Support

Benchmarks should declare how many parallel tasks they support:

```python
class MyBenchmark:
    def info(self):
        return {
            "capabilities": {
                "parallel_tasks": 10,  # Max concurrent tasks
                "parallel_mode": "shared_infra",  # or "isolated"
            }
        }
```

## Open Questions

Issues we're still exploring:

1. **Container Registry**: Where should benchmark containers be hosted?
2. **Versioning**: How to handle benchmark data versioning separately from code?
3. **Licensing**: How to represent dual licensing (code vs. data)?
4. **Offline Support**: Should benchmarks work without internet access?
5. **Resource Limits**: How to enforce resource limits during execution?
6. **Platform-Specific Builds**: How to handle OS-specific requirements?

## Examples

### Lightweight Pure Python Benchmark

```python
# Minimal package, no special infrastructure
# Just pip install and go
```

### Docker-Based Benchmark

```python
# Requires Docker
# Pulls images from registry
# Manages containers automatically
```

### HPC-Compatible Benchmark

```python
# Works on HPC clusters
# Uses Apptainer instead of Docker
# No root privileges required
```

## Community Feedback Needed

We need community input on:

- What deployment models are most important?
- How to handle complex multi-container setups?
- Resource requirement granularity
- Installation experience expectations
- Compliance testing automation

## Contributing

This specification is being developed collaboratively. To contribute:

1. Review the [GitHub Discussions](https://github.com/The-AI-Alliance/cube-standard/discussions) on package-level topics
2. Share your benchmark's deployment challenges
3. Propose concrete specification elements
4. Implement proof-of-concept examples

## Related Documentation

- [Task-Level API](task-level.html) - How agents interact with tasks
- [Benchmark-Level API](benchmark-level.html) - How tasks are orchestrated
- [Registry Standard](registry.html) - How benchmarks are discovered
- [Benchmark Author Guide](../guides/benchmark-authors.html) - Implementation guide

---

{: .note }
> **Status**: This page is a placeholder for the Package-Level Standard specification. Full details coming soon based on community discussion and real-world implementation experience.
