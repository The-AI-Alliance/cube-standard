---
layout: default
title: Registry Standard
parent: API Reference
nav_order: 4
---

# Registry Standard

{: .warning }
> **Work in Progress**: This specification is under active development. The content below represents initial thinking and is subject to significant changes.

## Overview

The CUBE Registry is a centralized metadata catalog that enables discovery, filtering, and automated installation of CUBE-compliant benchmarks. It does not host benchmark code or data - only metadata that points to standard distribution platforms (PyPI, Docker Hub, etc.).

## Purpose

The Registry Standard addresses:

1. **Discovery**: How users find benchmarks relevant to their research
2. **Filtering**: How to select benchmarks by requirements, domain, cost, etc.
3. **Metadata**: What information must be provided for each benchmark
4. **Submission**: How benchmark authors register their benchmarks
5. **Versioning**: How benchmark updates are tracked
6. **Compliance**: How conformance to CUBE standard is verified

## Key Benefits

### For Researchers

- **Discoverability**: Find benchmarks you didn't know existed
- **Filtering**: Match benchmarks to your infrastructure constraints
- **Cost Estimation**: Understand token usage and time requirements before running
- **Trust**: Compliance badges indicate standard conformance

### For Benchmark Authors

- **Visibility**: Make your benchmark discoverable to the entire community
- **Distribution**: Leverage existing platforms (PyPI) for hosting
- **Metadata**: Provide rich information about capabilities and requirements

### For Platform Developers

- **Automation**: Programmatically discover and install compatible benchmarks
- **Validation**: Check requirements before attempting installation
- **Integration**: Build benchmark selection UIs

## Registry Metadata Schema

Based on the position paper, each benchmark must provide:

### Identification Fields

```json
{
  "id": "webarena-verified-v1",
  "name": "WebArena Verified",
  "version": "1.2.0",
  "authors": ["Jane Researcher", "John Developer"],
  "paper": "https://arxiv.org/abs/...",
  "homepage": "https://webarena.dev"
}
```

### Distribution Fields

```json
{
  "package": "cube-benchmark-webarena-verified",
  "parent": "webarena-v1",  // If this is a variant
  "package_license": "MIT",
  "benchmark_license": "CC-BY-NC-4.0",
  "content_notice": "Contains cloned websites"
}
```

### Runtime Requirements

```json
{
  "runtime": "docker",  // local | docker | apptainer | vm | docker-root | docker-in-docker | live
  "hardware": {
    "ram_gb": 16,
    "gpu": false,
    "disk_gb": 50
  }
}
```

### Benchmark Characteristics

```json
{
  "task_count": 812,
  "domains": ["web_navigation", "e-commerce"],
  "estimated_tokens": 20000000,  // Total for all tasks
  "avg_time_per_task_minutes": 15
}
```

### Compliance Badges

```json
{
  "compliance": [
    "no-docker-root",
    "task-isolated",
    "deterministic-tasks",
    "supports-parallelization"
  ]
}
```

## Planned Registry Operations

### Search and Filter API

```python
from cube import Registry

registry = Registry()

# Filter by infrastructure
lightweight = registry.list(
    runtime="docker",
    max_ram_gb=8,
    no_gpu=True
)

# Filter by domain
web_benchmarks = registry.list(
    domains=["web_navigation"],
    benchmark_license="commercial_use_allowed"
)

# Filter by cost
affordable = registry.list(
    max_total_tokens=1000000,
    max_time_per_task_minutes=10
)

# Filter by compliance
safe = registry.list(
    compliance=["no-docker-root", "task-isolated"]
)
```

### Benchmark Details

```python
# Get detailed info about a specific benchmark
info = registry.get("webarena-verified-v1")

print(f"Tasks: {info.task_count}")
print(f"RAM needed: {info.hardware.ram_gb}GB")
print(f"License: {info.benchmark_license}")
print(f"Compliance: {', '.join(info.compliance)}")
```

### Installation Integration

```python
# Search and install workflow
benchmarks = registry.list(
    runtime="docker",
    max_ram_gb=16
)

for benchmark in benchmarks:
    print(f"Installing {benchmark.name}...")
    os.system(f"pip install {benchmark.package}")
```

## Registry Architecture

{: .note }
> This section will be expanded with details on:
>
> - Registry hosting and infrastructure
> - Submission and review process
> - Automated compliance testing
> - Version management
> - Search/filter implementation

## Compliance Badges

Proposed compliance badges:

| Badge | Meaning |
|-------|---------|
| `no-docker-root` | Does not require Docker root privileges |
| `task-isolated` | Tasks don't interfere with each other |
| `deterministic-tasks` | Same seed produces same task |
| `supports-parallelization` | Can run multiple tasks concurrently |
| `offline-capable` | Works without internet access |
| `reproducible` | Fully reproducible results |
| `no-external-apis` | Doesn't call external services |
| `gpu-optional` | GPU improves but not required |

## Metadata Validation

Benchmarks should validate their metadata:

```python
from cube.registry import validate_metadata

metadata = {
    "id": "my-benchmark-v1",
    "name": "My Benchmark",
    # ... rest of metadata
}

# Raises ValidationError if invalid
validate_metadata(metadata)
```

## Submission Process (Proposed)

1. **Implement CUBE Wrapper**: Follow [Benchmark Author Guide]({{site.baseurl}}/guides/benchmark-authors)
2. **Create Metadata**: Fill out registry metadata JSON
3. **Validate Locally**: Run compliance tests
4. **Submit PR**: Add metadata to registry repository
5. **Automated Tests**: CI validates compliance
6. **Community Review**: Maintainers review submission
7. **Merge & Publish**: Benchmark becomes discoverable

## Economic Filtering

Help users estimate costs:

```python
# Find benchmarks under budget
registry.list(
    max_total_tokens=5000000,  # 5M tokens total
    max_cost_usd=100,  # Estimated at $0.02/1k tokens
    task_count_min=50  # At least 50 tasks
)
```

## Open Questions

1. **Centralized vs Federated**: Single registry or multiple federated registries?
2. **Curation**: Who approves benchmark submissions?
3. **Quality Control**: How to verify metadata accuracy?
4. **Updates**: How to handle benchmark updates and deprecations?
5. **Mirrors**: Should there be regional mirrors?
6. **API Rate Limits**: How to prevent abuse?

## Example Registry Entry

Complete example for a benchmark:

```json
{
  "id": "miniwob-tasks-v1",
  "name": "MiniWoB Tasks",
  "version": "1.0.0",
  "authors": ["AI Lab"],
  "paper": null,
  "homepage": "https://miniwob.org",
  "package": "cube-benchmark-miniwob",
  "parent": null,
  "package_license": "MIT",
  "benchmark_license": "MIT",
  "content_notice": null,
  "compliance": [
    "no-docker-root",
    "task-isolated",
    "deterministic-tasks",
    "supports-parallelization",
    "offline-capable"
  ],
  "runtime": "docker",
  "hardware": {
    "ram_gb": 4,
    "gpu": false,
    "disk_gb": 5
  },
  "task_count": 100,
  "domains": ["web_navigation", "form_filling"],
  "estimated_tokens": 500000,
  "avg_time_per_task_minutes": 3,
  "description": "Simple web interaction tasks for agent evaluation",
  "created_at": "2026-01-15T10:00:00Z",
  "updated_at": "2026-01-20T15:30:00Z"
}
```

## Implementation Considerations

### Registry Backend

Options under consideration:

- **Static JSON/YAML**: Simple, GitHub-hosted, versioned via git
- **Database**: PostgreSQL/MongoDB for advanced querying
- **GraphQL API**: Flexible querying capabilities
- **REST API**: Simple HTTP endpoints

### Search Optimization

For large registry (100+ benchmarks):

- Indexed filtering by common fields
- Caching for frequent queries
- Pre-computed aggregations
- Full-text search on descriptions

## Community Feedback Needed

We need input on:

- Registry governance model
- Submission review process
- Metadata schema completeness
- Compliance badge definitions
- API design preferences
- Quality control mechanisms

## Contributing

To contribute to the Registry Standard:

1. Join [Registry Discussions](https://github.com/The-AI-Alliance/cube-standard/discussions)
2. Review and comment on metadata schema proposals
3. Suggest additional compliance badges
4. Help design the submission workflow
5. Contribute to registry implementation

## Related Documentation

- [Task-Level API]({{site.baseurl}}/api/task-level) - How agents interact with tasks
- [Benchmark-Level API]({{site.baseurl}}/api/benchmark-level) - How tasks are orchestrated
- [Package-Level Standard]({{site.baseurl}}/api/package-level) - Installation and deployment
- [Benchmark Author Guide]({{site.baseurl}}/guides/benchmark-authors) - How to submit benchmarks

---

{: .note }
> **Status**: This page is a placeholder for the Registry Standard specification. The schema is based on Table 3 from the position paper but needs community refinement. Full implementation details coming soon.

## Quick Reference: Registry Metadata Fields

From the position paper (Table 3):

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier |
| `name` | string | Human-readable name |
| `version` | string | Semantic version |
| `authors` | string[] | Package authors |
| `paper` | string? | Related paper URL |
| `package` | string | PyPI package name |
| `parent` | string? | Parent benchmark id |
| `package_license` | string | Wrapper code license |
| `benchmark_license` | string | Benchmark data/tasks license |
| `content_notice` | string? | Copyright warning |
| `compliance` | string[] | Compliance badges |
| `runtime` | enum | Deployment model |
| `hardware` | object | Resource requirements |
| `task_count` | int | Number of tasks |
| `estimated_tokens` | int | Total estimated tokens |
