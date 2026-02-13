---
layout: default
title: Platform Developer Guide
parent: Guides
nav_order: 2
---

# Platform Developer Guide

{: .warning }
> **Coming Soon**: This guide is under development. The outline below shows planned content.

## Overview

This guide will help developers integrate CUBE support into evaluation harnesses, training platforms, and agent frameworks. By supporting CUBE, your platform gains instant access to all CUBE-compliant benchmarks without custom integration work.

## What You'll Learn

By the end of this guide, you'll be able to:

- ✅ Integrate CUBE support into your platform
- ✅ Connect to and manage CUBE benchmarks
- ✅ Handle both local and remote benchmark execution
- ✅ Build multi-benchmark evaluation pipelines
- ✅ Implement proper error handling and resource management
- ✅ Provide a great developer experience for benchmark users

## Who Is This For?

This guide is for developers building:

- **Evaluation harnesses** (e.g., tools for running benchmarks at scale)
- **Training platforms** (e.g., systems that train agents on diverse tasks)
- **Agent frameworks** (e.g., libraries that provide agent scaffolding)
- **Research tools** (e.g., experiment management systems)

## Prerequisites

Before starting, you should have:

- A working platform or framework (or be building one)
- Python 3.9+ or ability to make HTTP requests
- Understanding of async/parallel execution
- Familiarity with the [CUBE API layers](../api/)

## Planned Content

### 1. Integration Patterns

**Quick Integration (1 hour)**
- Use the CUBE client library
- Connect to benchmarks
- Run basic evaluations

**Production Integration (1 day)**
- Resource management
- Error handling
- Logging and monitoring
- Configuration management

**Advanced Integration (2-3 days)**
- Multi-benchmark orchestration
- Parallel execution
- Custom schedulers
- Result aggregation

### 2. Working with Benchmarks

**Discovery and Installation**
```python
# Will cover:
- Using the Registry API to find benchmarks
- Filtering by requirements
- Automated installation
- Version management
```

**Connection Management**
```python
# Will cover:
- Local vs remote execution
- Connection pooling
- Timeout handling
- Health checks
```

**Task Orchestration**
```python
# Will cover:
- Spawning multiple tasks
- Resource allocation
- Task lifecycle management
- Cleanup strategies
```

### 3. Evaluation Loops

**Basic Agent-Task Loop**
```python
# Will demonstrate:
- Standard evaluation pattern
- Observation handling
- Action execution
- Reward collection
```

**Batched Evaluation**
```python
# Will demonstrate:
- Parallel task execution
- Result aggregation
- Progress tracking
- Failure recovery
```

**Multi-Benchmark Evaluation**
```python
# Will demonstrate:
- Sequential benchmark execution
- Resource-aware scheduling
- Cross-benchmark metrics
- Comparative analysis
```

### 4. Resource Management

**Memory Management**
- Estimating benchmark memory usage
- Preventing OOM errors
- Task-level resource limits
- Cleanup best practices

**Parallelization**
- Determining max parallel tasks
- Dynamic resource allocation
- Load balancing
- Avoiding resource contention

**Storage Management**
- Disk space requirements
- Cache management
- Temporary file cleanup
- Log rotation

### 5. Error Handling

**Common Errors**
- Task spawn failures
- Tool execution errors
- Timeout handling
- Network failures

**Retry Strategies**
- Transient vs permanent failures
- Exponential backoff
- Circuit breakers
- Graceful degradation

**User-Friendly Errors**
- Error message formatting
- Actionable suggestions
- Debugging information
- Support resources

### 6. Developer Experience

**Progress Indicators**
```python
# Will show:
- Progress bars for evaluations
- Real-time metrics
- ETA calculations
- Cancellation support
```

**Logging and Debugging**
```python
# Will show:
- Structured logging
- Debug modes
- Trace capture
- Performance profiling
```

**Configuration**
```python
# Will show:
- Config file formats
- Environment variables
- Runtime overrides
- Validation
```

### 7. Advanced Topics

**Custom Schedulers**
- Priority-based scheduling
- Cost-aware scheduling
- Deadline scheduling
- Fair-share scheduling

**Result Management**
- Storing evaluation results
- Replay capabilities
- Checkpoint/resume
- Export formats

**Integration Examples**
- AgentOps integration
- Weights & Biases integration
- MLflow integration
- Custom dashboards

## Example Platform Integration

Here's a preview of what a simple platform integration might look like:

```python
from cube import Registry, LocalRunner

class MyEvaluationPlatform:
    """Example platform with CUBE support."""

    def __init__(self):
        self.registry = Registry()
        self.benchmarks = {}

    def discover_benchmarks(self, **filters):
        """Find benchmarks matching criteria."""
        return self.registry.list(**filters)

    def add_benchmark(self, package_name):
        """Add a benchmark to the platform."""
        benchmark = LocalRunner(package_name)
        self.benchmarks[package_name] = benchmark
        return benchmark.info()

    def evaluate_agent(self, agent, benchmark_name, task_count=10):
        """Run agent on benchmark tasks."""
        benchmark = self.benchmarks[benchmark_name]
        tasks = benchmark.list_tasks(limit=task_count)

        results = []
        for task_info in tasks:
            task = benchmark.spawn(task_id=task_info.id)
            try:
                result = self._run_task(agent, task)
                results.append(result)
            finally:
                task.close()

        return self._aggregate_results(results)

    def _run_task(self, agent, task):
        """Execute a single task evaluation."""
        state = task.reset()
        total_reward = 0

        while not (state.terminated or state.truncated):
            action = agent.decide(state.observation)
            task.call_tool(action.name, action.args)
            state = task.evaluate()
            total_reward += state.reward

        return {
            'reward': total_reward,
            'success': state.info.get('success', False),
            'steps': state.info.get('step_count', 0)
        }

    def _aggregate_results(self, results):
        """Compute aggregate metrics."""
        return {
            'avg_reward': sum(r['reward'] for r in results) / len(results),
            'success_rate': sum(r['success'] for r in results) / len(results),
            'avg_steps': sum(r['steps'] for r in results) / len(results)
        }

# Usage
platform = MyEvaluationPlatform()

# Discover and add benchmarks
benchmarks = platform.discover_benchmarks(runtime="docker", max_ram_gb=8)
for bench in benchmarks[:3]:
    platform.add_benchmark(bench.package)

# Evaluate agent
my_agent = MyAgent()  # Your agent implementation
results = platform.evaluate_agent(my_agent, "cube-benchmark-miniwob")
print(f"Success rate: {results['success_rate']:.2%}")
```

## Status and Timeline

| Section | Status | ETA |
|---------|--------|-----|
| Integration Patterns | Planned | TBD |
| Working with Benchmarks | Planned | TBD |
| Evaluation Loops | Planned | TBD |
| Resource Management | Planned | TBD |
| Error Handling | Planned | TBD |
| Developer Experience | Planned | TBD |
| Advanced Topics | Planned | TBD |

## Early Access

Want to help shape this guide?

- Share your platform's integration challenges in [Discussions](https://github.com/The-AI-Alliance/cube-standard/discussions)
- Review draft content and provide feedback
- Contribute example integrations
- Test early versions of the guide

## Related Documentation

- [Overview]({{site.baseurl}}/overview) - Understanding CUBE's architecture
- [API Reference]({{site.baseurl}}/api/) - Complete API specifications
- [Benchmark Author Guide]({{site.baseurl}}/guides/benchmark-authors) - For benchmark implementers
- [Quick Start]({{site.baseurl}}/quickstart) - Basic usage examples

---

{: .note }
> **Want this guide sooner?** We're prioritizing documentation based on community needs. If your platform is ready to integrate CUBE, reach out via [GitHub Discussions](https://github.com/The-AI-Alliance/cube-standard/discussions) and we'll work with you on early documentation.

## Temporary Resources

Until this full guide is available, refer to:

1. **[Quick Start Guide]({{site.baseurl}}/quickstart)** - Basic usage patterns
2. **[API Reference]({{site.baseurl}}/api/)** - Complete API specifications
3. **[Benchmark Author Guide]({{site.baseurl}}/guides/benchmark-authors)** - Shows the other side of integration
4. **[GitHub Discussions](https://github.com/The-AI-Alliance/cube-standard/discussions)** - Ask specific questions

## Contributing

If you're building a CUBE integration:

1. Document your integration approach
2. Share code snippets and patterns
3. Report integration challenges
4. Suggest guide improvements

Your real-world experience will help shape this guide!
