---
layout: default
title: Home
nav_order: 10
has_children: false
---

# CUBE Standard

**Common Unified Benchmark Environments**

{: .warning }
> **Early Documentation**: This specification is in active development. APIs and documentation may change as we gather community feedback and refine the standard. Code examples are placeholders and may not work until the specification is finalized.

A protocol standard that eliminates the integration tax of agentic benchmarks by providing a universal interface between benchmarks and evaluation frameworks.

{: .note }
> **Wrap a benchmark once, use it everywhere.** CUBE makes any compliant benchmark immediately accessible to any compliant evaluation platform.

## The Problem

The AI agent evaluation field has over 100 diverse benchmarks, but integrating each one requires substantial custom work:
- Custom wrappers for every platform
- Complex infrastructure setup
- Platform-specific maintenance
- Fragmented ecosystem

**This integration tax limits research velocity and excludes smaller labs from comprehensive evaluation.**

## The Solution

CUBE provides a universal standard so that:
- 🎯 **Benchmark authors** wrap once, reach all platforms
- 🚀 **Platform developers** support CUBE, access all benchmarks instantly
- 🔬 **Researchers** evaluate across dozens of benchmarks without integration work
- 🌐 **The community** benefits from an interoperable ecosystem

## Quick Start

### Install and Run a Benchmark

```python
# Install any CUBE-compliant benchmark
pip install cube-benchmark-miniwob

# Use it immediately
from cube import LocalRunner

benchmark = LocalRunner("cube-benchmark-miniwob")
task = benchmark.spawn(task_id="click-button", seed=42)

# Standard interface across all benchmarks
state = task.reset()
tools = task.list_tools()
result = task.call_tool("click", {"x": 100, "y": 100})
eval_state = task.evaluate()

task.close()
```

### Evaluate Across Multiple Benchmarks

```python
from cube import Registry, LocalRunner

# Discover benchmarks matching your constraints
registry = Registry()
benchmarks = registry.list(
    runtime="docker",
    max_ram_gb=8,
    domain="web_navigation"
)

# Evaluate your agent on all of them
for bench_info in benchmarks:
    benchmark = LocalRunner(bench_info.package)
    # Same API for all benchmarks!
    tasks = benchmark.list_tasks(limit=10)
    # ... run evaluation ...
```

## Documentation

### Getting Started

- **[Overview]({{site.baseurl}}/overview)** - Understanding CUBE's architecture and design
- **[Quick Start]({{site.baseurl}}/quickstart)** - Get running in 5 minutes

### API Reference

- **[API Overview]({{site.baseurl}}/api/)** - Complete specification
  - [Task-Level API]({{site.baseurl}}/api/task-level) - Agent-environment interaction
  - [Benchmark-Level API]({{site.baseurl}}/api/benchmark-level) - Task orchestration
  - [Package-Level Standard]({{site.baseurl}}/api/package-level) - Installation & deployment
  - [Registry Standard]({{site.baseurl}}/api/registry) - Discovery & metadata

### Implementation Guides

- **[Benchmark Author Guide]({{site.baseurl}}/guides/benchmark-authors)** - Wrap your benchmark for CUBE
- **[Platform Developer Guide]({{site.baseurl}}/guides/platform-developers)** - Integrate CUBE support *(Coming Soon)*

## Core Principles

### 1. Build on Established Standards

CUBE combines proven protocols:

- **MCP (Model Context Protocol)** for tool discovery and invocation
- **Gym API** for environment stepping and evaluation
- **Standard Python packaging** for distribution

### 2. Separation of Concerns

Four distinct API layers:

- **Task Level**: How agents interact with individual tasks
- **Benchmark Level**: How tasks are discovered and spawned
- **Package Level**: How benchmarks are installed and deployed
- **Registry Level**: How benchmarks are discovered and filtered

### 3. Python-First with RPC Fallback

- Local execution for speed and debugging
- Remote execution for distributed deployments
- 1:1 API mapping between local and remote

### 4. Community-Driven

- Open standard under [The AI Alliance](https://aialliance.org)
- Vendor-neutral governance
- Apache 2.0 licensed

## Who Is CUBE For?

| You are... | You want to... | CUBE gives you... |
| ---------- | -------------- | ----------------- |
| **Researcher** | Evaluate agents on diverse benchmarks | One-line installation, consistent API |
| **Benchmark Author** | Make your benchmark widely accessible | Wrap once, works everywhere |
| **Platform Developer** | Support all benchmarks without custom wrappers | Standard interface, instant compatibility |

## Get Involved

We're building this standard as a community. Join us!

- **[Contributing Guide]({{site.baseurl}}/contributing)** - How to contribute
- **[GitHub Repo](https://github.com/The-AI-Alliance/cube-standard){:target="repo"}** - Code and issues
- **[Discussions](https://github.com/The-AI-Alliance/cube-standard/discussions){:target="discussions"}** - Community forum
- **[About The AI Alliance]({{site.baseurl}}/about)** - More about this project

{: .tip }
> **Tips for navigating this site:**
>
> 1. Use the search box at the top to find specific content
> 2. Check the sidebar for all documentation sections
> 3. Every page has an "Edit this page on GitHub" link for quick contributions

## Additional Resources

- [The AI Alliance](https://aialliance.org){:target="ai-alliance"} - Parent organization
- [Position Paper](https://github.com/The-AI-Alliance/cube-standard/tree/main/papers){:target="paper"} - Academic motivation and design rationale
- [Example Benchmarks](https://github.com/cube-benchmarks){:target="examples"} - Reference implementations

---

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>
