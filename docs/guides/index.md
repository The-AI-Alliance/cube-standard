---
layout: default
title: Guides
nav_order: 50
has_children: true
---

# CUBE Implementation Guides

Step-by-step guides for different CUBE user personas.

## Available Guides

### [Benchmark Author Guide](benchmark-authors.html)

**For**: Researchers and developers who have created a benchmark and want to make it CUBE-compliant

**You'll learn**:
- How to wrap an existing benchmark with the CUBE API
- Implementing Task-Level and Benchmark-Level interfaces
- Adding RPC support for remote execution
- Testing your implementation for compliance
- Publishing to the CUBE registry

**Time**: 1-2 hours to wrap a simple benchmark

### [Platform Developer Guide](platform-developers.html) *(Coming Soon)*

**For**: Developers building evaluation harnesses, training platforms, or agent frameworks

**You'll learn**:
- How to integrate CUBE support into your platform
- Connecting to and managing CUBE benchmarks
- Handling local vs remote execution
- Building multi-benchmark evaluation pipelines
- Error handling and resource management best practices

**Time**: 2-4 hours to add CUBE support

## Choosing the Right Guide

```
┌─────────────────────────────────────────────────────────┐
│              Who are you?                               │
└─────────────────────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌──────────────┐          ┌──────────────┐
│ I created a  │          │ I'm building │
│  benchmark   │          │  a platform  │
└──────────────┘          └──────────────┘
        │                         │
        ▼                         ▼
  Benchmark Author           Platform Developer
      Guide                      Guide
```

## Prerequisites

### For Benchmark Authors

Before wrapping your benchmark, you should have:

- ✅ A working benchmark implementation (even if it's benchmark-specific)
- ✅ Python 3.9+ installed
- ✅ Basic understanding of APIs and REST/RPC
- ✅ Familiarity with your benchmark's infrastructure (Docker, VMs, etc.)

**Optional but helpful**:
- Understanding of MCP (Model Context Protocol)
- Understanding of Gymnasium API
- Experience with containerization

### For Platform Developers

Before integrating CUBE, you should have:

- ✅ An existing evaluation/training platform or framework
- ✅ Python 3.9+ or ability to make HTTP requests
- ✅ Understanding of async/parallel execution
- ✅ Resource management experience (Docker, process management, etc.)

**Optional but helpful**:
- Experience with multi-benchmark evaluation
- Understanding of the CUBE API layers
- Familiarity with agent-environment loops

## Quick Start Paths

### Path 1: Wrap Your First Benchmark (30 minutes)

If you just want to get started quickly:

1. Read the [Overview](../overview.html) (10 min)
2. Follow the [Quick Start](../quickstart.html) to see CUBE in action (10 min)
3. Start the [Benchmark Author Guide](benchmark-authors.html) tutorial (10 min initial setup)

### Path 2: Understand Before Implementing (1 hour)

If you prefer to understand the full picture first:

1. Read the [Overview](../overview.html) (10 min)
2. Study the [API Reference](../api/) (30 min)
   - [Task-Level API](../api/task-level.html)
   - [Benchmark-Level API](../api/benchmark-level.html)
3. Review the [Quick Start](../quickstart.html) examples (10 min)
4. Begin the appropriate guide (10 min)

### Path 3: Deep Dive (2-3 hours)

For a comprehensive understanding:

1. Read the [Overview](../overview.html) (10 min)
2. Complete the [Quick Start](../quickstart.html) tutorial (20 min)
3. Read all [API Reference](../api/) pages (45 min)
   - [Task-Level API](../api/task-level.html)
   - [Benchmark-Level API](../api/benchmark-level.html)
   - [Package-Level Standard](../api/package-level.html)
   - [Registry Standard](../api/registry.html)
4. Review community examples (30 min)
5. Complete the relevant guide (1-2 hours)

## Community Examples

Learn from existing CUBE-compliant benchmarks:

- **cube-benchmark-miniwob** - Simple web tasks, good starter example
- **cube-benchmark-webarena-lite** - Complex shared infrastructure pattern
- **cube-benchmark-swebench-mini** - Containerized execution pattern
- **cube-benchmark-template** - Minimal template to fork and customize

{: .note }
> These examples are maintained by the community. Check the [GitHub repository](https://github.com/The-AI-Alliance/cube-standard) for the latest list.

## Getting Help

As you work through the guides:

- **API Questions**: Check the [API Reference](../api/)
- **Implementation Issues**: Search [GitHub Issues](https://github.com/The-AI-Alliance/cube-standard/issues)
- **Community Discussion**: Join [GitHub Discussions](https://github.com/The-AI-Alliance/cube-standard/discussions)
- **Direct Help**: Email [contact@thealliance.ai](mailto:contact@thealliance.ai?subject=CUBE%20Guide%20Question)

## Contributing to Guides

Found something unclear? Have a suggestion?

- Click the "Edit this page on GitHub" link at the bottom of any guide
- Submit a PR with improvements
- Open an issue describing what's confusing
- Share your implementation experience in Discussions

We're constantly improving these guides based on community feedback!

---

**Ready to start?**
- [Benchmark Author Guide →](benchmark-authors.html)
- [Platform Developer Guide →](platform-developers.html) *(Coming Soon)*
