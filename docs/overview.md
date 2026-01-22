---
layout: default
title: Overview
nav_order: 20
has_children: false
---

# CUBE Standard Overview

{: .note }
> CUBE (Common Unified Benchmark Environments) is a protocol standard that eliminates the integration tax of agentic benchmarks by providing a universal interface between benchmarks and evaluation frameworks.

## The Problem: Benchmark Fragmentation

The field of AI agent evaluation is experiencing explosive growth. We now have over 100 diverse benchmarks testing everything from web navigation to software engineering to desktop computer use. However, this abundance has created a critical problem: **the integration tax**.

Every time a researcher wants to evaluate an agent on a new benchmark, they must:

- Learn the benchmark's unique API and setup requirements
- Write custom integration code (wrappers, drivers, adapters)
- Handle deployment complexity (Docker, VMs, networking, state management)
- Debug environment-specific issues
- Repeat this process for every evaluation platform they use

This N-to-M mapping problem means:
- **Researchers waste time** on systems engineering instead of AI research
- **Smaller labs are excluded** from comprehensive evaluation due to integration costs
- **Great benchmarks remain unused** because they're too difficult to set up
- **Platform fragmentation deepens** as each framework requires custom wrappers

## The Solution: A Universal Standard

CUBE solves this by defining a standard protocol: **wrap a benchmark once, use it everywhere**.

When a benchmark is CUBE-compliant:
- Any CUBE-compatible evaluation harness can use it immediately
- No custom integration code needed
- Deployment is standardized and automated
- Benchmarks become discoverable through a central registry

When an evaluation platform supports CUBE:
- It instantly gains access to all CUBE-compliant benchmarks
- No per-benchmark wrapper development required
- Focus shifts to platform features instead of benchmark moats

## Core Design Principles

### 1. Build on Established Standards

CUBE doesn't reinvent the wheel. It combines two proven protocols:

- **Model Context Protocol (MCP)** for tool discovery and invocation
- **Gym API** for environment stepping and evaluation semantics

Researchers already familiar with these standards can immediately understand CUBE.

### 2. Separation of Concerns

CUBE separates benchmark integration into four distinct layers:

| Layer | Purpose | Key Question |
|-------|---------|--------------|
| **Task Level** | Agent-environment interaction | How does an agent interact with a single task instance? |
| **Benchmark Level** | Task discovery and spawning | How do I find and start tasks? |
| **Package Level** | Installation and deployment | How do I install and run this benchmark? |
| **Registry Level** | Discovery and metadata | What benchmarks exist and what are their requirements? |

This layered architecture means benchmark authors only implement what they need, and platform developers get clean interfaces at each level.

### 3. Python-First with RPC Fallback

CUBE prioritizes performance and developer experience:

- **Local execution**: Benchmarks implement a Python class for zero-overhead interaction
- **Remote execution**: The same interface is exposed via RPC for distributed deployments
- **1:1 API mapping**: Switching between local and remote requires no code changes

Use local execution for fast iteration and debugging. Use RPC for:
- Benchmarks that can't be containerized (e.g., SaaS platforms)
- Cross-platform execution (Windows benchmarks from Linux)
- Security sandboxing of untrusted code

### 4. Flexible Tooling

Different agents use different tools. A vision-based browser agent behaves differently than one using HTML parsing, even on identical tasks.

CUBE handles this by:
- **Requiring default tools**: Every benchmark ships with working tools out-of-the-box
- **Allowing tool reconfiguration**: Researchers can substitute alternative tools where the benchmark design permits
- **Making tool choice explicit**: The standard clearly separates tasks, environments, and tooling

## Who Is CUBE For?

### Researchers & End Users

**You want to**: Evaluate agents across many diverse benchmarks without integration headaches

**CUBE gives you**:
- One-command installation of any compliant benchmark
- Consistent API across all benchmarks
- Ability to filter benchmarks by resource requirements, cost, domain, etc.
- Focus on agent development instead of environment setup

### Benchmark Authors

**You want to**: Make your benchmark widely accessible without writing wrappers for every platform

**CUBE gives you**:
- Wrap your benchmark once, works everywhere
- Clear specification with examples
- Automatic discoverability through the registry
- Control over your distribution and licensing

### Platform Developers

**You want to**: Build the best evaluation/training platform without being bottlenecked by benchmark integration

**CUBE gives you**:
- Instant access to all compliant benchmarks
- Compete on features, not benchmark quantity
- Standard interface to build against
- Focus on innovation instead of wrapper maintenance

## Architecture at a Glance

```
┌─────────────────────────────────────────────────────────┐
│                    CUBE Registry                        │
│         (Metadata catalog for discovery)                │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                  Package Level                          │
│           (Installation & Deployment)                   │
│         pip install cube-benchmark-name                 │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│               Benchmark Level API                       │
│  cube/info, cube/tasks, cube/spawn, cube/status, ...    │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                Task Level API                           │
│         MCP (tools/*, resources/*)                      │
│    + CUBE (cube/evaluation, cube/reset, cube/close)     │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Agent performs  │
                  │   evaluation     │
                  └─────────────────┘
```

## What Makes a Benchmark CUBE-Compliant?

A CUBE-compliant benchmark must:

1. **Implement the Task-Level API**: Expose MCP methods for actions and observations, plus CUBE methods for evaluation
2. **Implement the Benchmark-Level API**: Provide task discovery, spawning, and lifecycle management
3. **Follow the Package Standard**: Support standard installation and declare resource requirements
4. **Register in the CUBE Registry**: Provide accurate metadata for discovery and filtering
5. **Ship with default tools**: Include working tools so the benchmark is immediately usable

## Getting Started

Ready to use or create CUBE-compliant benchmarks?

- **[Quick Start Guide](quickstart.html)**: Get up and running in 5 minutes
- **[API Reference](api/)**: Complete specification of all APIs
- **[Benchmark Author Guide](guides/benchmark-authors.html)**: Wrap your first benchmark
- **[Platform Developer Guide](guides/platform-developers.html)**: Integrate CUBE support into your harness

## Community & Governance

CUBE is an open standard developed under [The AI Alliance](https://aialliance.org).

- **Open development**: All discussions and decisions happen in public
- **Community-driven**: Changes require community consensus
- **Vendor-neutral**: No single company controls the standard
- **Apache 2.0 licensed**: Free to implement and extend

See our [Contributing Guide](contributing.html) to get involved.

## FAQ

**Q: Do I have to rewrite my existing benchmark?**
A: No. CUBE is a wrapper specification. You implement a thin adapter layer around your existing code.

**Q: What if my benchmark needs special infrastructure?**
A: The standard supports diverse deployment models: Docker, Apptainer, VMs, or even live SaaS platforms via RPC.

**Q: Can I still innovate on tool design?**
A: Yes. While benchmarks ship with default tools, the standard allows tool reconfiguration for research purposes.

**Q: What about benchmarks with stochastic elements?**
A: The Task-Level API includes seed parameters for reproducibility.

**Q: How do I handle large-scale parallel evaluation?**
A: The Package-Level standard specifies parallelization support and resource isolation.

**Q: Is this just for RL environments?**
A: No. While CUBE builds on Gym concepts, it's designed for diverse agentic benchmarks: web navigation, coding, desktop use, knowledge work, etc.

---

**Next Steps**:
- Try the [Quick Start](quickstart.html) to see CUBE in action
- Explore the [API Reference](api/) to understand the specification
- Join the discussion on [GitHub](https://github.com/The-AI-Alliance/cube-standard)
