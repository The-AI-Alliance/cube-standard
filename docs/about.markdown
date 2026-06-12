---
layout: default
title: About
nav_order: 100
has_children: false
---

**Common Unified Benchmark Environments** — a protocol standard that eliminates the integration tax of agentic benchmarks by providing a universal interface between benchmarks and evaluation frameworks.

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

## Is CUBE For You?

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

**Start here:** [Authoring a CUBE]({{site.baseurl}}/authoring-a-cube) — walkthrough with three starting paths, implementation order, validation, and submission.

### Platform Developers

**You want to**: Build the best evaluation/training platform without being bottlenecked by benchmark integration

**CUBE gives you**:

- Instant access to all compliant benchmarks
- Compete on features, not benchmark quantity
- Standard interface to build against
- Focus on innovation instead of wrapper maintenance

## About The AI Alliance

**CUBE Standard** is a project of the [The AI Alliance](https://thealliance.ai){:target="ai-alliance"}.

The AI Alliance is a global collaboration of startups, enterprises, academic, and other research institutions interested in advancing the state of the art, the availability, and the safety of AI technology and uses. The AI Alliance's core projects seek to address substantial cross-community challenges and are an opportunity for contributors to collaborate, build, and make an impact on the future of AI. Core Projects are managed directly by the AI Alliance and governed as described in our [community GitHub repository](https://github.com/The-AI-Alliance/community){:target="community"}.

You can find information about all AI Alliance projects on [our website](https://thealliance.ai/){:target="aia"} and our [GitHub organization](https://the-ai-alliance.github.io/){:target="aia-github"}.

If you have any questions or concerns about this effort, please contact us at [contact@thealliance.ai](mailto:contact@thealliance.ai?subject=Questions about CUBE Standard).

### Community & Governance

CUBE is an open standard with open governance:

- **Open development**: All discussions and decisions happen in public
- **Community-driven**: Changes require community consensus
- **Vendor-neutral**: No single company controls the standard
- **Apache 2.0 licensed**: Free to implement and extend

See our [Contributing Guide]({{site.baseurl}}/contributing) to get involved.

### Other AI Alliance Information

- [More About the AI Alliance](https://thealliance.ai/about){:target="aia"}
- [Contact Us](mailto:contact@thealliance.ai?subject=Questions about the AI Alliance) (email)
- Follow us on [LinkedIn](https://www.linkedin.com/company/the-aialliance/){:target="linkedin"} and [Bluesky](https://bsky.app/profile/aialliance.bsky.social){:target="bluesky"}
