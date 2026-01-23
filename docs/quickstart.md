---
layout: default
title: Quick Start
nav_order: 30
has_children: false
---

# Quick Start Guide

Get up and running with CUBE-compliant benchmarks in 5 minutes.

{: .note }
> This guide assumes you have Python 3.9+ installed. We'll install a sample CUBE-compliant benchmark and run a simple evaluation.

## Installation

Install the CUBE standard library and a sample benchmark:

```bash
# Install CUBE core library
pip install cube-standard

# Install a CUBE-compliant benchmark (example)
pip install cube-benchmark-miniwob
```

{: .tip }
> You can discover all available CUBE benchmarks using the registry (see [Registry Guide](api/registry.html)).

## Your First Evaluation

### Option 1: Local Python Execution (Recommended for getting started)

For most benchmarks, the fastest way to get started is using the Python API directly:

```python
from cube import Registry, LocalRunner

# Discover available benchmarks
registry = Registry()
benchmarks = registry.list(runtime="docker", max_ram_gb=8)
print(f"Found {len(benchmarks)} compatible benchmarks")

# Connect to a specific benchmark
benchmark = LocalRunner("cube-benchmark-miniwob")

# List available tasks
tasks = benchmark.list_tasks(limit=5)
for task in tasks:
    print(f"- {task.id}: {task.description}")

# Spawn a task instance
task = benchmark.spawn(task_id="click-button", seed=42)

# Get the task description and initial observation
initial_state = task.reset()
print(f"Task: {initial_state.info['task_description']}")
print(f"Observation: {initial_state.observation}")

# Discover available tools (actions)
tools = task.list_tools()
for tool in tools:
    print(f"Tool: {tool.name} - {tool.description}")

# Execute an action
result = task.call_tool("click", {"x": 100, "y": 200})
print(f"Action result: {result}")

# Check evaluation (reward, done, etc.)
eval_state = task.evaluate()
print(f"Reward: {eval_state.reward}")
print(f"Done: {eval_state.terminated}")
print(f"Score: {eval_state.info.get('score', 'N/A')}")

# Cleanup
task.close()
```

### Option 2: Remote Execution via RPC

For benchmarks that can't run locally or when you need distributed execution:

```python
from cube import RemoteRunner

# Connect to a remote CUBE benchmark server
benchmark = RemoteRunner("http://localhost:8000")

# Get benchmark information
info = benchmark.info()
print(f"Benchmark: {info.name} v{info.version}")
print(f"Tasks: {info.task_count}")

# Spawn a task (returns URL endpoint)
session = benchmark.spawn(task_id="example-task-1", seed=42)
print(f"Task endpoint: {session.url}")

# Connect to the task instance
task = RemoteRunner(session.url)

# Rest of the interaction is identical to local execution
tools = task.list_tools()
result = task.call_tool("navigate", {"url": "https://example.com"})
eval_state = task.evaluate()

# Cleanup
task.close()
benchmark.shutdown(session_id=session.id)
```

## Complete Agent Evaluation Loop

Here's a complete example of running an agent on a CUBE benchmark:

```python
from cube import LocalRunner

def simple_agent(task, max_steps=10):
    """
    A simple random agent for demonstration.
    Replace this with your actual agent logic.
    """
    # Reset the task
    state = task.reset()
    total_reward = 0

    # Get available tools
    tools = task.list_tools()

    for step in range(max_steps):
        # Your agent's decision logic here
        # For demo, we'll just print the observation
        print(f"\nStep {step + 1}")
        print(f"Observation: {state.observation}")

        # Agent selects an action (simplified)
        # In reality, your agent would use LLM reasoning, planning, etc.
        selected_tool = tools[0]  # Just pick first tool for demo

        # Execute the action
        result = task.call_tool(selected_tool.name, {})
        print(f"Executed: {selected_tool.name}")
        print(f"Result: {result}")

        # Evaluate the current state
        state = task.evaluate()
        total_reward += state.reward

        # Check if task is complete
        if state.terminated or state.truncated:
            print(f"\nTask completed!")
            print(f"Total reward: {total_reward}")
            print(f"Success: {state.info.get('success', False)}")
            break

    return total_reward

# Run the agent
benchmark = LocalRunner("cube-benchmark-miniwob")
task = benchmark.spawn(task_id="click-button", seed=42)

try:
    reward = simple_agent(task)
    print(f"\nFinal reward: {reward}")
finally:
    task.close()
```

## Multi-Benchmark Evaluation

One of CUBE's key benefits is easy evaluation across multiple benchmarks:

```python
from cube import Registry, LocalRunner

def evaluate_agent_on_benchmarks(agent_fn, benchmark_names):
    """Evaluate an agent across multiple CUBE benchmarks."""
    results = {}

    for benchmark_name in benchmark_names:
        print(f"\n{'='*60}")
        print(f"Evaluating on: {benchmark_name}")
        print(f"{'='*60}")

        benchmark = LocalRunner(benchmark_name)
        tasks = benchmark.list_tasks(limit=5)  # Evaluate on first 5 tasks

        benchmark_results = []
        for task_info in tasks:
            task = benchmark.spawn(task_id=task_info.id, seed=42)
            try:
                reward = agent_fn(task)
                benchmark_results.append({
                    'task_id': task_info.id,
                    'reward': reward
                })
            finally:
                task.close()

        # Compute aggregate metrics
        avg_reward = sum(r['reward'] for r in benchmark_results) / len(benchmark_results)
        results[benchmark_name] = {
            'task_count': len(benchmark_results),
            'average_reward': avg_reward,
            'per_task': benchmark_results
        }

        print(f"Average reward: {avg_reward:.3f}")

    return results

# Evaluate across multiple benchmarks
benchmarks = [
    "cube-benchmark-miniwob",
    "cube-benchmark-webarena-lite",
    "cube-benchmark-gaia-mini"
]

results = evaluate_agent_on_benchmarks(simple_agent, benchmarks)

# Print summary
print(f"\n{'='*60}")
print("EVALUATION SUMMARY")
print(f"{'='*60}")
for benchmark, metrics in results.items():
    print(f"{benchmark}: {metrics['average_reward']:.3f} avg reward ({metrics['task_count']} tasks)")
```

## Filtering Benchmarks by Requirements

Use the registry to find benchmarks that match your infrastructure constraints:

```python
from cube import Registry

registry = Registry()

# Find lightweight benchmarks suitable for laptop development
lightweight = registry.list(
    runtime="docker",
    max_ram_gb=4,
    max_disk_gb=10,
    no_gpu=True
)

print(f"Lightweight benchmarks: {len(lightweight)}")
for b in lightweight:
    print(f"  - {b.name}: {b.task_count} tasks, {b.estimated_tokens}k tokens")

# Find benchmarks in specific domain
web_benchmarks = registry.list(
    domain="web_navigation",
    benchmark_license="commercial_use_allowed"
)

# Find benchmarks with specific compliance
safe_benchmarks = registry.list(
    compliance=["no-docker-root", "task-isolated"]
)
```

## Next Steps

Now that you've run your first CUBE evaluation:

1. **Understand the APIs**: Read the [API Reference](api/) to learn about all available methods
2. **Create your own wrapper**: Follow the [Benchmark Author Guide](guides/benchmark-authors.html) to make your benchmark CUBE-compliant
3. **Integrate with your platform**: See the [Platform Developer Guide](guides/platform-developers.html) to add CUBE support to your evaluation harness
4. **Explore examples**: Check out [Examples](examples/) for more advanced use cases

## Common Issues

### ModuleNotFoundError: No module named 'cube'

Make sure you've installed the CUBE standard library:
```bash
pip install cube-standard
```

### Benchmark requires Docker but Docker is not running

Some benchmarks require Docker. Start the Docker daemon:
```bash
# macOS/Linux
sudo systemctl start docker

# Or use Docker Desktop
```

### Port already in use

If running multiple benchmark instances, ports may conflict. Use the benchmark-level API to manage multiple task instances:

```python
benchmark = LocalRunner("cube-benchmark-example")

# Spawn multiple tasks - the benchmark handles port allocation
task1 = benchmark.spawn(task_id="task-1")
task2 = benchmark.spawn(task_id="task-2")
task3 = benchmark.spawn(task_id="task-3")
```

### Task hangs or times out

Set timeout parameters when connecting:

```python
from cube import LocalRunner

benchmark = LocalRunner("cube-benchmark-example", timeout=300)  # 5 minutes
task = benchmark.spawn(task_id="slow-task", timeout=600)  # 10 minutes for this task
```

## Getting Help

- **Documentation**: [Full API Reference](api/)
- **GitHub Issues**: [Report bugs or ask questions](https://github.com/The-AI-Alliance/cube-standard/issues)
- **Discussions**: [Community forum](https://github.com/The-AI-Alliance/cube-standard/discussions)
- **Email**: [contact@thealliance.ai](mailto:contact@thealliance.ai?subject=CUBE%20Question)

---

**Continue learning**:
- [Overview](overview.html): Understand CUBE's architecture
- [API Reference](api/): Complete specification
- [Benchmark Authors Guide](guides/benchmark-authors.html): Create CUBE benchmarks
