# Toy Counter Benchmark

A minimal toy benchmark for testing and demonstrating the CUBE benchmark and task-level API with the simplest possible example.

## Overview

This benchmark provides a simple counter that increments from 0 to a target value. It's designed to be the absolute minimum implementation of the CUBE framework:

- **Single integer state**: Just a counter value
- **Two simple actions**: `increment()` and `get_value()`
- **Clear validation**: Task completes when counter reaches target
- **Minimal code**: All components in a single file (~220 lines)

## Structure

```
toy_benchmark/
└── counter_benchmark/
    └── counter.py       # Complete implementation in one file
```

## Counter Benchmark

The counter is the simplest possible state machine:
- Starts at 0
- Can be incremented by 1
- Tasks validate when a target value is reached

### Actions

- **increment()**: Adds 1 to the counter, returns "Counter incremented"
- **get_value()**: Returns current counter value as a string

### Tasks

The benchmark includes 2 tasks:
- **count-to-3**: Increment counter to reach value 3
- **count-to-5**: Increment counter to reach value 5

## Key Concepts Demonstrated

### 1. Side-Effect Actions

```python
def increment(self) -> str:
    """Increment counter by 1."""
    self.value += 1  # SIDE EFFECT: modify internal state
    self.history.append("increment")
    return "Counter incremented"  # Simple confirmation
```

**Key Point**: The action modifies `self.value` (side effect) and returns a simple confirmation message.

### 2. State-Based Validation

```python
def validate_task(self, obs: Observation) -> tuple[float, dict[str, Any]]:
    """Validate if counter reached target."""
    if self._tool.value == self.target:  # Check internal state
        return 1.0, {"solved": True, "value": self._tool.value}
    # Partial reward based on progress
    progress = min(1.0, self._tool.value / self.target)
    return progress * 0.5, {"solved": False, "value": self._tool.value}
```

**Key Point**: Tasks validate by checking `tool.value` (internal state), not by parsing observation strings.

## Running the Test

```bash
# From the cube-standard root directory
uv run python examples/counter_benchmark/counter.py
```

Expected output:
```
Starting counter benchmark test...
Step 1: Counter incremented
Step 2: Counter incremented
Step 3: Counter incremented

Evaluation: reward=1.0, info={'solved': True, 'value': 3, 'steps': 3}

✓ Test passed!
```

## API Example

### Complete Test (from counter.py)

```python
from counter import CounterBenchmark
from cube.types import (
    MCPCallToolRequest,
    MCPCallToolRequestParams,
    ResetRequest,
    ShutdownRequest,
    SpawnRequest,
)

# Create and setup benchmark
benchmark = CounterBenchmark()
benchmark.setup(available_ports=[9000], server_mode=False)

# Spawn task "count-to-3"
spawn_resp = benchmark.spawn(SpawnRequest(task_id="count-to-3"))
session = spawn_resp.other["session"]

# Reset task
session.reset(ResetRequest())

# Call increment() 3 times
for i in range(3):
    result = session.call_tool(
        MCPCallToolRequest(
            params=MCPCallToolRequestParams(name="increment", arguments={})
        )
    )
    print(f"Step {i+1}: {result.content[0].text}")

# Evaluate - should be solved
eval_result = session.evaluate()
assert eval_result.reward == 1.0
assert eval_result.info["solved"]

# Cleanup
benchmark.shutdown(ShutdownRequest(session_id=spawn_resp.session_id))
benchmark.close()
```

## Example Walkthrough: Counting to 3

```python
# Task: Increment counter to 3 starting from 0

# Step 1: Call increment (side effect: 0 → 1)
session.call_tool(MCPCallToolRequest(
    params=MCPCallToolRequestParams(name="increment", arguments={})
))
# Returns: "Counter incremented"
# Internal state: tool.value = 1

# Step 2: Call increment (side effect: 1 → 2)
session.call_tool(MCPCallToolRequest(
    params=MCPCallToolRequestParams(name="increment", arguments={})
))
# Returns: "Counter incremented"
# Internal state: tool.value = 2

# Step 3: Call increment (side effect: 2 → 3)
session.call_tool(MCPCallToolRequest(
    params=MCPCallToolRequestParams(name="increment", arguments={})
))
# Returns: "Counter incremented"
# Internal state: tool.value = 3

# Step 4: Evaluate (checks internal state)
eval_result = session.evaluate()
# Task checks: tool.value == 3 (target value)
# Returns: reward=1.0, solved=True
```

## What This Example Verifies

This minimal benchmark verifies the complete CUBE pipeline:

1. **Tool action execution** - increment() works
2. **Internal state modification** - value increases correctly
3. **Task validation** - detects when target is reached
4. **Reward calculation** - returns 1.0 when solved, partial rewards for progress
5. **Benchmark orchestration** - spawn, reset, evaluate, shutdown all work
6. **Complete framework integration** - all CUBE components work together

## Learning from This Example

This toy benchmark demonstrates:

1. **Simplest Possible State**: A single integer (can't get simpler)
2. **Side-Effect Pattern**: Actions modify internal state, return confirmations
3. **State vs Observations**: State lives in tool, observations confirm actions
4. **Minimal Implementation**: All components in ~220 lines of readable code
5. **Easy to Verify**: Count to 3 - anyone can validate it works
6. **Complete CUBE API**: Despite being minimal, it exercises the full framework

## Next Steps

To create your own benchmark:

1. **Define your tool** by subclassing `Tool` and creating a `Protocol` for actions
2. **Implement actions** that modify internal state and return simple strings
3. **Create tasks** by subclassing `Task` and validating based on tool's internal state
4. **Create a benchmark** by subclassing `Benchmark` and implementing `load_tasks()`
5. **Add a test function** to verify everything works

See `examples/counter_benchmark/counter.py` for a complete, minimal reference implementation.
