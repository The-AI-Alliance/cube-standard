# Toy Counter Benchmark

A minimal example demonstrating CUBE's task/tool API and ToolConfig flexibility.

## What it tests

- **Single action execution** via `.step()`
- **Multiple action execution** via `.step()`
- **Lower level API** via `.tool.execute_action()`
- **ToolConfig flexibility**: adding/removing actions and changing action behavior
- **Task isolation** (independent tool instances)

## Run

```bash
# Install dependencies
make install

# Run without containers
PYTHONPATH=src uv run python examples/toy_benchmark/counter.py --backend none

# Swap container backend (same benchmark code)
PYTHONPATH=src uv run --extra docker python examples/toy_benchmark/counter.py --backend docker
PYTHONPATH=src uv run --extra daytona python examples/toy_benchmark/counter.py --backend daytona
PYTHONPATH=src uv run --extra modal python examples/toy_benchmark/counter.py --backend modal

# Quick backend wiring check
PYTHONPATH=src uv run --extra docker python examples/toy_benchmark/counter.py --backend docker --smoke
```

Check [counter.py](counter.py) for the complete implementation.
