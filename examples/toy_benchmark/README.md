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

# Run tests
python examples/toy_benchmark/counter.py
```

Check [counter.py](counter.py) for the complete implementation.
