# new-cube-package — CUBE benchmark template

This directory is the canonical starting point for a new CUBE benchmark package.
Copy it, rename things, and follow the TODOs in each file.

## Quick start

```bash
# 1. Copy the template
cp -r src/cube/_template/new_cube_package my-bench

# 2. Rename placeholders (all at once with sed, or use your editor)
#    cube_package  → my_bench
#    CubeBenchmark → MyBenchmark
#    new-cube-package → my-bench

# 3. Install in editable mode
cd my-bench
uv sync

# 4. Run the debug compliance suite
cube test my-bench        # uses the registered entry-point name
# or: cube test my_bench.debug
```

## File map

```
new_cube_package/
├── pyproject.toml              ← package metadata & cube.benchmarks entry point
└── src/cube_package/
    ├── __init__.py
    ├── benchmark.py            ← CubeBenchmark (registry, metadata, task list)
    ├── benchmark_metadata.csv  ← Option B: load benchmark metadata from CSV
    ├── task.py                 ← CubeTask + CubeTaskConfig (episode loop)
    ├── task_metadata.csv       ← Option B: load task metadata from CSV
    ├── tool.py                 ← CubeTool + CubeToolConfig + @tool_action methods
    └── debug.py                ← deterministic agent for `cube test`
```

## Checklist

- [ ] `pyproject.toml` — update `name`, `description`, and the `cube.benchmarks` entry-point key
- [ ] `benchmark.py` — fill in `BenchmarkMetadata` and `task_metadata` (or switch to CSV)
- [ ] `tool.py` — add `@tool_action` methods; delete `example_action` placeholder
- [ ] `task.py` — implement `reset()`, `evaluate()`, and (optionally) `finished()`
- [ ] `debug.py` — add one entry to `_TASK_ACTIONS` per task; sequences must reach `reward == 1.0`
- [ ] Run `cube test <your-benchmark-name>` — all tasks must pass

## How `cube test` works

`cube test` accepts either the registered benchmark name (`my-bench`) or the
dotted module path (`my_bench.debug`).  When given a name it resolves it via
the `cube.benchmarks` entry-point group and automatically loads
`<package_root>.debug`.

The debug module must expose two callables:

| symbol | signature |
|---|---|
| `get_debug_task_configs()` | `() → list[TaskConfig]` |
| `make_debug_agent(task_id)` | `(str) → agent callable` |

See `debug.py` for the full template.
