# counter-cube

The canonical minimal CUBE implementation for contributors. It solves the
simplest possible task — increment a counter to reach a target — so the
framework patterns are never obscured by domain complexity.

Read `src/counter_cube/counter.py` top-to-bottom. Each of the four layers has
a block comment explaining its role and the design decisions behind it.

## Architecture

```
ConfigurableCounterTool   (Tool)          — the environment the agent acts in
        ↑ created by
CounterToolConfig         (ToolConfig)    — serializable factory; crosses process boundaries
        ↑ used by
ReachTargetTask           (Task)          — owns one Tool; implements reset/step/evaluate
        ↑ created by
CounterTaskConfig         (TaskConfig)    — serializable task description; calls make()
        ↑ registered in
CounterBenchmark          (Benchmark)     — registry; holds metadata; vends TaskConfigs
```

## The four layers at a glance

| Layer | Your job |
|---|---|
| **Tool** | Wrap the environment. Decorate each agent-callable method with `@tool_action`. |
| **ToolConfig** | Pydantic model. Carry constructor params. Implement `make()` to build the Tool. |
| **Task** | Implement `reset()`, `evaluate()`, and optionally `finished()`. |
| **Benchmark** | Set three `ClassVar` attributes: `benchmark_metadata`, `task_metadata`, `task_config_class`. |

## Install

```bash
cd examples/counter-cube
uv sync
```

This installs `counter-cube` and picks up `cube-standard` as an editable
dependency from `../../`.

## Run tests

```bash
pytest tests/ -v
```

Expected: 7 tests collected, all passing.

## Usage

```python
from counter_cube import CounterBenchmark
from cube.core import Action

benchmark = CounterBenchmark()
benchmark.setup()

for task_config in benchmark.get_task_configs():
    task = task_config.make()
    obs, info = task.reset()
    while True:
        env_out = task.step(Action(name="increment", arguments={}))
        if env_out.done:
            break
    print(f"{task_config.task_id}: reward={env_out.reward}")
    task.close()

benchmark.close()
```

## Further reading

- **[Authoring a CUBE guide](https://the-ai-alliance.github.io/cube-standard/authoring-a-cube)** — if you're new to CUBE, start here; this directory is the reference implementation it points to
- [`openspec/specs/`](../../openspec/specs/) — formal per-layer contracts
- [`examples/toy_benchmark/`](../toy_benchmark/) — flat single-file variant of the same example
- [`cube-harness/cubes/osworld-cube/`](https://github.com/The-AI-Alliance/cube-harness/tree/main/cubes/osworld-cube) — a real cube using VMs and desktop automation (lives in the cube-harness repo)
