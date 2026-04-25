# CUBE architecture — 5 layers + invariants

Read this before phase 2 (Reflect). Keeps you honest when filling TODOs later.

## The 5 layers

| Layer | Module | What it contains |
|-------|--------|------------------|
| 1. Core | `cube.core` | `Action`, `Observation`, `Content`, `EnvironmentOutput`, `StepError`, `TypedBaseModel` |
| 2. Tool | `cube.tool` | `Tool`, `AsyncTool`, `@tool_action`, `ToolConfig`, `Toolbox` |
| 3. Task | `cube.task` | `Task`, `TaskMetadata`, `TaskConfig`, gym-style `reset / step / evaluate` |
| 4. Benchmark | `cube.benchmark` | `BenchmarkConfig` (serializable registry) + `Benchmark` (runtime pair), `BenchmarkMetadata`, ClassVar registries |
| 5. Testing | `cube.testing` | `run_debug_suite`, `assert_debug_tasks_reward_one` |

Cross-cutting:
- `cube.resource` — L1/L2/L3 resource lifecycle (L1 provisioned images, L2 benchmark-scoped, L3 task-scoped)
- `cube.container` — single-container abstraction
- `cube.server` — JSON-RPC 2.0 endpoints (MCP-compatible)
- `cube.cli` — `cube init / list / test / registry`

## Hard invariants — always true

### Tool layer
- Action discovery happens via `@tool_action`. No decorator ⇒ action invisible to the agent.
- `ToolConfig` must be JSON-serializable. `Tool` instance is never serialized.
- `AsyncTool` methods must be `async`. Mixing sync `@tool_action` into an async tool = class-definition error.
- `execute_action(action)` must never raise. Return an `Observation` or `StepError`.

### Task layer
- `reset()` MUST call `self.tool.reset()`.
- `evaluate()` is pure — reads state, never mutates.
- `step()` is concrete on the base class. Do not override.
- Tool launches in `model_post_init` — construction failures can leak containers; clean up in `close()`.
- `TaskConfig` carries `metadata: TaskMetadata` stamped by `get_task_configs()`. Workers are self-contained — the config carries everything needed to build the task.

### Benchmark layer
- Two classes: `BenchmarkConfig(TypedBaseModel, ABC)` is serializable; `Benchmark(ABC)` is the runtime pair (plain class, not Pydantic). Never serialize a `Benchmark`.
- Four ClassVars are **required on `BenchmarkConfig`**: `benchmark_metadata`, `task_metadata`, `task_config_class`, `benchmark_class`. Missing any ⇒ `TypeError`.
- `task_metadata` lives on **`BenchmarkConfig`**, not directly on `TaskConfig`. `get_task_configs()` stamps each emitted `TaskConfig` with `metadata=` from the ClassVar at emit time.
- If the ClassVars aren't declared, the framework looks for `benchmark_metadata.json` / `task_metadata.json` next to the benchmark module.
- `BenchmarkConfig.make(infra=None)` provisions resources, calls `Benchmark.setup()`, returns a ready `Benchmark`.
- `Benchmark._setup()` populates `self._runtime_context` with shared L2 infra. `Benchmark.close()` tears it down.
- `BenchmarkConfig.install()`, `Benchmark._setup()`, `Benchmark.close()` must all be idempotent.
- `_runtime_context` is read-only after `_setup()` in parallel runs.

### Testing layer
- Every debug task must reach `reward == 1.0`.
- `task.close()` is called twice in the compliance suite (idempotency check).
- `reset()` must be reproducible: two resets of the same config ⇒ identical first observation.

## Serialization boundary

`TaskConfig` crosses process boundaries. `Task` and `Benchmark` never do. Workers receive a `TaskConfig` (JSON round-tripped), call `.make()` locally, and read `self.metadata` directly — no disk access needed.

`BenchmarkConfig` is also serializable (it's a `TypedBaseModel`). It crosses process boundaries when spawning the RPC server subprocess.

All polymorphic fields use `SerializeAsAny` to preserve subclass state through JSON round-trip.

## Config → Factory pattern

Everywhere: `XyzConfig.make() → Xyz`. The Config is pure data and serializable. The live object owns runtime state and is not serialized.

Credentials belong in env vars, resolved at runtime. Never store credentials on any `Config` subclass.

## BenchmarkConfig vs Benchmark mental model

```
BenchmarkConfig   →  make(infra=None)  →  Benchmark
  TypedBaseModel                           plain class
  serializable                             not serializable
  ClassVar task_metadata                   _runtime_context
  install() / get_task_configs()           _setup() / close() / spawn()
  subset_from_list() / subset_from_glob()
```

**BenchmarkConfig** — safe to call on a fresh instance:
- `install()` — populate any heavy per-task caches, build `task_metadata.json`
- `subset_from_list([...])`, `subset_from_glob(...)`, `subset_from_name(...)`
- `get_task_configs()` — yields serializable `TaskConfig`s, each stamped with `metadata=`

**`make(infra=None)`** — provisions resources, calls `_setup()`, returns a ready `Benchmark`. Context-manager form (`with config.make() as bench`) calls `close()` automatically.

**Benchmark** — safe to call only after `make()`:
- `spawn(task_config)` — validates task_id and calls `task_config.make(runtime_context=self._runtime_context, ...)`
- `close()` — tears down whatever `_setup()` created. Idempotent.

## Forward-looking notes

- **harness #300** is deprecating `container_backend` in favor of `InfraConfig` + `ResourceConfig` + lazy `ResourceHandle`. Template still carries `container_backend`; this is tracked for future cleanup.
- **Streaming actions** and **streaming observations** are not in the current protocol. Flag any audio/video benchmark as requiring extension work.
- **Multi-agent** and fully async tools are on the `core-extensions/` RFC roadmap. Scope down if a user asks for them.
