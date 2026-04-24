# CUBE architecture — 5 layers + invariants

Read this before phase 2 (Reflect). Keeps you honest when filling TODOs later.

## The 5 layers

| Layer | Module | What it contains |
|-------|--------|------------------|
| 1. Core | `cube.core` | `Action`, `Observation`, `Content`, `EnvironmentOutput`, `StepError`, `TypedBaseModel` |
| 2. Tool | `cube.tool` | `Tool`, `AsyncTool`, `@tool_action`, `ToolConfig`, `Toolbox` |
| 3. Task | `cube.task` | `Task`, `TaskMetadata`, `TaskConfig`, gym-style `reset / step / evaluate` |
| 4. Benchmark | `cube.benchmark` | `Benchmark`, `BenchmarkMetadata`, ClassVar registries |
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
- `TaskConfig.make()` imports the Benchmark class **inside the method**, not at module top, to avoid a circular import.

### Benchmark layer
- Three ClassVars are **required at class definition**: `benchmark_metadata`, `task_metadata`, `task_config_class`. Missing any ⇒ `TypeError`.
- `task_metadata` lives on the **Benchmark**, not on `TaskConfig`. Keep `TaskConfig` as a lean serialization payload.
- If the ClassVars aren't declared, the framework looks for `benchmark_metadata.json` / `task_metadata.json` next to the benchmark module.
- `_setup()` populates `self._runtime_context` with shared L2 infra. `close()` tears it down.
- `install()`, `_setup()`, `close()` must all be idempotent.
- `_runtime_context` is read-only after `setup()` in parallel runs.

### Testing layer
- Every debug task must reach `reward == 1.0`.
- `task.close()` is called twice in the compliance suite (idempotency check).
- `reset()` must be reproducible: two resets of the same config ⇒ identical first observation.

## Serialization boundary

`TaskConfig` crosses process boundaries. `Task` never does. Workers receive a pickled `TaskConfig` and call `.make()` locally.

## Config → Factory pattern

Everywhere: `XyzConfig.make() → Xyz`. The Config is pure data and serializable. The live object owns runtime state and is not serialized.

Credentials belong in env vars, resolved at runtime. Never store credentials on any `Config` subclass.

## Pre-setup vs post-setup mental model

Today `Benchmark` is one class. After #111 lands it will split into `BenchmarkConfig` (pure data, serializable) and `Benchmark` (live, post-setup). Use that split mentally even now:

**Pre-setup (BenchmarkConfig-shaped)** — safe to call on a fresh instance:
- `install()` — populate any heavy per-task caches, build `task_metadata.json`
- `subset_from_list([...])`, `subset_from_glob(...)`, `subset_from_name(...)`
- `get_task_configs()` — yields serializable `TaskConfig`s

**`setup()`** — populates `_runtime_context` with shared L2 infra (browser session, DB, remote API). Call it **after** any subsetting / iteration.

**Post-setup (Benchmark-shaped)** — safe to call only after `setup()`:
- `task_config.make(runtime_context=...)` uses the populated runtime context to build live tasks.

**`close()`** — tears down whatever `setup()` created. Idempotent.

Rule of thumb: the "shape" of the benchmark can change pre-setup (subsetting, installing) but is frozen post-setup.

## Forward-looking notes

- **#111** will split `Benchmark` into `BenchmarkConfig` + `Benchmark`, mirroring `TaskConfig` / `Task`. Not yet landed — don't anticipate it in scaffolded code.
- **harness #300** is deprecating `container_backend` in favor of `InfraConfig` + `ResourceConfig` + lazy `ResourceHandle`. Template still carries `container_backend`; this is tracked for future cleanup.
- **Streaming actions** and **streaming observations** are not in the current protocol. Flag any audio/video benchmark as requiring extension work.
- **Multi-agent** and fully async tools are on the `core-extensions/` RFC roadmap. Scope down if a user asks for them.
