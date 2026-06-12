# CUBE architecture — 5 layers + invariants

Read this before phase 2 (Reflect). Keeps you honest when filling TODOs later.

## The 5 layers

| Layer | Module | What it contains |
|-------|--------|------------------|
| 1. Core | `cube.core` | `Action`, `Observation`, `Content`, `EnvironmentOutput`, `StepError`, `TypedBaseModel` |
| 2. Tool | `cube.tool` | `Tool` (one unified class), `@tool_action`, `ToolConfig`, `Toolbox` |
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
- **One unified `Tool` class.** `@tool_action` methods may be sync `def` OR `async def` on the **same** class — `Tool` dispatches per the method's own keyword. There is no separate `AsyncTool` / `AsyncToolbox`; a cube always subclasses `Tool`.
- **`final_step` is universal.** Every `Tool` already exposes a `final_step` action that raises `AgentStop` to end the episode. Cubes must NOT define their own `final_step` (overriding it to return a value breaks termination) and must NOT add a STOP schema by hand.
- **Errors are observations, not exceptions.** A `@tool_action` that raises is caught: `execute_action` returns an `Observation` with the error text in `contents` plus a structured `StepError` on `obs.error` (non-terminal — the agent reads it and retries). Don't try/except inside actions just to report normal errors. `execute_action` itself never raises (except `AgentStop`, which it lets through).

### Task layer
- `reset()` MUST call `self.tool.reset()`.
- `evaluate()` is pure — reads state, never mutates.
- `step()` is concrete on the base class. Do not override.
- **One tool-lifecycle hook: `_make_tool(self, role=None) -> Tool`.** It RETURNS the tool; the base stores `self._tool = self._make_tool()` in `model_post_init`. This is the single place for once-per-task world prep (relocate a read-only dir, fix perms) AND building the tool. There is no `make_tool` / `prepare_world` / `_build_tool` — those are gone. Default `_make_tool` just returns `self.tool_config.make(container=self._container)`; override it (and `return` the new tool) for any per-task setup.
- **Restrict advertised actions via `_filter_actions(self, actions, role=None)`** (advisory whitelist/mask, recomputed per `action_set` access). Shape per-seat observations via `obs_postprocess(self, obs, role=None)`. (The old `filter_actions` is gone.)
- The agent never touches the Task. It holds an `AgentView` (formerly `TaskTool`) from `task.get_agent_view()`, which exposes only `action_set` + `execute_action`.
- Tool launches in `model_post_init` — construction failures can leak containers; clean up in `close()`.
- `TaskConfig` carries `metadata: TaskMetadata` stamped by `get_task_configs()`. Workers are self-contained — the config carries everything needed to build the task.
- `Task` is generic over `[TMeta, TTool]`. Bind both when the tool surface is known: `class FooTask(Task[FooMeta, TerminalTool])` types `self.tool` directly. Second param defaults to `AbstractTool` — `Task[Meta]` keeps working.

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
- `subset_from_list([...])`, `subset_from_glob(glob_key, pattern)`, `named_subset(name)`
- `get_task_configs()` — yields serializable `TaskConfig`s, each stamped with `metadata=`

**`make(infra=None)`** — provisions resources, calls `_setup()`, returns a ready `Benchmark`. Context-manager form (`with config.make() as bench`) calls `close()` automatically.

**Benchmark** — safe to call only after `make()`:
- `spawn(task_config)` — validates task_id and calls `task_config.make(runtime_context=self._runtime_context, ...)`
- `close()` — tears down whatever `_setup()` created. Idempotent.

## Forward-looking notes

- **`container_backend` is now `Field(deprecated=True)`** on `BenchmarkConfig` — slated for removal once all in-tree benchmarks migrate to declaring container needs via `resources: list[ResourceConfig]` (`InfraConfig` + `ResourceConfig` + lazy `ResourceHandle`). The template still carries `container_backend` for compatibility; do not strip it from `TaskConfig.make()` signatures yet.
- **Streaming actions** and **streaming observations** are not in the current protocol. Flag any audio/video benchmark as requiring extension work.
- **Multi-agent is supported** (only reach for it when the cube genuinely needs cooperating seats): `agent_roles() -> {role: count}` (default `{None: 1}`) declares the roster; override `get_agent_view(role)` to build each seat (the base raises `NotImplementedError` for a named role, so the cube owns the seat index and per-role tool via `_make_tool(role)`). Single-agent cubes ignore all of this. `accept_agent_stop` no longer exists.
