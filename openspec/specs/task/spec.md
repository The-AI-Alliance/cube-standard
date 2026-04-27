# Task Layer

**Module:** `cube.task` | **Layer:** 3 (per-task environment dynamics + evaluation)

## Purpose

A `Task` is a single scoreable problem. It unifies gym-style dynamics
(`reset`/`step`/`close`) with task-specific logic (`evaluate`, `filter_actions`,
`obs_postprocess`). One class, one place for the benchmark author to define both
what the agent can do and how it is scored.

## Public API

### `TaskMetadata` (serializable)
```python
class TaskMetadata(TypedBaseModel):
    id: str                                    # unique task identifier
    split: Literal["train", "val", "test"] = "test"
    abstract_description: str = ""             # for search/filtering only — NOT the objective
    recommended_max_steps: int | None = None   # harness hint, not enforced
    container_config: ContainerConfig | None = None
```

`TaskMetadata` is the lightweight, eager-loaded view of a task: it ships in
the wheel and powers `cube list`, registry listings, glob-based subsetting,
and human inspection. Cube authors needing additional per-task fields
subclass `TaskMetadata` with named typed fields; polymorphism is preserved
through the `TypedBaseModel` `_type` discriminator. The base class accepts
only the framework-defined fields above.

Heavy per-task data (problem statements, patches, archives, evaluator
scripts, …) lives on a `TaskExecutionInfo` subclass surfaced via
`Task.execution_info`, not on `TaskMetadata`. See below.

The actual task objective is surfaced in the first `Observation` returned by `reset()`.
`abstract_description` is for tooling (search, subsetting), never to be shown to the agent.

### `TaskExecutionInfo` (serializable)
```python
class TaskExecutionInfo(TypedBaseModel):
    """Heavy, lazy per-task execution data."""
```

Subclassed by cubes that need heavy per-task data, e.g.:
```python
class SWEBenchExecutionInfo(TaskExecutionInfo):
    problem_statement: str
    patch: str
    test_patch: str
    fail_to_pass: list[str]
    pass_to_pass: list[str]
```

Polymorphic via the `TypedBaseModel` `_type` discriminator. Populated on
the worker — typically inside `TaskConfig.make()` by validating
`cls.load_task_execution_info(self.task_id)` against the subclass, but
`Task.model_post_init` and `Task.reset()` are also valid hydration points.
The framework never reads `execution_info` itself; it is for cube authors
to surface domain-specific heavy data with autocomplete and Pydantic
validation.

Cubes with no heavy data leave the slot `None`; the base class is
instantiable but carries no fields.

### `Task` (abstract, Pydantic)
```python
class Task(TypedBaseModel, ABC):
    # Serializable fields
    metadata: TaskMetadata
    execution_info: TaskExecutionInfo | None = None    # heavy lazy data; populated on the worker
    tool_config: ToolConfig
    container_backend: ContainerBackend | None = None
    runtime_context: RuntimeContext | None = None      # from Benchmark._setup()
    validate_per_step: bool = False                    # eval on every step, not just done
    accept_agent_stop: bool = True                     # accept STOP_ACTION from agent

    # Runtime (PrivateAttr, set in model_post_init)
    _tool: AbstractTool | None
    _container: Container | None
```

`execution_info` is the typed surface for heavy per-task data. Cubes
populate it from one of three places:
- inside `TaskConfig.make()` — most common; validate
  `cls.load_task_execution_info(self.task_id)` against the cube's
  `TaskExecutionInfo` subclass and pass to the `Task` constructor.
- inside `Task.model_post_init` — for cubes that prefer hydration during
  Task construction.
- inside `Task.reset()` — for cubes that defer hydration to first reset.

Tasks read typed fields directly: `self.execution_info.problem_statement`,
`self.execution_info.patch`, …

`model_post_init` (runs after Pydantic `__init__`):
1. If `container_backend` and `metadata.container_config` are both set, launch the container.
2. Call `tool_config.make(container=self._container)` to build the tool.

**Abstract methods (implementers MUST provide):**
- `reset() -> (Observation, dict)` — set up initial state; also call `self.tool.reset()`
- `evaluate(obs: Observation | None = None) -> (float, dict)` — score the current state

**Optional hooks (default: identity / no-op):**
- `filter_actions(actions: list[ActionSchema]) -> list[ActionSchema]` — whitelist subset of tool actions
- `obs_postprocess(obs: Observation) -> Observation` — transform observations before returning
- `finished(obs: Observation | None = None) -> bool` — early termination check
- `get_privileged_info() -> Content` — solution, eval source, internal state (for debug/oracle agents)
- `get_status() -> str` — free-form status string
- `close()` — cleanup; default calls `self.tool.close()`. Override to add cleanup and call `super().close()`.

### `Task.step()` (concrete; do not override)

Signature: `step(action: Action | list[Action]) -> EnvironmentOutput`

Accepts single Action or list (atomic multi-action step). Logic:

1. Loop over actions:
   - If `action.name == STOP_ACTION.name` and `self.accept_agent_stop`: append
     `"Task finished by the agent."` observation, set `done=True`, break.
   - Otherwise, call `self.tool.execute_action(action)`. Time it.
   - If result is `Observation`: `obs += result`
   - If result is `StepError`: set `error`, `done=True`, break
   - Any other type → raise `ValueError`
2. `done = done or self.finished(obs)`
3. If `done` or `self.validate_per_step`: call `self.evaluate(obs)` → `(reward, info)`
4. Apply `obs = self.obs_postprocess(obs)`
5. Populate `info["profiling"]` with `tool_execute`, `evaluate`, `obs_postprocess` timings

Returns `EnvironmentOutput(obs, reward, done, info, error)`. `truncated` is always
`False` — step/time-limit truncation is the harness's responsibility (TODO in code).

### `STOP_ACTION` (module-level constant)
```python
STOP_ACTION = ActionSchema(name="final_step", description="Stop the task execution.")
```
Protocol for agent-initiated termination. Tasks that reject it must set
`accept_agent_stop = False` (e.g., tasks that require the agent to reach a terminal
state via environment interaction, not a self-declaration).

### `RuntimeContext`
```python
RuntimeContext = dict[str, Any]
```
Free-form dict populated by `Benchmark._setup()` with shared infrastructure references
(server URLs, DB connections, handles to launched L2 resources). Passed to every Task
spawned from that benchmark.

**Concurrency:** after `setup()` returns, `RuntimeContext` is treated as read-only by
concurrent tasks. Writes are not safe across workers.

### `TaskConfig` (abstract, serializable)
```python
class TaskConfig(TypedBaseModel, ABC):
    task_id: str
    metadata: TaskMetadata                      # travels with the config
    seed: int | None = None
    tool_config: ToolConfig | None = None
    sub_bench_name: str | None = None            # composite routing hint

    @abstractmethod
    def make(
        self,
        runtime_context: RuntimeContext | None = None,
        container_backend: ContainerBackend | None = None,
    ) -> Task

    @classmethod
    def task_execution_cache_dir(cls) -> Path:
        """Default: ~/.cube/<top-level-package-name>/tasks_execution_info/."""

    @classmethod
    def load_task_execution_info(cls, task_id: str) -> dict[str, Any]:
        """Read processed per-task data written by BenchmarkConfig.install()."""

    @classmethod
    def verify_installed(cls) -> None:
        """Optional fail-fast check. Default: no-op."""
```

**Self-contained unit.** Workers receive a `TaskConfig` and have everything
they need — metadata is embedded via the `metadata` field (stamped on each
emitted config by `BenchmarkConfig.get_task_configs()` on the driver).
`make()` uses `self.metadata` directly; no import of the owning
`BenchmarkConfig` is needed on the worker. This is the single most important
invariant of the layer: the serialization boundary is self-describing.

Subclasses that carry heavy install-time data (e.g. SWE-bench problem
statements, OSWorld evaluator configs) declare a `TaskExecutionInfo`
subclass for the heavy fields and populate `Task.execution_info` inside
`TaskConfig.make()` — typically by calling
`cls.load_task_execution_info(self.task_id)` (read from the per-task
on-disk cache) and validating the resulting dict against the
`TaskExecutionInfo` subclass. The cache itself is written by
`BenchmarkConfig.install()` — operators run `cube install <bench>` once
per worker environment.

**Per-task cache helpers (worker-side classmethods).**
- `task_execution_cache_dir()` — default
  `~/.cube/<top-level-package-name>/tasks_execution_info/` where
  `<top-level-package-name>` is `cls.__module__.split(".")[0]`. Override
  on subclasses whose owning benchmark uses a non-default cache layout
  (e.g. when the benchmark display name differs from the Python package
  name). `BenchmarkConfig.install()` writes via
  `cls.task_config_class.task_execution_cache_dir()` so the path has a
  single owner.
- `load_task_execution_info(task_id)` — reads
  `task_execution_cache_dir() / f"{task_id}.json"`. Raises `RuntimeError`
  with an actionable remediation message if the file is missing.
- `verify_installed()` — optional fail-fast check that data this task
  relies on is locally available on this worker. Default: no-op. Cube
  authors override with a check appropriate to their cache. Convention:
  `TaskConfig.make()` calls `type(self).verify_installed()` at the top
  so misconfigured workers fail fast with an actionable error instead of
  timing out on a surprise download.

These helpers live on `TaskConfig` (worker-side) so workers do not need
to import the owning `BenchmarkConfig` to verify their environment or
resolve the cache path.

**`sub_bench_name`** is an optional routing tag. Standalone benchmarks leave
it None. `CompositeBenchmarkConfig.get_task_configs()` sets it to the
origin sub-benchmark's name so `CompositeBenchmark.spawn()` can route the
task back to the correct sub-benchmark's `_runtime_context`. No separate
wrapper type is needed — the emitted TaskConfig stays the sub-config's
native subclass.

**`task_id` in composites** is prefixed with the sub-benchmark's name
(`"{sub_bench_name}/{task_id}"`) so every id across the composite is
unique, while `metadata.id` keeps the native un-prefixed id.

## Invariants

1. `reset()` must call `self.tool.reset()` (implementer responsibility).
2. `step()` is concrete — do not override. All task-specific behavior goes in
   `evaluate`, `filter_actions`, `obs_postprocess`, `finished`.
3. Tool is built eagerly in `model_post_init` — once a Task is constructed, its tool is live.
4. `accept_agent_stop=True` (default) means the agent can self-terminate via
   `Action(name="final_step")`. Evaluate is called on termination.
5. `info["profiling"]` is always populated after `step()` unless no actions ran (empty list).

## Contracts for implementers

- Your `reset()` must populate the environment to the state the agent will see for the
  first observation. Include the task objective as text in that observation.
- `evaluate()` is pure — no side effects on external systems. It may read tool state.
- If you hold long-lived resources beyond the tool (containers, VMs, processes), override
  `close()` and call `super().close()`.
- `get_privileged_info()` is for debug/oracle agents and tests. Ship it for any task
  that has a known solution — empowers the harness's debug mode.

## Gotchas

- `validate_per_step=True` means `evaluate()` runs every step — expensive. Default is
  only on termination.
- STOP_ACTION is not automatically in the tool's action set — the harness / agent
  framework is responsible for including it in the action list shown to the LLM.
- `runtime_context` is a dict, not a Pydantic model — no type safety. Document keys
  in your `Benchmark._setup()` docstring.
- `model_post_init` launches the container. If your ToolConfig `make()` fails and
  you set `container_backend`, the container is already running — may leak unless the
  caller handles construction errors.
