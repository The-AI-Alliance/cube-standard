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
    extra_info: dict[str, Any] = {}
```

The actual task objective is surfaced in the first `Observation` returned by `reset()`.
`abstract_description` is for tooling (search, subsetting), never to be shown to the agent.

### `Task` (abstract, Pydantic)
```python
class Task(TypedBaseModel, ABC):
    # Serializable fields
    metadata: TaskMetadata
    tool_config: ToolConfig
    container_backend: ContainerBackend | None = None
    runtime_context: RuntimeContext | None = None      # from Benchmark._setup()
    validate_per_step: bool = False                    # eval on every step, not just done
    accept_agent_stop: bool = True                     # accept STOP_ACTION from agent

    # Runtime (PrivateAttr, set in model_post_init)
    _tool: AbstractTool | None
    _container: Container | None
```

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
    seed: int | None = None
    tool_config: ToolConfig | None = None

    @abstractmethod
    def make(
        self,
        runtime_context: RuntimeContext | None = None,
        container_backend: ContainerBackend | None = None,
    ) -> Task
```

Minimal JSON-serializable handle that workers receive. `make()` runs on the worker,
looks up `TaskMetadata` via `task_id` from the owning `BenchmarkConfig`'s
class-level `task_metadata` dict, potentially merges in heavy fields from
`BenchmarkConfig.load_task_execution_info(task_id)`, and constructs the Task.

Class-level lookup is stable across subsetting: `BenchmarkConfig.subset_from_*`
narrows the view via the instance-level `task_ids` field without touching the
ClassVar. Workers can therefore deserialize a `TaskConfig` in isolation and
always find a valid `TaskMetadata` entry by id.

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
