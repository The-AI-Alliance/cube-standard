# Task Layer

**Module:** `cube.task` | **Layer:** 3 (per-task environment dynamics + evaluation)

## Purpose

A `Task` is a single scoreable problem — **the Task is the world**. It unifies gym-style
dynamics (`reset`/`step`/`close`) with task-specific logic (`evaluate`, `_filter_actions`,
`obs_postprocess`). One class, one place for the benchmark author to define both what the
agent can do and how it is scored.

An agent never holds the Task. It holds an **`AgentView`** — the obs-in/action-out
agent-facing facet — obtained via `task.get_agent_view(role)`. The Task supports
single-agent (the default) and multi-agent benchmarks through the same `role` thread.

**Two-audience `_` convention:** override points meant for *benchmark developers* only
(not downstream harness users) are `_`-prefixed (`_make_tool`, `_filter_actions`); the
public surface (`reset`, `evaluate`, `obs_postprocess`, `get_agent_view`, `agent_roles`)
is not.

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

Per-task **clarifications** for brittle tasks (whose original wording omits a
step a reasonable LLM would not infer) are not stored on `TaskMetadata`. They
live in the benchmark's prompt overlay — a `{task_id: text}` dict loaded by
`BenchmarkConfig.load_benchmark_clarifications()` (see
[benchmark/spec.md](../benchmark/spec.md)). This keeps the original benchmark
wording intact and lets one clarification be reused across many tasks.

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
`self.load_task_execution_info()` against the subclass. The framework
never reads `execution_info` itself; it is for cube authors to surface
domain-specific heavy data with autocomplete and Pydantic validation.

Cubes with no heavy data leave the slot `None`; the base class is
instantiable but carries no fields.

### `Task` (abstract, Pydantic)
```python
class Task(TypedBaseModel, Generic[TTMetadata, TTool], ABC):
    # Serializable fields — SerializeAsAny preserves subclass-specific fields
    # through JSON round-trip (Pydantic would otherwise strip them to the
    # declared base type). Required for every polymorphic field.
    metadata: SerializeAsAny[TTMetadata]
    execution_info: SerializeAsAny[TaskExecutionInfo] | None = None  # heavy lazy data; populated on the worker
    tool_config: SerializeAsAny[ToolConfig]
    runtime_context: RuntimeContext | None = None      # from Benchmark._setup()
    validate_per_step: bool = False                    # eval on every step, not just done

    # Runtime (PrivateAttr, set in model_post_init)
    _tool: TTool | None                         # the task's own no-role tool
    _container: Container | None
    _resource_handle: ResourceHandle | None    # handle from InfraConfig.launch(); torn down by close()

    @property
    def tool(self) -> TTool: ...                # the no-role tool; raises if absent (see below)
```

`Task` carries two type parameters:

- `TTMetadata` (bound `TaskMetadata`) narrows `self.metadata` so cubes don't
  re-annotate the field on every access.
- `TTool` (bound `AbstractTool`, default `AbstractTool`) narrows `self.tool`
  to a specific tool surface. Cubes that bind it (e.g.
  `Task[FooMeta, TerminalTool]`) drop `isinstance(self.tool, FooTool)`
  asserts and per-cube `tool` property overrides — `self.tool` is the right
  type by construction.

Defaults make `TTool` non-breaking: `Task[Meta]` resolves to
`Task[Meta, AbstractTool]`. `typing_extensions.TypeVar` is used to back
default-on-TypeVar (PEP 696) to Python 3.12; the stdlib `TypeVar` gained
the feature in 3.13.

`execution_info` is the typed surface for heavy per-task data. Cubes
populate it inside `TaskConfig.make()`: validate
`self.load_task_execution_info()` against the cube's `TaskExecutionInfo`
subclass and pass to the `Task` constructor.

Tasks read typed fields directly: `self.execution_info.problem_statement`,
`self.execution_info.patch`, …

`model_post_init` (runs after Pydantic `__init__`):
1. If `metadata.container_config` is set and `runtime_context["infra"]` is
   present, the container is provisioned via the injected `InfraConfig`
   (`cube.task_infra.launch_task_container`); the live `Container` and its
   `ResourceHandle` are stored in `_container` / `_resource_handle`.
2. `self._tool = self._make_tool()` — build the task's own no-role tool.

**Abstract methods (implementers MUST provide):**
- `reset() -> (Observation, dict)` — set up initial state; also call `self.tool.reset()`
- `evaluate(obs: Observation | None = None) -> (float, dict)` — score the current state

**Tool lifecycle / multi-agent (benchmark-dev override points):**
- `_make_tool(role: str | None = None) -> TTool` — the **single** tool-lifecycle hook
  (replaces `make_tool` / `prepare_world` / `_build_tool`). Does any once-per-task world
  prep AND builds the tool. Default: `tool_config.make(container=self._container)`, ignoring
  `role`. Called once for the no-role `_tool` and once per seat by `get_agent_view`; each
  call returns a fresh session. A *multi-agent* task whose tools are strictly per-role may
  raise `NotImplementedError` for `role=None` (honored only for multi-agent — `_tool` stays
  `None` and `tool` raises; for a single-agent task it propagates as a real bug).
- `agent_roles() -> dict[str | None, int]` — the roster, role → seat count. Default
  `{None: 1}` (single-agent). Multi-agent cubes override, e.g. `{"buyer": 2, "seller": 1}`.
- `get_agent_view(role: str | None = None) -> AgentView` — the agent-facing view. Base
  implements **only** the single-agent case (`role=None`, over the task's own tool) and
  **raises `NotImplementedError`** for a named role. There is **no `seat` param**:
  multi-agent benchmarks **override** this and own the per-role seat index internally (the
  runtime calls it once per seat declared in `agent_roles()`).

**Optional hooks (default: identity / no-op):**
- `_filter_actions(actions: list[ActionSchema], role: str | None = None) -> list[ActionSchema]`
  — advisory whitelist/mask of *advertised* actions. Applied in BOTH `Task.action_set` and
  `AgentView.action_set` so the gym and agent paths never diverge. Recomputed each access
  (may vary across an episode from task state). Advisory — shapes what the agent *sees*, not
  execute-time enforcement.
- `obs_postprocess(obs: Observation, role: str | None = None) -> Observation` — per-seat
  observation post-processing (the twin of `_filter_actions`; `role` threads through both).
- `finished(obs: Observation | None = None) -> bool` — early termination check
- `get_privileged_info() -> Content` — solution, eval source, internal state (for debug/oracle agents)
- `get_status() -> str` — free-form status string
- `close()` — cleanup; default calls `self.tool.close()`. Override to add cleanup and call `super().close()`.

`role` belongs on exactly the two per-seat view-shaping hooks (`_filter_actions`,
`obs_postprocess`), never on the world-global `evaluate` / `reset` / `finished`.

### `tool` property + `AgentView` (two distinct surfaces)

```python
@property
def tool(self) -> TTool: ...                # the task's own no-role tool
def get_agent_view(self, role=None) -> AgentView
```

- **`Task.tool`** is the task's own no-role tool — what the Task itself drives (reset /
  evaluate / setup) and what cube-standard internals (server, nemogym, debug suite) use.
  It is the raw environment tool, **not** an agent surface. Raises `RuntimeError` if the
  task has no no-role tool (a multi-agent task whose `_make_tool(None)` raised
  `NotImplementedError`) — drive each seat via `get_agent_view(role)` instead.
- **`AgentView`** is the ONLY surface an agent holds (obs in, action out, no lifecycle) —
  see below.

### `AgentView`
```python
class AgentView:                                  # a facet of a Task, NOT a Tool
    role: str | None                              # None for single-agent
    seat: int
    @property def agent_id(self) -> str           # "agent" (single) else "{role}-{seat}"
    @property def action_set(self) -> list[ActionSchema]    # tool actions after _filter_actions(role)
    def execute_action(self, action) -> Observation         # obs only — no reward, no done
    async def async_execute_action(self, action) -> Observation
    def set_eval_callback(self, cb: Callable[[float, dict], None]) -> None
```

The agent gets exactly `action_set` (what it may do now) and `execute_action` (do one
thing, see the obs) — never `reset`/`evaluate`/`close`/`step`. `execute_action` relays to
the same per-action core as gym `step`, applies `obs_postprocess(role)`, and returns the
**observation only**. Dispatch goes through the seat's own tool, so "which agent acted" is
implicit in which session ran the action.

**Per-step eval:** when `task.validate_per_step` is set, `execute_action` fires the
per-step `evaluate` and surfaces `(reward, info)` through the registered eval callback —
out-of-band, never in the returned obs (reward is not the agent's concern). A
`validate_per_step` task with **no** callback registered is a wiring bug, so it **raises**
loudly rather than silently dropping the reward. `agent_id` is `"agent"` for the single
default seat, else `"{role}-{seat}"` (e.g. `"buyer-0"`).

### `Task.step()` (concrete; do not override)

Signature: `step(action: Action | list[Action]) -> EnvironmentOutput`

The **gym-compatibility view**. Accepts single Action or list (sequential multi-action
step). Logic:

1. Loop over actions, each through `self._tool.execute_action(action)` (the same dispatch
   `AgentView` uses), timed:
   - `except AgentStop`: `obs += stop.observation`, `done=True`, break.
   - Else `obs += result`; if `result.error is not None`, lift it onto `error` (the
     `StepError` returned via `EnvironmentOutput.error`). A tool error is non-terminal.
2. `done = done or self.finished(obs)`
3. If `done` or `self.validate_per_step`: call `self.evaluate(obs)` → `(reward, info)`
4. Apply `obs = self.obs_postprocess(obs)`
5. Populate `info["profiling"]` with `tool_execute`, `evaluate`, `obs_postprocess` timings

Returns `EnvironmentOutput(obs, reward, done, info, error)`. `truncated` is always
`False` — step/time-limit truncation is the harness's responsibility (TODO in code).

`STOP_ACTION` (the `final_step` schema) and `AgentStop` live in [`cube.core`](../core/spec.md)
(re-exported from `cube.task` for back-compat). STOP is a real tool action
(`Tool.final_step`) that raises `AgentStop` — there is **no STOP special-casing** in this
layer anymore.

### `Task.action_set` (concrete property)

```python
@property
def action_set(self) -> list[ActionSchema]:
    return self._filter_actions(self.tool.action_set)
```

The gym-view action set — the task's own tool's actions after `_filter_actions` (role=None).
Already includes `final_step` (every Tool exposes it; nothing appends a STOP schema).
Mirrors what an `AgentView` advertises so the gym and agent paths never diverge.

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
class TaskConfig[TTMetadata: TaskMetadata](TypedBaseModel, ABC):
    metadata: SerializeAsAny[TTMetadata]         # travels with the config
    seed: int | None = None
    tool_config: SerializeAsAny[ToolConfig] | None = None
    sub_bench_name: str | None = None            # composite routing hint (see below)

    @property
    def task_id(self) -> str:
        """Derived: ``"{sub_bench_name}/{metadata.id}"`` for composites, else ``metadata.id``."""

    @abstractmethod
    def make(
        self,
        runtime_context: RuntimeContext | None = None,
    ) -> Task

    # ClassVar back-stamped by BenchmarkConfig.__init_subclass__ to
    # cls.cache_dir() so the default task-execution cache lives directly
    # under the benchmark's cache dir. None for TaskConfig subclasses
    # constructed without an owning BenchmarkConfig.
    _benchmark_cache_dir: ClassVar[Path | None] = None

    @classmethod
    def task_execution_cache_dir(cls) -> Path:
        """Default: ``BenchmarkConfig.cache_dir() / "tasks_execution_info"``,
        falling back to ``~/.cube/<top-level-package-name>/tasks_execution_info/``
        when ``_benchmark_cache_dir`` is not set."""

    def load_task_execution_info(self) -> dict[str, Any]:
        """Read processed per-task data for ``self.task_id`` from the cache."""

    def verify_installed(self) -> None:
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
`self.load_task_execution_info()` (read from the per-task on-disk cache)
and validating the resulting dict against the `TaskExecutionInfo`
subclass. The cache itself is written by `BenchmarkConfig.install()` —
operators run `cube install <bench>` once per worker environment.

**Per-task cache helpers (worker-side).**
- `task_execution_cache_dir()` (classmethod) — default
  `BenchmarkConfig.cache_dir() / "tasks_execution_info"`, where the
  cache dir is back-stamped by `BenchmarkConfig.__init_subclass__` onto
  the owning `task_config_class` via `_benchmark_cache_dir`. Falls back
  to `~/.cube/<top-level-package-name>/tasks_execution_info/` when no
  owner is attached — relevant for direct test instantiation.
  Override on subclasses that use a non-default cache layout (e.g. cubes
  that co-locate the cache with other on-disk state).
  `BenchmarkConfig.install()` writes via
  `cls.task_config_class.task_execution_cache_dir()` so the path has a
  single owner.
- `load_task_execution_info()` (instance method) — reads
  `type(self).task_execution_cache_dir() / f"{self.task_id}.json"`.
  Raises `RuntimeError` with an actionable remediation message if the
  file is missing.
- `verify_installed()` (instance method) — optional fail-fast check that
  data this task relies on is locally available on this worker. Default:
  no-op. Cube authors override with a check appropriate to their cache.
  Convention: `TaskConfig.make()` calls `self.verify_installed()` at the
  top so misconfigured workers fail fast with an actionable error
  instead of timing out on a surprise download.

These helpers live on `TaskConfig` (worker-side) so workers do not need
to import the owning `BenchmarkConfig` to verify their environment or
resolve the cache path.

**`sub_bench_name`** is an optional routing tag. Standalone benchmarks leave
it `None`. `CompositeBenchmarkConfig.get_task_configs()` sets it to the
origin sub-benchmark's name. For **nested composites** (composites of
composites) each outer layer prepends its sub-benchmark name, producing a
`"/"`-joined path (e.g. `"inner-suite/bench-a"`). `CompositeBenchmark.spawn()`
peels the path one hop at a time, delegating to inner composites until the
leaf sub-benchmark is reached. No separate wrapper type is needed — the
emitted `TaskConfig` stays the sub-config's native subclass.

**`task_id`** is a derived `@property` (not a serialized field). For
standalone tasks it returns `metadata.id`; for composite tasks it returns
`"{sub_bench_name}/{metadata.id}"`, which for nested composites produces a
fully-qualified path (e.g. `"inner-suite/bench-a/task-1"`). `metadata.id`
always retains the native un-prefixed id.

## Invariants

1. `reset()` must call `self.tool.reset()` (implementer responsibility).
2. `step()` is concrete — do not override. All task-specific behavior goes in
   `evaluate`, `_filter_actions`, `obs_postprocess`, `finished`.
3. Tool is built eagerly in `model_post_init` (`self._tool = self._make_tool()`) — once a
   single-agent Task is constructed, its tool is live.
4. The agent self-terminates via `Action(name="final_step")` — a real tool action that
   raises `AgentStop`, caught by `step()` (→ `done=True`, then `evaluate()`). There is no
   `accept_agent_stop` flag and no STOP schema injection: `final_step` is always present.
5. `_filter_actions` and `obs_postprocess` are applied identically on the gym path
   (`Task.action_set` / `step`) and the agent path (`AgentView`) — they never diverge.
6. `info["profiling"]` is always populated after `step()` unless no actions ran (empty list).

## Contracts for implementers

- Your `reset()` must populate the environment to the state the agent will see for the
  first observation. Include the task objective as text in that observation.
- `evaluate()` is pure — no side effects on external systems. It may read tool state.
- If you hold long-lived resources beyond the tool (containers, VMs, processes), override
  `close()` and call `super().close()`.
- `get_privileged_info()` is for debug/oracle agents and tests. Ship it for any task
  that has a known solution — empowers the harness's debug mode.

## Gotchas

- Polymorphic fields (`metadata`, `execution_info`, `tool_config`) use
  `SerializeAsAny[Base]` rather than the bare base type. Without it Pydantic
  serializes only the base-class fields and silently drops subclass-specific
  state on every JSON round-trip. This is already in the base classes — cube
  authors don't need to repeat it on their subclasses.
- Cubes that need narrower static types on `Task.metadata` use the
  parametrised form `class FooTask(Task[FooTaskMetadata]):` rather than
  re-annotating `metadata: FooTaskMetadata` on the subclass. Re-annotation
  is unsound under invariant-field semantics and type checkers reject it;
  the parametrised form expresses the intent correctly without an override.
  Pairs naturally with `class FooTaskConfig(TaskConfig[FooTaskMetadata]):`.
- `task_execution_cache_dir()` lives directly under `BenchmarkConfig.cache_dir()`
  (back-stamped onto the `TaskConfig` subclass at class-definition time by
  `BenchmarkConfig.__init_subclass__` via `_benchmark_cache_dir`), so cubes
  that override `cache_dir()` (e.g. to co-locate with VM data) get the
  override applied to the per-task cache too without an extra override. The
  fallback to `~/.cube/<top-level-package-name>/` only kicks in for
  `TaskConfig` subclasses that have no owning `BenchmarkConfig` (direct test
  instantiation).
- `validate_per_step=True` means `evaluate()` runs every step — expensive. Default is
  only on termination. On the agent path it requires `AgentView.set_eval_callback()` to be
  wired, else `execute_action` raises (silent reward-drop is a bug).
- `_filter_actions` must NOT append `STOP_ACTION` — `final_step` is always advertised by
  the tool. Per-role action *sets* are better expressed by `_make_tool(role)` returning a
  role-bound tool; use `_filter_actions` for task-state-dependent masking the tool can't see.
- A multi-agent task that overrides `get_agent_view` owns the per-seat index internally;
  the runtime never passes a seat. If its tools are strictly per-role, `_make_tool(None)`
  may raise `NotImplementedError` and `Task.tool` then raises — drive seats via
  `get_agent_view(role)`.
- `runtime_context` is a dict, not a Pydantic model — no type safety. Document keys
  in your `Benchmark._setup()` docstring.
- `model_post_init` launches the container (via the injected `InfraConfig`)
  before building the tool. If your `ToolConfig.make()` / `_make_tool()` fails,
  the container is already running — it may leak unless the caller handles
  construction errors (e.g. wraps `spawn()`/`make()` and calls
  `_resource_handle.close()` on failure).
