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
    task_clarification: str | None = None      # optional opt-in clarification (see below)
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

`task_clarification` is an optional short string a harness may append to
the task objective **only when** the owning
`BenchmarkConfig.add_task_clarification` is `True` (see
[benchmark/spec.md](../benchmark/spec.md)). Populated over time for
brittle tasks whose original wording omits a step a reasonable LLM would
not infer — for instance a miniwob task whose objective reads "set slider
to 32 and string value to 'foo'" but whose verifier only rewards if the
agent then clicks submit. Storing the fix as metadata keeps the original
benchmark wording intact and lets evaluations toggle the clarifications
on or off without forking the benchmark. Leave `None` for tasks whose
wording is unambiguous. The framework only declares the slot — actual
prompt assembly happens in the harness.

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
    accept_agent_stop: bool = True                     # accept STOP_ACTION from agent

    # Runtime (PrivateAttr, set in model_post_init)
    _tool: TTool | None
    _container: Container | None
    _resource_handle: ResourceHandle | None    # handle from InfraConfig.launch(); torn down by close()

    @property
    def tool(self) -> TTool: ...
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
2. Call `_build_tool()` (default: `tool_config.make(container=self._container)`)
   to build the tool.

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
STOP_ACTION = ActionSchema(
    name="final_step",
    description="Stop the task execution.",
    parameters={"type": "object", "properties": {}},
)
```
Protocol for agent-initiated termination. Tasks that reject it must set
`accept_agent_stop = False` (e.g., tasks that require the agent to reach a terminal
state via environment interaction, not a self-declaration).

`Task.action_set` auto-appends `STOP_ACTION` whenever `accept_agent_stop=True`
(deduping by name), so cube authors do not re-add it in `filter_actions`. The
non-empty `parameters` schema is the minimal payload Anthropic accepts for
`input_schema`; LiteLLM passes it through verbatim.

### `Task.action_set` (concrete property)

```python
@property
def action_set(self) -> list[ActionSchema]:
    actions = self.filter_actions(self.tool.action_set)
    if self.accept_agent_stop and not any(a.name == STOP_ACTION.name for a in actions):
        actions = [*actions, STOP_ACTION]
    return actions
```

Returns the action schemas the agent should see. Runs `filter_actions()`
first, then appends `STOP_ACTION` when `accept_agent_stop=True` unless an
action with that name is already present. The dedup branch keeps legacy
`filter_actions` overrides that still append `STOP_ACTION` working.

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
   `evaluate`, `filter_actions`, `obs_postprocess`, `finished`.
3. Tool is built eagerly in `model_post_init` — once a Task is constructed, its tool is live.
4. `accept_agent_stop=True` (default) means the agent can self-terminate via
   `Action(name="final_step")`. `Task.action_set` auto-appends `STOP_ACTION`
   in this case (dedup by name); `step()` recognises it and calls
   `evaluate()` on termination.
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
  only on termination.
- `filter_actions` overrides MUST NOT append `STOP_ACTION` — `Task.action_set`
  does it automatically when `accept_agent_stop=True`. Manual appends still
  work (dedup by name) but are redundant and should be removed.
- `runtime_context` is a dict, not a Pydantic model — no type safety. Document keys
  in your `Benchmark._setup()` docstring.
- `model_post_init` launches the container (via the injected `InfraConfig`)
  before building the tool. If your `ToolConfig.make()` / `_build_tool()` fails,
  the container is already running — it may leak unless the caller handles
  construction errors (e.g. wraps `spawn()`/`make()` and calls
  `_resource_handle.close()` on failure).
