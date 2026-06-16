# Tool Layer

**Module:** `cube.tool` | **Layer:** 2 (action interface)

## Purpose

Tools expose a set of actions an agent can invoke. A `ToolConfig` is the serializable
factory that produces a `Tool`. Swappable so researchers can vary tool implementations
(Playwright vs Selenium, basic vs advanced actions) without touching benchmark code.

## Public API

### `AbstractTool`
```python
class AbstractTool(ABC):
    def reset(self) -> None                                          # optional
    def close(self) -> None                                          # optional
    @abstractmethod
    def execute_action(self, action: Action) -> Observation          # sync call-site
    async def async_execute_action(self, action: Action) -> Observation  # default: delegates to execute_action
    @property @abstractmethod
    def action_set(self) -> list[ActionSchema]
```

There is **one** tool dispatch class. Both `execute_action` (sync call-site) and
`async_execute_action` (async call-site) **always return an `Observation`** — never a raw
`StepError`, never raising for a tool error. An action exception folds into the returned
`Observation`: the error text in `contents` AND the structured `StepError` on
`Observation.error` (via `StepError.from_exception(e).to_observation()`). The default
`async_execute_action` delegates to `execute_action` on the current thread; `Tool`
overrides it for native async dispatch.

`action_set` returns litellm/OpenAI-compatible function-calling descriptors (list of
`ActionSchema`). Agents discover available actions from this list without knowing the
tool implementation.

### `Tool` (concrete base)

Subclass `Tool` to get automatic action discovery via the `@tool_action` decorator. No
boilerplate — method signature and docstring become the action's schema.

`@tool_action` methods may be **sync OR async on the same class** — dispatch routes per
the method's own `def` / `async def` keyword. The two call-sites bridge the impedance
mismatch internally:

- `tool.execute_action(action)` — sync caller. Sync action → direct call. Async action →
  bridged via a one-shot daemon thread + new event loop (~2–5 ms; `contextvars` propagate
  the caller's OTel/logging context).
- `await tool.async_execute_action(action)` — async caller. Async action → awaited
  directly. Sync action → `asyncio.to_thread` (pooled worker, enables real OS-thread
  parallelism under `asyncio.gather`). Override for tools wrapping thread-affine resources
  (sync Playwright, some DB drivers) to dispatch through a per-instance single-threaded
  executor.

```python
class MyTool(Tool):
    @tool_action
    def click(self, selector: str) -> str:
        """Click on a CSS selector."""
        ...

    @tool_action
    async def navigate(self, url: str) -> str:
        """Async action on the same class."""
        ...
```

> The `AsyncTool` / `AbstractAsyncTool` / `AsyncToolbox` dispatch classes no longer exist
> (merged into `Tool` / `AbstractTool` / `Toolbox`). The async *configs*
> (`AsyncToolConfig`, `AsyncToolboxConfig`) — whose `make()` is a coroutine — remain.

#### `Tool.final_step` — the universal STOP action
```python
@tool_action
def final_step(self) -> str:
    """Stop the task execution."""
    raise AgentStop()
```
Every `Tool` inherits `final_step` — it **is** the STOP action ([`STOP_ACTION`](../core/spec.md)
is its schema). There is no STOP special-casing anywhere and nothing appends a STOP schema:
executing it just raises `AgentStop` (a `BaseException`, so the dispatch's `except Exception`
lets it through to the caller; `Task.step()` catches it → `done=True`).

#### `Tool.execute_action()` logic
1. Resolve the method via `get_action_method(action)` — raises `ValueError` if the method
   doesn't exist OR exists but isn't decorated with `@tool_action`.
2. Validate `action.arguments` against the method's signature via
   `inspect.signature(method).bind(**arguments)`. On `TypeError` (unknown / missing kwargs),
   return a plain error `Observation` (text only, **no** `obs.error`) so the agent can
   correct itself next step — an arg typo should not terminate the episode.
3. Dispatch by the method's kind (sync direct / async bridged). If result is falsy,
   substitute `"Success"`.
4. Success → `Observation(contents=[Content.from_data(result, tool_call_id=action.id)])`.
5. Action raises → log + return `StepError.from_exception(e).to_observation()` (error text
   in `contents`, structured `StepError` on `obs.error`). `AgentStop` is a `BaseException`,
   so it propagates instead of folding.

### `@tool_action` decorator
Sets `func._is_action = True`. Action discovery walks the MRO — an override in a
subclass is still an action even without re-decoration, as long as the decorator is
present somewhere in the ancestry.

Instance-level actions (set via `setattr`) are also discovered if they have `_is_action`.

### `ToolConfig` / `AsyncToolConfig` (abstract, serializable)
```python
class ToolConfig(ValidatedConfig, ABC):
    @abstractmethod
    def make(self, container: Container | None = None) -> AbstractTool
```

The `container` argument is the container launched by the Task for this run (if any).
Use it to extract connection info (`container.get_url(port)`, `container.forward_port(port)`)
before returning the `Tool`.

`AsyncToolConfig.make()` is a coroutine — for tools whose construction needs async
resource acquisition (browser launch, network connections).

### `Toolbox(Tool)`
Composite `Tool` that holds a list of `AbstractTool` leaves and routes `execute_action` /
`async_execute_action` to the leaf owning that action name (the leaf handles its own
sync↔async bridging).

- `action_set` returns the union of all leaves' action sets, **deduped by name**.
- Identical actions across leaves (e.g. the `final_step` every `Tool` inherits) dedup to
  the first owner; same name with a **different** schema raises `ValueError` ("Conflicting
  action") at construction.
- `find_tool(cls)` returns the first contained tool that `isinstance(cls)`, else `None`.
- `reset()` / `close()` fan out to all leaves.

`ToolboxConfig(tool_configs: list[ToolConfig])` builds a `Toolbox` (sync configs);
`AsyncToolboxConfig(tool_configs: list[AsyncToolConfig])` builds one from async configs.

## Invariants

1. `execute_action()` / `async_execute_action()` always return an `Observation` and never
   raise for a tool error — the error folds into the obs (`obs.error` carries the
   `StepError`). The sole exception that propagates is `AgentStop` (a `BaseException`),
   raised by `final_step`. (Callers — `Task.step()`, `AgentView` — depend on this.)
2. `@tool_action` methods may be sync or async on the same class; dispatch routes per the
   method's `def` keyword.
3. Properties on Tool subclasses are never auto-discovered as actions (action_set
   skips them to avoid invoking side-effecting getters).
4. Every `Tool` exposes `final_step` (the STOP action). No layer special-cases STOP or
   appends a STOP schema.
5. `Toolbox` dedups identical actions and rejects same-name-different-schema conflicts at
   construction — not at dispatch time.
6. `ToolConfig` / `AsyncToolConfig` subclasses are `ValidatedConfig` (`TypedBaseModel`) →
   must be JSON-serializable.

## Contracts for implementers

- If your tool holds external resources (browser process, SSH session, DB connection),
  implement `close()`. The Task will call it on cleanup.
- If your tool has mutable state between episodes, implement `reset()`.
- Docstrings on `@tool_action` methods are part of the API — they become the action's
  description fed to the LLM.
- Action parameter names and types must be JSON-Schema-expressible (primitive types,
  `list`, `dict`, Optional). Pydantic types work via `function_to_dict`.

### New abstract tool bases capture the task-side contract only

When introducing a new abstract base for a tool family (`BrowserTool`,
`TerminalTool`, a future `ComputerTool`, …), declare only the methods that
*tasks* call for setup, validation, or observation. Do **not** decorate any
method with `@tool_action`, and do **not** enumerate an agent-facing action
surface in the abstract. The action space is a concrete-implementation
choice so a future variant (a restricted impl, an alternate transport, a
different action vocabulary) can satisfy the same abstract while exposing a
different action set to the agent. See `cube.tools.browser.BrowserTool`
and `cube.tools.terminal.TerminalTool` for worked examples.

## Packaging conventions

Where generalist tool code lives:

- **Abstract bases** (e.g. `AbstractBrowserTool`) live in `cube-standard/src/cube/tools/`
  alongside this spec.
- **Concrete implementations** live in `cube-standard/cube-tools/cube-<name>-tool/` as
  optional sub-packages **when they pull a non-trivial runtime dependency** (Playwright,
  BrowserGym, PyAutoGUI, MCP SDK, …). Otherwise — like the `TerminalTool` abstract base
  and its `ContainerTerminalTool` reference implementation — the concrete implementation
  can sit directly in `cube-standard/src/cube/tools/`, in the same module as the
  contract.
- **Tool implementations never live in `cube-harness`.** The harness consumes
  contracts and concrete implementations from this repo; it does not own tool code.
  Harness-internal infrastructure that wraps tools (e.g. telemetry decorators) is not
  itself a tool and is not in scope of this rule.
- **Cube-specific tools** (a tool whose only consumer is a single benchmark) live in
  that cube's own package, typically by subclassing a generalist tool from
  `cube-tools/`. They are not generalist tools and do not move here.

## Gotchas

- Forgetting `@tool_action` is the most common mistake. `execute_action` will raise
  `ValueError: "exists but is not decorated with @tool_action"`.
- A sync caller invoking an async `@tool_action` pays a ~2–5 ms thread+loop bridge per
  call. Use `async_execute_action` from an async call-site to avoid it.
- `Content.from_data()` can't handle raw `bytes` from a tool's return value. Return
  strings, dicts, lists, BaseModel, PIL Image, or pre-construct a `Content` subclass.
- Action name collisions in `Toolbox` fail loudly at construction only when the schemas
  *differ*; identical actions (e.g. the inherited `final_step`) dedup silently. Namespace
  your action names if you'll combine tools with genuinely distinct same-name actions.
