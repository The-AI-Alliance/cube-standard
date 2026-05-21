# Tool Layer

**Module:** `cube.tool` | **Layer:** 2 (action interface)

## Purpose

Tools expose a set of actions an agent can invoke. A `ToolConfig` is the serializable
factory that produces a `Tool`. Swappable so researchers can vary tool implementations
(Playwright vs Selenium, basic vs advanced actions) without touching benchmark code.

## Public API

### `AbstractTool` (sync) / `AbstractAsyncTool` (async)
```python
class AbstractTool(ABC):
    def reset(self) -> None                                   # optional
    def close(self) -> None                                   # optional
    @abstractmethod
    def execute_action(self, action: Action) -> Any           # sync
    @property @abstractmethod
    def action_set(self) -> list[ActionSchema]
```

`AbstractAsyncTool` is identical but `reset`, `close`, `execute_action` are coroutines.

`execute_action()` returns either `Observation` (success) or `StepError` (exception).
Must not raise — catch and wrap via `StepError.from_exception()`.

`action_set` returns litellm/OpenAI-compatible function-calling descriptors (list of
`ActionSchema`). Agents discover available actions from this list without knowing the
tool implementation.

### `Tool` / `AsyncTool` (concrete bases)

Subclass these (not the abstract bases) to get automatic action discovery via the
`@tool_action` decorator. No boilerplate — method signature and docstring become the
action's schema.

```python
class MyTool(Tool):
    @tool_action
    def click(self, selector: str) -> str:
        """Click on a CSS selector."""
        ...
```

`AsyncTool.__init_subclass__` enforces that every `@tool_action` method is `async def`.
Raises `TypeError` at class-definition time otherwise.

`Tool.execute_action()` logic:
1. Resolve the method via `get_action_method(action)` — raises `ValueError` if the
   method doesn't exist OR exists but isn't decorated with `@tool_action`.
2. Validate `action.arguments` against the method's signature via
   `inspect.signature(method).bind(**arguments)`. On `TypeError` (unknown or
   missing kwargs), return an error `Observation` so the agent can correct
   itself on the next step. This is *not* a `StepError` — argument typos
   should not terminate the episode.
3. Call `method(**action.arguments)`. If result is falsy, substitute `"Success"`.
4. Wrap result in `Observation(contents=[Content.from_data(result, tool_call_id=action.id)])`.
5. On exception during method execution, log and return `StepError.from_exception(e)`.

### `@tool_action` decorator
Sets `func._is_action = True`. Action discovery walks the MRO — an override in a
subclass is still an action even without re-decoration, as long as the decorator is
present somewhere in the ancestry.

Instance-level actions (set via `setattr`) are also discovered if they have `_is_action`.

### `ToolConfig` / `AsyncToolConfig` (abstract, serializable)
```python
class ToolConfig(TypedBaseModel, ABC):
    @abstractmethod
    def make(self, container: Container | None = None) -> AbstractTool
```

The `container` argument is the container launched by the Task for this run (if any).
Use it to extract connection info (`container.get_url(port)`, `container.forward_port(port)`)
before returning the `Tool`.

`AsyncToolConfig.make()` is a coroutine.

### `Toolbox` / `AsyncToolbox`
Composite tool that holds a list of `AbstractTool` and delegates `execute_action()` to
the one owning that action name.

- Action names must be unique across tools in the toolbox — duplicates raise `ValueError`
  at construction.
- `action_set` returns the union of all contained tool action sets.
- `find_tool(cls)` returns the first contained tool that `isinstance(cls)`, else `None`.
- `reset()` / `close()` fan out to all contained tools.

`ToolboxConfig(tool_configs: list[ToolConfig])` builds a `Toolbox` from a list of configs.

## Invariants

1. `Tool.execute_action()` and `AsyncTool.execute_action()` never raise. On exception,
   return `StepError`. (Callers — i.e. `Task.step()` — depend on this.)
2. `@tool_action` on `AsyncTool` subclasses MUST be `async def`. Enforced at class
   definition.
3. Properties on Tool subclasses are never auto-discovered as actions (action_set
   skips them to avoid invoking side-effecting getters).
4. `Toolbox` rejects duplicate action names at construction — not at dispatch time.
5. `ToolConfig` subclasses are `TypedBaseModel` → must be JSON-serializable.

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
- In `AsyncTool` subclasses, a sync `@tool_action` method is a class-definition error,
  not a runtime error — you'll see the failure on import.
- `Content.from_data()` can't handle raw `bytes` from a tool's return value. Return
  strings, dicts, lists, BaseModel, PIL Image, or pre-construct a `Content` subclass.
- Action name collisions in `Toolbox` fail loudly at construction — design your tools
  with namespaced action names if you'll combine them.
