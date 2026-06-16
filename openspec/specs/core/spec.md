# Core Types

**Module:** `cube.core` | **Layer:** 1 (shared types, no implementation)

## Purpose

Fundamental data types exchanged between Tool, Task, and harness. All serializable
via Pydantic. No business logic.

## Public API

### `TypedBaseModel`
Pydantic base that round-trips polymorphic subclasses. Serializes with a `_type` field
(fully qualified class name). Deserialization reinstantiates the correct concrete class.
Refuses deserialization of abstract classes directly.

All CUBE serializable configs subclass this. Required for any field typed as an abstract
base class that holds a concrete subclass value (e.g., `tool_config: ToolConfig` holding
a `BrowserToolConfig`).

### `ValidatedConfig`
`TypedBaseModel` + `ConfigDict(validate_assignment=True)`: bad attribute assignment
raises at the assignment site, not later in a worker. Subclassed by the user-mutable
config ABCs (`ToolConfig`, `AsyncToolConfig`, `InfraConfig`, `BenchmarkConfig`); other
`TypedBaseModel` types keep construction-only validation. `model_copy(update=...)`
bypasses it (so subsetting helpers are unaffected); `_type` round-trip preserved.

### `ConfigRegistry[T: BaseModel]`
Read-only `Mapping[str, T]` for named-config catalogs (canonical agent/benchmark/infra
configs). Every `reg[name]` returns a fresh `model_copy(deep=True)` so callers can't
mutate the shared instance; unknown name → `KeyError` listing available names. A
`Mapping`, not a `dict` subclass, so no accessor bypasses the copy.

### `Action`
```python
class Action(TypedBaseModel):
    id: str | None = None        # tool_call_id from the LLM
    name: str                    # action name (method on Tool)
    arguments: dict[str, Any] = {}
```
Built from OpenAI tool calls via `Action.from_openai_tool_call(dict)`. Supports both
Chat Completions (`{"function": {...}}`) and Responses API (flat) formats.

### `ActionSchema`
```python
class ActionSchema(TypedBaseModel):
    name: str                    # non-empty
    description: str             # non-empty
    parameters: dict = {}        # JSON Schema
```
Describes one callable action. Compatible with litellm/OpenAI function-calling format
via `as_dict()`. Built from a Python function via `ActionSchema.from_function(func)`.
`validate_param_descriptions()` enforces non-empty descriptions on every parameter
except `self`.

### `Content` (abstract)
```python
class Content(TypedBaseModel, ABC):
    tool_call_id: str | None     # set if this is a tool-call result
    name: str | None             # optional label
    data: Any                    # narrowed per subclass

    @abstractmethod def to_markdown(self) -> str
    @abstractmethod def to_llm_message(self) -> dict  # OpenAI/litellm format
```

**Concrete subclasses:**
- `TextContent(data: str)` — plain text; coerces int/float to str
- `StructuredContent(data: dict | list | BaseModel)` — JSON code block
- `ImageContent(data: PILImage.Image)` — base64 PNG; round-trips via `data:image/png;base64,` prefix
- `AudioContent(data: bytes, duration_seconds: float | None)` — placeholder; `to_*()` raise `NotImplementedError`
- `VideoContent(data: bytes, duration_seconds: float | None)` — placeholder; same

**Dispatch helper:** `Content.from_data(data, **kwargs)` auto-selects subclass from type.
Rejects raw `bytes` (audio vs video ambiguous — construct explicitly).

### `Observation`
```python
class Observation(TypedBaseModel):
    contents: list[Content] = []
    error: StepError | None = None    # set when this obs reports a failed action

    @classmethod def from_text(cls, text: str) -> Observation
    def to_llm_messages(self) -> list[dict]          # one per content
    def to_markdown(self) -> str                      # joined with \n\n
    def __add__(self, other: Observation) -> Observation  # appends contents (NOT error)
```

A failed action is **non-terminal**: the error text is in `contents` (the agent reads it
and retries), and the structured `StepError` is also attached on `error` (machine-readable
copy for telemetry / `EnvironmentOutput.error`). `__add__` merges only `contents`.

### `StepError`
```python
class StepError(TypedBaseModel):
    error_type: str
    exception_str: str
    stack_trace: str

    @classmethod def from_exception(cls, exc: Exception) -> StepError
    def to_observation(self) -> Observation    # error text in contents + self on .error
```
Built (never raised) from an action exception inside `Tool.execute_action()` /
`async_execute_action()`, which always return an `Observation`: the error folds in via
`StepError.from_exception(e).to_observation()`. `Task.step()` lifts `obs.error` onto
`EnvironmentOutput.error`.

### `AgentStop` (exception)
```python
class AgentStop(BaseException):
    observation: Observation    # terminal obs; default "Task finished by the agent."
```
Raised by the STOP action (`Tool.final_step`) to end the episode. A `BaseException` (not
`Exception`) so a tool's / agent's `except Exception` never swallows it. The gym
`Task.step()` catches it (→ `done=True`); the agent-facing path lets it propagate.

### `STOP_ACTION` (module-level constant)
```python
STOP_ACTION = ActionSchema(
    name="final_step",
    description="Stop the task execution.",
    parameters={"type": "object", "properties": {}},
)
```
The schema of `Tool.final_step` — the universal STOP action every tool exposes. There is
no STOP special-casing anywhere: executing it just raises `AgentStop`. The empty-but-typed
`parameters` is the minimal payload Anthropic accepts for `input_schema`; LiteLLM passes
it through verbatim.

### `EnvironmentOutput`
```python
class EnvironmentOutput(TypedBaseModel):
    obs: Observation
    reward: float = 0.0
    done: bool = False
    truncated: bool = False        # time/step-limit termination
    info: dict = {}                # always includes "profiling" key post-step
    error: StepError | None = None # the step's per-action StepError (lifted from obs.error)
```
Follows Gymnasium API conventions. `done=True` means terminated; `truncated=True` means
cut short by a harness-imposed limit.

## Invariants

1. `TypedBaseModel` subclasses must be importable at deserialization time (the `_type`
   field is a fully qualified import path).
2. `ActionSchema.name` and `.description` are non-empty (Pydantic-enforced).
3. `Content.from_data()` raises `TypeError` on raw bytes — callers must construct
   `AudioContent` or `VideoContent` explicitly.
4. `ImageContent` deserialization calls `img.load()` to prevent `BytesIO` GC issues
   (PIL is lazy by default).
5. `Observation + Observation` mutates the left operand in-place (appends `contents` only,
   never `error`); this is intentional for accumulation in `Task.step()`.
6. A failed action is non-terminal — it folds into the returned `Observation`
   (`StepError.to_observation()`), never raised. Only `AgentStop` (a `BaseException`)
   ends an episode.

## Gotchas

- `TypedBaseModel`'s `_type` path uses `__module__.__name__` — renaming a class or
  moving a module breaks existing serialized data. Plan migrations explicitly.
- `AudioContent` / `VideoContent` are placeholders. Calling `to_markdown()` or
  `to_llm_message()` raises `NotImplementedError`. Rendering strategy TBD.
- Abstract `Content` subclass raises on direct deserialization — always serialize
  from a concrete subclass so `_type` is set.
