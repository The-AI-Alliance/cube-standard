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

    @classmethod def from_text(cls, text: str) -> Observation
    def to_llm_messages(self) -> list[dict]          # one per content
    def to_markdown(self) -> str                      # joined with \n\n
    def __add__(self, other: Observation) -> Observation  # appends contents
```

### `StepError`
```python
class StepError(TypedBaseModel):
    error_type: str
    exception_str: str
    stack_trace: str

    @classmethod def from_exception(cls, exc: Exception) -> StepError
```
Returned (not raised) from `Tool.execute_action()` when the action itself raises.
Caught by `Task.step()` and embedded in `EnvironmentOutput.error`.

### `EnvironmentOutput`
```python
class EnvironmentOutput(TypedBaseModel):
    obs: Observation
    reward: float = 0.0
    done: bool = False
    truncated: bool = False        # time/step-limit termination
    info: dict = {}                # always includes "profiling" key post-step
    error: StepError | None = None
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
5. `Observation + Observation` mutates the left operand in-place (appends contents);
   this is intentional for accumulation in `Task.step()`.

## Gotchas

- `TypedBaseModel`'s `_type` path uses `__module__.__name__` — renaming a class or
  moving a module breaks existing serialized data. Plan migrations explicitly.
- `AudioContent` / `VideoContent` are placeholders. Calling `to_markdown()` or
  `to_llm_message()` raises `NotImplementedError`. Rendering strategy TBD.
- Abstract `Content` subclass raises on direct deserialization — always serialize
  from a concrete subclass so `_type` is set.
