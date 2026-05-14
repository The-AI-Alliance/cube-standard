# Deltas — `Task` generic over tool type

**Targets:** `openspec/specs/task/spec.md`

## MODIFIED — `Task` class signature

**Spec:** task

The class signature block:

```python
class Task[TTMetadata: TaskMetadata](TypedBaseModel, ABC):
    ...
    _tool: AbstractTool | None
```

becomes:

```python
class Task(TypedBaseModel, Generic[TTMetadata, TTool], ABC):
    ...
    _tool: TTool | None
```

The change adds a second type parameter `TTool` (bound `AbstractTool`,
default `AbstractTool`) and switches the class declaration from PEP 695
syntax to old-style `Generic[…]`. The switch is required because PEP 695
supports default-typed parameters only from Python 3.13; cube-standard
supports 3.12 and backports defaults via `typing_extensions.TypeVar`.

Inheritance order puts `TypedBaseModel` before `Generic[…]` to avoid
Pydantic's `GenericBeforeBaseModelWarning`.

## MODIFIED — `Task.tool` return type

**Spec:** task

```python
@property
def tool(self) -> TTool: ...
```

The return type is `TTool` instead of `AbstractTool`. `Task[Meta]` resolves
to `Task[Meta, AbstractTool]` via the default, so the previously documented
behavior (return `AbstractTool`) is preserved for unparameterized cubes.
Cubes that parameterize `Task[Meta, FooTool]` see `self.tool: FooTool` and
can drop per-cube property overrides and `isinstance` asserts.

## ADDED — `TTool` type parameter convention

**Spec:** task

New section in the `Task` documentation:

> `TTool` (bound `AbstractTool`, default `AbstractTool`) narrows `self.tool`
> to a specific tool surface. Cubes that bind it (e.g. `Task[FooMeta,
> TerminalTool]`) drop `isinstance(self.tool, FooTool)` asserts and per-cube
> `tool` property overrides — `self.tool` is the right type by construction.
> Defaults make `TTool` non-breaking: `Task[Meta]` is equivalent to
> `Task[Meta, AbstractTool]`.

## Not changed

- `TaskConfig` remains generic over `TTMetadata` only. `TaskConfig.make()`
  still returns `Task[TTMetadata]` (i.e. `Task[TTMetadata, AbstractTool]`
  via default). Narrowing `TaskConfig` to be aware of the tool type is a
  future change; it is not required for the cube-side ergonomic win.
- Existing cube-side fields, methods, invariants, and gotchas are unchanged.
- The serialization boundary (`TaskConfig` is the pickled serializable; live
  `Task` objects do not cross processes) is unchanged.
