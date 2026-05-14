# Make `Task` generic over the tool type

**Status:** Proposed
**Date:** 2026-05-14
**Scope:** `cube.task`

## Problem

`cube.task.Task` exposes its tool through a single property typed as the
framework-level base:

```python
class Task[TTMetadata: TaskMetadata](TypedBaseModel, ABC):
    _tool: AbstractTool | None = PrivateAttr(default=None)

    @property
    def tool(self) -> AbstractTool:
        return self._tool  # type: ignore[return-value]
```

Cubes overwhelmingly know which tool surface they use (terminal, browser,
computer, …) and call surface-specific methods on it (`bash`, `goto`,
`mouse_click_xy`, …). `AbstractTool` does not expose those methods, so cubes
fall back to either:

1. **Per-method `isinstance` asserts** — type-narrowing crutch, sprinkled on
   every call site. cube-harness was carrying 10 such asserts across three
   SWE cubes before the cleanup PR.
2. **Per-cube property override** — each cube defines a typed `tool` property
   that returns `cast(FooTool, self._tool)` or `assert isinstance(self._tool,
   FooTool); return self._tool`. Required across every cube that wants a
   narrowed surface; pure boilerplate.

Neither is a runtime problem — `tool_config.make()` reliably produces the
right tool. The friction is purely static-type-checker noise.

## Solution

Add a second type parameter to `Task`:

```python
TTool = TypeVar("TTool", bound=AbstractTool, default=AbstractTool)

class Task(TypedBaseModel, Generic[TTMetadata, TTool], ABC):
    _tool: TTool | None = PrivateAttr(default=None)

    @property
    def tool(self) -> TTool:
        return self._tool  # type: ignore[return-value]
```

Cubes that parameterize `Task[FooMeta, FooTool]` get `self.tool` typed
directly as `FooTool` — no `isinstance` assert, no property override, no
`cast`.

Implementation choices:

- **`typing_extensions.TypeVar` instead of `typing.TypeVar`.** `default=` on
  TypeVar (PEP 696) is in the stdlib from Python 3.13. `cube-standard`
  supports Python 3.12, so we backport via `typing_extensions`.
  `typing-extensions` is added as a direct dep (it was already a transitive
  one).
- **Old-style `Generic[T1, T2]` instead of PEP 695 `[T1, T2]` syntax.** PEP
  695 supports default-typed parameters only from Python 3.13. Old-style
  Generic plus `typing_extensions.TypeVar` keeps the change non-breaking on
  3.12.
- **Inheritance order: `TypedBaseModel` before `Generic[…]`.** Pydantic emits
  `GenericBeforeBaseModelWarning` if `Generic[…]` comes first. Tested: no
  warnings with `TypedBaseModel, Generic[TTMetadata, TTool], ABC`.

## Backwards compatibility

The change is **non-breaking** thanks to the default value on `TTool`:

| Existing cube | Resolves to | Status |
|---|---|---|
| `class FooTask(Task):` | `Task[TaskMetadata, AbstractTool]` | works |
| `class FooTask(Task[FooMeta]):` | `Task[FooMeta, AbstractTool]` | works |
| `class FooTask(Task[FooMeta, FooTool]):` (new) | `Task[FooMeta, FooTool]` | opt-in |

No cube needs to change. The 10+ existing cubes that use `Task[Meta]` keep
compiling and behaving identically. Cubes that want a narrowed `self.tool`
opt in by adding the second parameter.

## Migration

**This PR (cube-standard):**

- `src/cube/task.py`: switch `Task` from PEP 695 syntax to old-style
  `Generic[TTMetadata, TTool]`; add `TTMetadata` and `TTool` `TypeVar`
  declarations; narrow `_tool` and `tool` to `TTool`. Net: ~15 lines.
- `pyproject.toml`: promote `typing-extensions` to a direct dep.
- `openspec/specs/task/spec.md`: update the `Task` class signature block;
  add a paragraph explaining the new `TTool` parameter.

**Companion PR (cube-harness, supersedes #392's current implementation):**

- The three SWE cubes (`swebench-verified-cube`, `swebench-live-cube`,
  `terminalbench2-cube`) switch from `class FooTask(Task[FooMeta]):` plus a
  `tool` property override + `cast(TerminalTool, self._tool)` to:

  ```python
  class FooTask(Task[FooMeta, TerminalTool]):
      ...  # no property override, no cast, no type: ignore
  ```

  Drops `from typing import cast`, drops the property method, drops the
  `# type: ignore[override]` comment. Net per cube: smaller diff than #392
  currently has.

**Other cubes:** unchanged. They keep using `Task[Meta]` (default
`AbstractTool`) until/unless they choose to narrow.

**Templates and examples (`_template/`, `examples/counter-cube`):** show the
new optional second parameter as a tip in the docstring; the default form
remains the recommended shape for cubes that don't have a strong tool-type
binding.

See [deltas.md](deltas.md) for the spec contract change.
