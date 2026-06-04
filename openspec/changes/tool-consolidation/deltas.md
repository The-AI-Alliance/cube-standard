# Deltas: Collapse `Tool` + `AsyncTool`

Applies to: `openspec/specs/tool/spec.md`.

See proposal: `proposal.md`.

---

## MODIFIED — `openspec/specs/tool/spec.md`

### `AbstractTool` — dual call surface, accepts both sync and async actions

```python
class AbstractTool(ABC):
    @property
    @abstractmethod
    def action_set(self) -> list[ActionSchema]: ...

    def execute_action(self, action: Action) -> Any:
        """Sync dispatch. Subclasses override.

        Implementations MUST raise `TypeError` if the resolved action
        method is async (point the caller at `async_execute_action`)."""

    async def async_execute_action(self, action: Action) -> Any:
        """Async dispatch — universal call-site.

        Default: call sync `execute_action` directly on the current
        thread (already shipped by cube-standard #152). Subclasses
        with async-native dispatch (e.g. `Tool` with async actions)
        override to handle both kinds without a `to_thread` hop."""
        return self.execute_action(action)
```

### `Tool` (concrete base) — sync OR async `@tool_action` methods on the same class

```python
class Tool(_ToolActionsMixin, AbstractTool):
    """Concrete base for any tool. `@tool_action` methods may be sync
    or async, per method. Action discovery via `action_set` still works
    identically; dispatch routes per the method's kind.
    """

    def execute_action(self, action: Action) -> Observation | StepError:
        method = self.get_action_method(action)
        if inspect.iscoroutinefunction(method):
            raise TypeError(
                f"Action {action.name!r} is async — call "
                f"`async_execute_action` or use an async call-site."
            )
        # sync dispatch (existing body)
        ...

    async def async_execute_action(self, action: Action) -> Observation | StepError:
        method = self.get_action_method(action)
        result = method(**action.arguments)
        if inspect.iscoroutine(result):
            result = await result
        # wrap → Observation | StepError (existing wrap logic)
        ...
```

### Deprecated aliases

```python
# Backward-compat aliases — emit DeprecationWarning when subclassed.
# Will be removed after one release window.
AbstractAsyncTool = AbstractTool
AsyncTool = Tool
```

### `_ToolActionsMixin.__init_subclass__` validation — relaxed

The current AsyncTool subclass hook validates that every `@tool_action`
method is async. With `Tool` accepting both kinds, that validation goes
away. Authors who mix sync and async actions on one class are explicitly
supported.

Class-definition-time errors that were caught before are now caught at
first-call time (sync `execute_action` raises `TypeError` naming the
action). Acceptable trade for the consolidation; error messages are
descriptive.

### Invariants

1. A `@tool_action`-decorated method is reachable through `Tool.execute_action` iff it is sync.
2. A `@tool_action`-decorated method is reachable through `Tool.async_execute_action` regardless of its sync/async kind.
3. `action_set` lists all `@tool_action`-decorated methods irrespective of kind.

### Gotchas

- `AsyncBrowserTool` (the single in-tree `AsyncTool` subclass) flips to `Tool` with no body change. After the alias is removed, downstream cubes that still subclass `AsyncTool` need a one-line edit.
- Tools that previously got an `__init_subclass__` error for mixed async/sync now silently pass; the failure mode shifts from "import error" to "TypeError on first call." Tests should exercise both call sites.

---

## REMOVED — none

`AbstractAsyncTool` and `AsyncTool` are not deleted — they become aliases of `AbstractTool` and `Tool` for one release window. Removal is a follow-up RFC after migrations land.

---

## ADDED — none beyond the dual surface above

`async_execute_action` was already added on `AbstractTool` by cube-standard
#152. This change reuses it as the universal call-site and removes the
class-level split that made the dual surface invisible.
