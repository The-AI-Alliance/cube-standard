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
    identically; dispatch routes per the method's kind and bridges the
    impedance mismatch when caller and method differ.
    """

    def execute_action(self, action: Action) -> Observation | StepError:
        """Sync call-site. Works for both sync and async actions.

        Sync action → direct call, no overhead.
        Async action → bridge via a one-shot worker thread with its own
        event loop (~2-5 ms). `contextvars.copy_context()` propagates
        the caller's tracing/OTel context into the worker.
        """
        method = self.get_action_method(action)
        if inspect.iscoroutinefunction(method):
            return self._bridge_async_to_sync(method, action)
        # sync dispatch (existing body — validate args, call, wrap result)
        ...

    async def async_execute_action(self, action: Action) -> Observation | StepError:
        """Async call-site. Works for both sync and async actions.

        Async action → direct await, no thread hop.
        Sync action → `asyncio.to_thread(method)` — pooled worker, no
        per-call spawn, enables real OS-thread parallelism when wrapped
        in `asyncio.gather`.

        Tools with thread-affinity needs (sync Playwright, etc.) override
        to dispatch through a per-instance single-threaded executor.
        """
        method = self.get_action_method(action)
        if inspect.iscoroutinefunction(method):
            result = await method(**action.arguments)
        else:
            result = await asyncio.to_thread(method, **action.arguments)
        # wrap → Observation | StepError (existing wrap logic)
        ...

    def _bridge_async_to_sync(
        self, async_method: Callable, action: Action
    ) -> Observation | StepError:
        """Run an async method to completion from a sync call-site.

        Implementation: spawn a daemon thread, run a new event loop
        inside it, block on the result. ~2-5 ms overhead per call.
        Caller's `contextvars` are propagated so OTel spans /
        logging state carry into the worker.
        """
        ctx = contextvars.copy_context()
        fut: concurrent.futures.Future = concurrent.futures.Future()

        def runner() -> None:
            try:
                fut.set_result(ctx.run(asyncio.run, async_method(**action.arguments)))
            except Exception as e:
                fut.set_exception(e)

        threading.Thread(target=runner, daemon=True).start()
        # wrap fut.result() → Observation | StepError
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

1. Every `@tool_action`-decorated method is reachable through BOTH `Tool.execute_action` and `Tool.async_execute_action`, regardless of the method's sync/async kind.
2. The fast paths (sync caller × sync action, async caller × async action) introduce no overhead beyond a normal Python method call.
3. The bridge paths (sync caller × async action, async caller × sync action) introduce a thread hop:
   - sync → async: one-shot daemon thread + new event loop, ~2-5 ms per call.
   - async → sync: pooled worker via `asyncio.to_thread` — no per-call spawn cost, enables real OS-thread parallelism inside `asyncio.gather`.
4. `action_set` lists all `@tool_action`-decorated methods irrespective of kind.
5. Tools wrapping thread-affine resources MAY override `async_execute_action` to route through a per-instance single-threaded executor; the base class's default uses the global thread pool.

### Gotchas

- `AsyncBrowserTool` (the single in-tree `AsyncTool` subclass) flips to `Tool` with no body change. After the alias is removed, downstream cubes that still subclass `AsyncTool` need a one-line edit.
- Tools that previously got an `__init_subclass__` error for mixed async/sync now pass silently. The previous structural mistake is now legal (mixing is supported); intentional misuse surfaces as a runtime bridge invocation that's slower than expected, not as an error.
- A sync caller invoking a hot-path async action repeatedly pays ~2-5 ms × N for the bridge. For agents that care about per-call latency, switch to `Agent._arun` so calls go through `async_execute_action` directly.

---

## REMOVED — none

`AbstractAsyncTool` and `AsyncTool` are not deleted — they become aliases of `AbstractTool` and `Tool` for one release window. Removal is a follow-up RFC after migrations land.

---

## ADDED — none beyond the dual surface above

`async_execute_action` was already added on `AbstractTool` by cube-standard
#152. This change reuses it as the universal call-site and removes the
class-level split that made the dual surface invisible.
