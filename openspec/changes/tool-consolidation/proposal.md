# Proposal: Collapse `Tool` + `AsyncTool` into one `Tool`

## What this is, in one sentence

Right now, when you write a cube tool, you have to pick whether it's a *sync tool* or an *async tool* upfront — and once you pick, every `@tool_action` method on the class has to match. This proposal lets one tool class carry both kinds of methods, with dispatch routed per method.

## The friction today

Two parallel hierarchies live in `cube.tool`:

```python
# Option A: sync
class CalculatorTool(Tool):
    @tool_action
    def add(self, a: float, b: float) -> str:
        return f"{a + b}"

# Option B: async — and every @tool_action MUST be async
class AsyncBrowserTool(AsyncTool):
    @tool_action
    async def click(self, selector: str) -> str:
        await self._page.click(selector)
        return f"clicked {selector}"
```

If you're writing a browser tool that has one async operation (a Playwright `click` that awaits a network event) but ten sync helpers (a `screenshot` that reads a local PNG file, a `scroll_to_top` that's a one-liner), you can't put them on one class. Either:

- Make everything async (the sync helpers acquire a meaningless `async`/`await`), or
- Split into two classes (your cube now has `BrowserSyncHelpers` and `AsyncBrowserTool`, which is weird).

This is why **`AsyncBrowserTool` is the only `AsyncTool` subclass in the entire codebase today**. Tool authors hit this friction once and just pick sync.

There's a second problem: the `sync`/`async` distinction is encoded in the **class name** and the **base class**, but conceptually it's a per-method property. A reader can't tell from `class WebTool(AsyncTool)` whether `screenshot()` is fast-local or slow-network — they have to read each method.

## The change in one diagram

```mermaid
flowchart LR
    classDef gone fill:#fee2e2,stroke:#dc2626
    classDef stay fill:#dcfce7,stroke:#16a34a

    subgraph Today["Today — class-level split"]
        T1["AbstractTool"]:::stay
        T2["AbstractAsyncTool"]:::gone
        T3["Tool<br/>(all @tool_action sync)"]:::stay
        T4["AsyncTool<br/>(all @tool_action async)"]:::gone
        T1 --> T3
        T2 --> T4
    end

    subgraph After["After — one class, per-method routing"]
        A1["AbstractTool"]:::stay
        A2["Tool<br/>(@tool_action methods<br/>can be sync OR async)"]:::stay
        A1 --> A2
    end

    Today -.->|"AsyncTool, AbstractAsyncTool<br/>become aliases<br/>(deprecation window)"| After
```

`AsyncTool` and `AbstractAsyncTool` stay as deprecated aliases of the unified classes — existing code keeps working, downstream cubes migrate over one release window.

## What you write after this lands

The three cases authors care about, each in 4 lines:

```python
# Case 1 — pure sync (unchanged from today).
class CalculatorTool(Tool):
    @tool_action
    def add(self, a: float, b: float) -> str:
        return f"{a + b}"

# Case 2 — pure async (was AsyncTool; same body).
class AsyncBrowserTool(Tool):
    @tool_action
    async def click(self, selector: str) -> str:
        await self._page.click(selector)
        return f"clicked {selector}"

# Case 3 — mixed (the new affordance).
class BrowserTool(Tool):
    @tool_action
    def screenshot(self) -> bytes:
        return self._page.screenshot()         # sync local file read

    @tool_action
    async def navigate(self, url: str) -> str:
        await self._page.goto(url)             # async network roundtrip
        return f"loaded {url}"
```

Case 3 is the new thing. Today it's impossible without two classes.

## How dispatch works (the two call sites)

A `Tool` exposes **two** ways to execute an action. The caller picks based on their own call-site shape, not based on the tool:

```python
# Sync call-site (e.g. inside Agent._run, or a sync test):
result = tool.execute_action(action)

# Async call-site (e.g. inside Agent._arun + asyncio.gather):
result = await tool.async_execute_action(action)
```

The dispatch routes per the action method's kind:

```mermaid
flowchart TD
    classDef sync fill:#e0f2fe,stroke:#0284c7
    classDef async_ fill:#fef3c7,stroke:#d97706
    classDef bad fill:#fee2e2,stroke:#dc2626

    CS["sync call:<br/>tool.execute_action(a)"]:::sync
    AS["async call:<br/>await tool.async_execute_action(a)"]:::async_

    CS --> CSK{"action's<br/>def keyword?"}
    CSK -- "def" --> CSR["✓ run method, return value"]:::sync
    CSK -- "async def" --> CSE["✗ TypeError: 'action X is async —<br/>use async_execute_action'"]:::bad

    AS --> ASK{"action's<br/>def keyword?"}
    ASK -- "def" --> ASR1["✓ run method directly<br/>(no thread hop)"]:::async_
    ASK -- "async def" --> ASR2["✓ await method"]:::async_
```

Read this as: **the async call-site is the universal one** — it handles both kinds. **The sync call-site is the debuggable one** — it works only when the action method is sync, and refuses (with a clear error message naming the action) when it isn't.

This is the same dual-API shape cube-harness's `MonitoredTool` already ships. Mirroring it in cube-standard makes the pattern consistent end-to-end.

## What this means for agent authors

cube-harness's `Agent` base class already has two loop-body methods, picked by a config flag:

- **`Agent._run`** — sync body. The default. Loops `step → execute_action → step → ...`. No `await` anywhere on the action path.
- **`Agent._arun`** — async body. Opt-in via `AgentConfig.parallel_actions=True`. Dispatches N actions per step in parallel via `asyncio.gather`.

After this consolidation, each loop body maps cleanly to one dispatch method:

| Agent loop | Tool call site | Best for |
|---|---|---|
| `_run` (sync, default) | `tool.execute_action(a)` | The 99% case. Single-stack pdb. |
| `_arun` (async, opt-in) | `await tool.async_execute_action(a)` | LLM emits N independent tool calls per turn; wall-clock = max(latencies). |

Both bodies work against the **same** tool class. The agent author picks based on their loop semantics, not the tool's. A tool author shipping `BrowserTool` (mixed case 3 above) doesn't have to know or care which agent will use it.

## Why this helps debugging

A picture is worth a thousand words. Here are the actual call stacks when you hit a `breakpoint()` inside a tool action method:

**Sync agent (`_run`) calling a sync action via `execute_action`:**

```
pdb> bt
agent.py        _run                 result = env_tool.execute_action(action)
toolbox.py      execute_action       return leaf.execute_action(action)
user_tool.py    my_action            x = self.compute(...)      ← breakpoint here
```

Three frames. All on the main thread. `pdb> step` walks the code linearly. No event-loop frames, no thread-pool worker frames, no `to_thread` resumption frames in between. **This is the property `Agent._run` was designed to preserve, and the unified `Tool` keeps it intact.**

**Async agent (`_arun`) calling a sync action via `async_execute_action` + `gather`:**

```
pdb> bt
agent.py        _arun                results = await asyncio.gather(...)
asyncio         gather               (event-loop machinery)
toolbox.py      async_execute_action result = await leaf.async_execute_action(a)
monitored_tool  async_execute_action await asyncio.to_thread(inner.execute_action, a)
<worker thread> user_tool   my_action    x = self.compute(...)  ← breakpoint here
```

More frames, jumps to a thread-pool worker for the sync method. Still debuggable, just harder to reason about. This is the cost of opting into parallel dispatch.

**The point**: agent authors who don't need parallelism keep the cheap, debuggable path. Agent authors who do need parallelism pay only for what they ask for. Tool authors write one class and don't care which side calls them.

## Why this helps efficiency

**Sync default**: zero async overhead, zero thread-pool overhead. The function call is one frame deep through the toolbox dispatch into the tool method. Same as today.

**Parallel dispatch**: N actions of K seconds each finish in `max(K_i)` wall-clock instead of `sum(K_i)`. Sync actions inside the gather hop to a thread-pool worker via `asyncio.to_thread`; async actions are awaited directly. Real OS-thread parallelism for I/O-bound work.

A reference data point: cube-harness's parallelism smoke (`scripts/smoke/investigator_drivers.py`) measures clean dispatch through 24 parallel sync-action calls before any CLI-side contention. The new shape preserves that.

**Cost of misuse is a clear error, not a slow path**: if a sync agent accidentally calls into a tool method that's `async def`, `execute_action` raises `TypeError` naming the action and pointing at `async_execute_action`. No silent coroutine leaks, no surprise blocking.

## Migration story

| What you have today | After this lands |
|---|---|
| `class FooTool(Tool)` (all sync) | unchanged |
| `class FooTool(AsyncTool)` (all async) | works via alias; one-line edit to `Tool` recommended |
| `AsyncBrowserTool(AsyncTool)` | flips to `Tool` in this PR (canonical example) |
| `from cube.tool import AsyncTool, AbstractAsyncTool` | works, `DeprecationWarning` emitted on subclass |

The deprecation window stays open for one release. Downstream cubes can migrate at their own pace; no urgent action.

## Why not also collapse `Toolbox` + `AsyncToolbox`? (Phase 2)

`AsyncToolbox.execute_action` is *async by contract* — callers do `await tb.execute_action(action)`. If we collapsed `AsyncToolbox` into a sync-execute Toolbox, every existing `await tb.execute_action` call site would break.

The collapse is doable (mirror the dual API at the toolbox level too) but:

- ~5 `await tb.execute_action` sites in cube-harness today; breaking them all for marginal gain is the wrong trade.
- `AsyncToolbox` already accepts mixed sync + async leaves (cube-standard #152). So mixed-action `Tool` instances work inside it today without any further change.

Phase 2 is a separate RFC if the cost-benefit ever shifts.

## Alternatives we considered

- **Keep the split.** Status quo. The "one async tool method forces the whole class to be async" friction stays. Inconsistent with cube-harness `MonitoredTool` shape.
- **Make `execute_action` always async on the unified class, drop the sync method.** Cleaner one-method API, but the debuggable single-stack pdb story for `Agent._run` goes away — every tool call would have at least an event-loop frame in between. The dual surface is the point.
- **Drop the aliases entirely with no deprecation window.** Cleaner repo state, but breaks every downstream cube subclassing `AsyncTool` in one shot. The one-release-window cost is small and worth it.

## Risks

- **Class-definition-time validation goes away.** Today, `class FooTool(AsyncTool)` with a sync `@tool_action def` is an import-time error (`__init_subclass__` raises). After this PR, that's silently accepted; the failure surfaces at first sync `execute_action` call as a `TypeError`. Mitigation: the error message names the action and points at the right call site.
- **`Tool` is a moderately exposed name.** Every cube subclasses it. The unit-test matrix in `tests/test_tool.py` covers all four dispatch combinations (sync action / async action) × (sync call / async call); downstream cubes should also run their own `pytest tests/` once the change lands.

## Companion work

- **No cube-harness change required.** Once cube-standard ships the next rc with this change, `MonitoredTool.async_execute_action` automatically becomes the unified call-site for any inner kind without extra branching.
- **`AsyncBrowserTool`** (in `cube-resources/cube-browser-playwright/`) flips from `AsyncTool` → `Tool` in this PR (canonical example; one-line edit).
