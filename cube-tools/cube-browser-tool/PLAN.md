# cube-browser-tool — Implementation Plan

Concrete implementations of `AbstractBrowserTool` / `AsyncAbstractBrowserTool`
(protocols defined in `cube-standard/src/cube/tools/browser.py`).

This package lives at `cube-standard/cube-tools/cube-browser-tool/` and is published to PyPI as `cube-browser-tool`.

---

## Package layout

```
cube-browser-tool/
  pyproject.toml
  src/
    cube_browser_tool/
      __init__.py          # public exports
      action_spaces.py     # BrowserActionSpace, BidBrowserActionSpace ABCs
      playwright_tool.py   # PlaywrightConfig, SyncPlaywrightTool, AsyncPlaywrightTool
      _utils.py            # flatten_axtree, prune_html (shared by playwright and bgym tools)
      bgym_tool.py         # BgymToolConfig, BgymTool  (Phase 3 — optional dep)
  tests/
    test_playwright_tool.py
    test_bgym_tool.py      # Phase 3
```

---

## Dependencies

```toml
[project]
dependencies = [
    "cube-standard",
    "playwright",
    "pillow",
    "beautifulsoup4",
]

[project.optional-dependencies]
bgym = ["browsergym-core"]
dev  = ["pytest", "pytest-asyncio"]
```

`cube-standard` is required for `Tool`, `ToolConfig`, `Action`, `ActionSchema`,
`Observation`, `StepError`, `Content`.

`beautifulsoup4` is required for `prune_html` in `_utils.py`.

`browsergym-core` is an optional extra — importing `cube_browser_tool.bgym_tool`
raises a clear `ImportError` with install instructions if it is not present.

---

## Relationship to cube-standard's Tool infrastructure

`cube-standard` ships a `Tool` base class with:
- `execute_action(action)` — dispatches to the method named `action.name`
- `action_set` (property) — auto-discovers methods decorated with `@tool_action`
- `async_execute_action(action)` — async dispatch for coroutine methods

`SyncPlaywrightTool` subclasses `Tool` and decorates its browser methods with
`@tool_action`. `execute_action` is overridden once to append `page_obs()` after
every successful action:

```python
def execute_action(self, action: Action) -> Observation | StepError:
    result = super().execute_action(action)
    if isinstance(result, StepError):
        return result
    return result + self.page_obs()
```

`AsyncPlaywrightTool` does the same via `async_execute_action` and overrides
`execute_action` with an async version (structurally satisfying
`AsyncAbstractBrowserTool` without inheriting from it).

---

## Phase 1 — SyncPlaywrightTool ✓ DONE

All files implemented, tests passing. Adapted from AgentLab2 with key changes:
`BrowserActionSpace`/`BidBrowserActionSpace` ABCs in `action_spaces.py`;
`flatten_axtree`/`prune_html` in `_utils.py`; `SyncPlaywrightTool` extends `Tool`
with `@tool_action` methods; `execute_action` appends `page_obs()` on success;
`browser_scroll` added (new vs AL2); `goto` kept as task-internal only.

### `PlaywrightConfig`

Pydantic model extending `ToolConfig` (for serialization via `TypedBaseModel`).
`make()` creates a `SyncPlaywrightTool`.

```python
class PlaywrightConfig(ToolConfig):
    headless: bool = True
    viewport: dict = {"width": 1280, "height": 720}
    chromium_sandbox: bool = True
    max_wait: int = 60
    use_html: bool = True
    use_axtree: bool = False
    use_screenshot: bool = True
    prune_html: bool = True
    pw_kwargs: dict = {}

    def make(self, container=None) -> "SyncPlaywrightTool": ...
```

### `action_spaces.py`

Two ABCs that define the canonical action contracts. Both use `@tool_action` on
abstract methods so that subclass overrides inherit the registration without
repeating the decorator. Neither includes `goto` — that is a task-internal method
on the tool, not an agent-facing action.

- **`BrowserActionSpace`** — CSS-selector-based (used by `SyncPlaywrightTool`)
- **`BidBrowserActionSpace`** — BID-based (used by `BgymTool` in Phase 3)

Both define the same set of actions:
`browser_click`, `browser_type`, `browser_press_key`, `browser_hover`,
`browser_drag`, `browser_select_option`, `browser_mouse_click_xy`,
`browser_scroll`, `browser_back`, `browser_forward`, `browser_wait`, `noop`.

All docstrings use NumPy format — the summary line becomes the LLM-facing action
description; `Parameters` sections populate the per-argument descriptions in the
JSON schema (parsed by `cube-standard`'s `function_to_dict`).

### `SyncPlaywrightTool`

Extends `Tool` and `BrowserActionSpace`. Playwright context created eagerly in
`__init__`. Methods:

| Method | Type | Notes |
|--------|------|-------|
| `reset()` | lifecycle | close + reopen page (clear state between episodes) |
| `close()` | lifecycle | close page, browser, playwright |
| `goto(url)` | task-internal | `page.goto(url)` |
| `evaluate_js(js)` | task-internal | `page.evaluate(js)` |
| `page_obs()` | task-internal | assembles Observation from html/axtree/screenshot per config |
| `execute_action(action)` | override | calls super, appends page_obs() on success |
| `browser_click(selector)` | `@tool_action` | |
| `browser_type(selector, text)` | `@tool_action` | |
| `browser_press_key(key)` | `@tool_action` | |
| `browser_hover(selector)` | `@tool_action` | |
| `browser_drag(from_selector, to_selector)` | `@tool_action` | |
| `browser_select_option(selector, value)` | `@tool_action` | |
| `browser_mouse_click_xy(x, y)` | `@tool_action` | |
| `browser_scroll(selector, direction, amount)` | `@tool_action` | new vs AL2 |
| `browser_back()` | `@tool_action` | |
| `browser_forward()` | `@tool_action` | |
| `browser_wait(seconds)` | `@tool_action` | capped at `max_wait` |
| `noop()` | `@tool_action` | |

`page_obs()` helpers (not actions):
- `page_html() -> str` — `page.content()`
- `page_screenshot() -> Image` — `page.screenshot()` → PIL Image
- `page_axtree() -> str` — `page.accessibility.snapshot()` → `flatten_axtree()`

`flatten_axtree` lives in `_utils.py` (shared with the future bgym tool).

### Tests

`tests/test_playwright_tool.py`:
- Requires a live Playwright/Chromium install; marked `@pytest.mark.integration` or
  guarded with `pytest.importorskip("playwright")`.
- Round-trip: create tool → goto a local data URL → check `page_obs()` content.
- Action dispatch: execute a `browser_click` action, confirm `page_obs()` appended.
- `action_set` lists expected action names.

---

## Phase 2 — AsyncPlaywrightTool

**Files modified:** `playwright_tool.py`, `__init__.py`, `tests/test_playwright_tool.py`

### Design

`AsyncPlaywrightTool` cannot create the Playwright context in `__init__`
(Playwright async API requires an async context). Initialization is deferred to
`reset()`, which is the first async call in the episode lifecycle:

```python
class AsyncPlaywrightTool:
    async def reset(self) -> None:
        if self._apw is None:
            self._apw = await async_playwright().start()
            self._abrowser = await self._apw.chromium.launch(...)
        else:
            await self._page.close()
        self._page = await self._abrowser.new_page()
```

`execute_action` is overridden as a coroutine (satisfying `AsyncAbstractBrowserTool`
structurally without inheritance):

```python
async def execute_action(self, action: Action) -> Observation | StepError:
    result = await self.async_execute_action(action)
    if isinstance(result, StepError):
        return result
    return result + await self.page_obs()
```

All `@tool_action` methods become coroutines; `action_set` (the property) remains
sync — it reads method metadata without I/O.

### Tests

`tests/test_playwright_tool.py` gets an async section with `pytest-asyncio`:
- `reset()` initializes the browser lazily.
- Same action dispatch and `page_obs()` checks as Phase 1.

---

## Phase 3 — BgymTool (optional dep)

**Files created:** `src/cube_browser_tool/bgym_tool.py`,
`tests/test_bgym_tool.py`

`BgymTool` wraps BrowserGym's action and observation functions. Two approaches
are possible; the decision is deferred until after Phases 1–2.

### Approach A — Port AL2's `BrowsergymTool` (wraps `BrowserEnv`)

AL2's `BrowsergymTool` wraps BrowserGym's `BrowserEnv` class, which manages the
full BrowserGym episode lifecycle (browser launch, BID injection, obs extraction).

**Pros:**
- Closest to the battle-tested AL2 implementation — lower implementation risk.
- BrowserGym task classes (`AbstractBrowserTask.setup/validate`) integrate
  naturally via `BrowserEnv`.

**Cons:**
- `BrowserEnv` carries significant complexity (BrowserGym task lifecycle, internal
  state) that `BgymTool` doesn't need — the tool's only job is executing actions
  and extracting observations.
- `set_gym_task()` anti-pattern: BrowserEnv expects to own the task, but in our
  design the task logic lives in TaskLogic, not the tool.
- Hard to decouple; porting essentially brings in `BrowserEnv` as a black box.

### Approach B — Lean implementation per browser-tool.md (direct Page functions)

`BgymTool` calls BrowserGym's standalone functions directly on a Playwright `Page`:
- **Actions:** `HighLevelActionSet` for schema; `execute_python_code(code, page)`
  for dispatch after serializing the `Action` back to a code string.
- **Observations:** `_pre_extract(page)` (BID injection) + `extract_screenshot` +
  `extract_merged_axtree` + `extract_dom_snapshot`.

**Pros:**
- No `BrowserEnv` — `BgymTool` stays thin and focused.
- Works with any `Page` (from `SyncBrowserSession` or passed directly) — no
  coupling to BrowserGym's task lifecycle.
- Clean separation: TaskLogic drives the task; tool only handles actions + obs.
- Matches the target architecture in browser-tool.md.

**Cons:**
- More implementation work: requires understanding BrowserGym's standalone APIs.
- `HighLevelActionSet → list[ActionSchema]` mapping needs to be written
  (open question #2 from browser-tool.md: one schema per BrowserGym action vs.
  a single `execute_browser_action(action_str)` wrapper).

**Recommendation (to revisit):** Approach B if BrowserGym exposes stable
standalone APIs for action execution. Fall back to Approach A if those APIs are
private or unstable.

---

## What is NOT in scope for this package

The `BrowserSession` / `BrowserSessionConfig` / `BrowserTask` / `AbstractTaskLogic`
redesign from `browser-tool.md` is a cube-standard concern (not cube-browser-tool).
That redesign is tracked separately and does not block this package.
