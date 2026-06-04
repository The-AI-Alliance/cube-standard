# Proposal: Collapse `Tool` + `AsyncTool` into one `Tool`

## Problem

`cube.tool` exposes two parallel hierarchies:

- `AbstractTool` / `Tool` — sync; `@tool_action` methods MUST be sync.
- `AbstractAsyncTool` / `AsyncTool` — async; `@tool_action` methods MUST be async (validated at class-definition time).

Cube authors have to pick the right base class up front. A tool with one async operation (one HTTP roundtrip, one Playwright click) plus N sync helpers cannot exist as one class — it must split or block. The single tool that takes this route today, `AsyncBrowserTool`, is the only `AsyncTool` subclass in tree.

Downstream, cube-harness's `MonitoredTool` already shipped (PR #386 / #152) the dual-API pattern that resolves this:

- `execute_action(action)` — sync. Debuggable single-stack pdb. Works for sync actions; raises for async actions on this tool.
- `async_execute_action(action)` — async. Universal call-site. Works for both kinds: sync runs on the current thread; async is awaited.

cube-standard's `Tool` should mirror this so the pattern is consistent end-to-end (and so the per-action async/sync distinction lives where it belongs — at the method, not at the class).

## Change (Phase 1)

**Collapse `Tool` + `AsyncTool` into one `Tool`. Keep containers as-is.**

- `AbstractTool` gets the dual call surface: sync `execute_action` + async `async_execute_action` (`async_execute_action` already exists as a non-abstract method from cube-standard #152; this PR makes it the canonical universal call-site).
- Drop `AbstractAsyncTool` as a distinct ABC. Keep `AbstractAsyncTool = AbstractTool` as a deprecated alias (one release).
- `Tool` (concrete base) accepts `@tool_action` methods that are sync OR async on the same class. Per-method dispatch.
- `AsyncTool` becomes a deprecated alias of `Tool` (one release).
- `_ToolActionsMixin.__init_subclass__` validation relaxes — no longer requires "all-sync" or "all-async"; mixed is allowed.

**Toolbox / AsyncToolbox stay as today** (cube-standard #152 already relaxed `AsyncToolbox` to accept mixed leaves; sync `Toolbox` still requires all-sync action leaves). A Phase 2 RFC could unify those too, but the leverage is much smaller.

## Migration

Existing code keeps working unchanged:

- `class FooTool(Tool)` with sync `@tool_action` methods — unchanged.
- `class FooTool(AsyncTool)` with async `@tool_action` methods — unchanged via the alias. Recommendation: switch to `class FooTool(Tool)` (one-line edit) before the deprecation window closes.

The single in-tree `AsyncTool` subclass — `cube.tools.browser.AsyncBrowserTool` — flips from `class AsyncBrowserTool(AsyncTool)` to `class AsyncBrowserTool(Tool)` with no body change.

Downstream cubes get a deprecation warning when they subclass the alias; the warning is suppressible for one release while they migrate.

## Why not Phase 2 (collapse `Toolbox` + `AsyncToolbox`)

`AsyncToolbox.execute_action` is async; callers do `await tb.execute_action(action)`. Collapsing it into a sync-execute-action `Toolbox` would break every `await` call site. The collapse is doable (mirror the dual API at the toolbox level) but the cost-benefit is poor today: 2 toolbox classes vs ~5 instances of "await on AsyncToolbox.execute_action" in cube-harness. Phase 2 can land later when the cost-benefit shifts.

## Alternatives considered

- **Keep the split.** Status quo. Forces cube authors to pick a class hierarchy up front; per-action sync/async distinction is invisible in the class name. Inconsistent with cube-harness's `MonitoredTool` dual API.
- **Drop the alias entirely (no deprecation window).** Cleaner but breaks every downstream cube subclassing `AsyncTool` in one shot.

## Risks

- Class-definition-time validation goes away (`__init_subclass__` no longer enforces all-sync or all-async). Authors who put a sync method in an async class previously got an import-time error; now they get a runtime `TypeError` when the sync dispatch path hits the async method. Mitigation: clear error message naming the action.
- `Tool` is a moderately exposed name. Every cube subclasses it. The risk surface for an `__init_subclass__` change is wide; tests in cube-standard cover the matrix; downstream cubes should run their own `pytest tests/`.

## Companion work

- No companion cube-harness change required. Once cube-standard ships rc10 (or whatever the next release is) with this change, cube-harness `MonitoredTool` automatically benefits (its `async_execute_action` becomes the unified call-site for any inner kind without extra branching).
- `AsyncBrowserTool` (in `cube-resources/cube-browser-playwright/`) gets a one-line edit migrating from `AsyncTool` to `Tool`.
