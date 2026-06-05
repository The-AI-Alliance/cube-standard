#!/usr/bin/env python3
"""Smoke: end-to-end exercise of the consolidated Tool dispatch.

Verifies the four cells of the (sync/async caller) × (sync/async action)
matrix on a real `Tool` subclass with mixed @tool_action methods, plus
the deprecation shims (AsyncTool, AsyncToolbox) still work end-to-end.

Run:
    uv run scripts/smoke/tool_consolidation.py

Exits 0 with `SMOKE OK: tool_consolidation` on success, 1 with
`SMOKE FAIL: …` otherwise.
"""

from __future__ import annotations

import asyncio
import time
import warnings

from cube.core import Action, Observation
from cube.tool import AsyncTool, AsyncToolbox, Tool, Toolbox, tool_action

NAME = "tool_consolidation"


class MixedTool(Tool):
    """One Tool, both kinds of @tool_action method."""

    @tool_action
    def fast_sync(self, x: int) -> str:
        """Sync, fast."""
        return f"sync:{x * 2}"

    @tool_action
    async def slow_async(self, x: int) -> str:
        """Async, fast (no real I/O — keeps the smoke quick)."""
        return f"async:{x * 2}"

    @tool_action
    def sleep_a_bit(self, ms: int) -> str:
        """Sync sleep — used to time parallel gather."""
        time.sleep(ms / 1000)
        return f"slept:{ms}"


def _fail(msg: str) -> int:
    print(f"  ✗ {msg}")
    print(f"SMOKE FAIL: {NAME}")
    return 1


def _ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def _cell_sync_caller_sync_action(tool: MixedTool) -> str | None:
    result = tool.execute_action(Action(name="fast_sync", arguments={"x": 5}))
    if not isinstance(result, Observation):
        return f"got non-Observation result: {type(result).__name__}"
    if result.contents[0].data != "sync:10":
        return f"got data {result.contents[0].data!r}, expected 'sync:10'"
    return None


def _cell_sync_caller_async_action(tool: MixedTool) -> str | None:
    # Bridge path: sync caller invoking an async action.
    # Should spawn a worker thread and bridge cleanly.
    result = tool.execute_action(Action(name="slow_async", arguments={"x": 5}))
    if not isinstance(result, Observation):
        return f"got non-Observation result: {type(result).__name__}"
    if result.contents[0].data != "async:10":
        return f"got data {result.contents[0].data!r}, expected 'async:10'"
    return None


async def _cell_async_caller_sync_action(tool: MixedTool) -> str | None:
    # to_thread path: async caller invoking a sync action.
    result = await tool.async_execute_action(Action(name="fast_sync", arguments={"x": 5}))
    if not isinstance(result, Observation):
        return f"got non-Observation result: {type(result).__name__}"
    if result.contents[0].data != "sync:10":
        return f"got data {result.contents[0].data!r}, expected 'sync:10'"
    return None


async def _cell_async_caller_async_action(tool: MixedTool) -> str | None:
    # Direct-await path: async caller invoking an async action.
    result = await tool.async_execute_action(Action(name="slow_async", arguments={"x": 5}))
    if not isinstance(result, Observation):
        return f"got non-Observation result: {type(result).__name__}"
    if result.contents[0].data != "async:10":
        return f"got data {result.contents[0].data!r}, expected 'async:10'"
    return None


async def _parallel_gather_realism(tool: MixedTool) -> str | None:
    """4 × 100ms sync sleeps via gather should land in well under 4×100=400ms
    if the to_thread path is providing real OS-thread parallelism."""
    actions = [Action(name="sleep_a_bit", arguments={"ms": 100}) for _ in range(4)]
    start = time.time()
    results = await asyncio.gather(*[tool.async_execute_action(a) for a in actions])
    elapsed = time.time() - start
    if elapsed > 0.25:
        return f"parallel gather ran sequentially? elapsed={elapsed:.3f}s (expected < 0.25s)"
    if not all(isinstance(r, Observation) for r in results):
        return "non-Observation in gather results"
    return None


async def _bridge_from_running_loop(tool: MixedTool) -> str | None:
    """The bridge MUST work from inside an already-running event loop —
    motivating scenario is `Agent._run` inside Episode's `asyncio.run`.
    Bridge uses a thread (not the running loop), so this should succeed.
    """

    def sync_call() -> Observation:
        # Wrapped in a function so the bridge sees a fresh sync caller.
        return tool.execute_action(Action(name="slow_async", arguments={"x": 7}))

    # Yield to the loop and call sync — simulates calling sync code mid-await.
    await asyncio.sleep(0)
    result = sync_call()
    if not isinstance(result, Observation):
        return f"got non-Observation result inside running loop: {type(result).__name__}"
    if result.contents[0].data != "async:14":
        return f"got data {result.contents[0].data!r}, expected 'async:14'"
    return None


def _legacy_async_tool_shim_emits_deprecation() -> str | None:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")

        class _Legacy(AsyncTool):
            @tool_action
            async def hi(self) -> str:
                """Legacy."""
                return "hi"

    deprecations = [w for w in captured if issubclass(w.category, DeprecationWarning)]
    if not any("AsyncTool is deprecated" in str(w.message) for w in deprecations):
        return "no DeprecationWarning emitted on subclassing AsyncTool"
    return None


async def _legacy_asynctoolbox_shim_works() -> str | None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)

        class _LegacyAsyncEcho(AsyncTool):
            @tool_action
            async def echo(self, msg: str) -> str:
                """Echo."""
                return msg

        box = AsyncToolbox(tools=[_LegacyAsyncEcho()])

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = await box.execute_action(Action(name="echo", arguments={"msg": "hi"}))

    if not isinstance(result, Observation):
        return f"AsyncToolbox.execute_action gave non-Observation: {type(result).__name__}"
    if result.contents[0].data != "hi":
        return f"AsyncToolbox.execute_action gave {result.contents[0].data!r}, expected 'hi'"
    deprecations = [w for w in captured if issubclass(w.category, DeprecationWarning)]
    if not any("AsyncToolbox.execute_action is deprecated" in str(w.message) for w in deprecations):
        return "no DeprecationWarning emitted on calling AsyncToolbox.execute_action"
    return None


def _unified_toolbox_routes_both_kinds() -> str | None:
    """Toolbox containing a single MixedTool routes sync and async actions
    through the sync `execute_action` (sync caller path)."""
    box = Toolbox(tools=[MixedTool()])
    sync_result = box.execute_action(Action(name="fast_sync", arguments={"x": 1}))
    if not isinstance(sync_result, Observation) or sync_result.contents[0].data != "sync:2":
        return f"Toolbox sync-action dispatch wrong: {sync_result}"
    async_result = box.execute_action(Action(name="slow_async", arguments={"x": 1}))
    if not isinstance(async_result, Observation) or async_result.contents[0].data != "async:2":
        return f"Toolbox async-action dispatch (via bridge) wrong: {async_result}"
    return None


async def main_async() -> int:
    tool = MixedTool()

    for label, err in [
        ("(sync caller × sync action) direct dispatch", _cell_sync_caller_sync_action(tool)),
        ("(sync caller × async action) bridge dispatch", _cell_sync_caller_async_action(tool)),
        ("(async caller × sync action) to_thread", await _cell_async_caller_sync_action(tool)),
        ("(async caller × async action) direct await", await _cell_async_caller_async_action(tool)),
        ("parallel gather over sync actions (real parallelism)", await _parallel_gather_realism(tool)),
        ("bridge from inside a running loop", await _bridge_from_running_loop(tool)),
        ("AsyncTool shim emits DeprecationWarning", _legacy_async_tool_shim_emits_deprecation()),
        ("AsyncToolbox.execute_action shim still works + warns", await _legacy_asynctoolbox_shim_works()),
        ("Toolbox routes both kinds through bridge", _unified_toolbox_routes_both_kinds()),
    ]:
        if err is not None:
            return _fail(f"{label}: {err}")
        _ok(label)

    print(f"SMOKE OK: {NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async()))
