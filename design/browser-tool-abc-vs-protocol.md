# BrowserTool: ABC vs Protocol

## Context

`cube-standard` currently has two overlapping abstractions for browser tools:

- `tool.py::BrowserTool(Tool)` — ABC extending `Tool`, with abstract methods `session`, `noop`, `page_obs`
- `tools/browser.py::AbstractBrowserTool` — `@runtime_checkable` Protocol, with `reset`, `close`, `execute_action`, `action_set`, `goto`, `evaluate_js`, `page_obs`

These serve different purposes but overlap significantly, causing confusion.

---

## ABC vs Protocol: General Analysis

### ABC

**Advantages:**
- Enforcement at class definition time — forgetting to implement an abstract method raises `TypeError` immediately
- Can provide shared implementation (default methods inherited by subclasses)
- Accurate `isinstance()` checks
- Better IDE support ("you must implement these methods")
- Works well when you own all implementations and want to enforce a hierarchy

**Disadvantages:**
- Concrete implementations must inherit from the ABC
- Creates a hard cross-package dependency if the ABC lives in `cube-standard` and the impl lives in `cube-browser-tool`

### Protocol

**Advantages:**
- Structural subtyping (duck typing) — any class with the right methods qualifies
- No inheritance required — concrete impls don't need to know about the Protocol
- Perfect for cross-package contracts
- Decoupled — any harness can provide an implementation without importing from `cube-standard`
- `runtime_checkable` enables `isinstance()` checks at runtime

**Disadvantages:**
- `isinstance()` only checks method *existence*, not signatures
- No enforcement at class definition time
- No shared implementation

---

## Architecture Consideration

The key factor: **concrete implementations live in a separate package** (`cube-tools/cube-browser-tool/` or `cube-harness`).

This makes Protocol the more natural fit for the external/structural contract — implementations don't need to import from `cube-standard` just to satisfy the interface.

The current ABC (`BrowserTool` in `tool.py`) was useful for in-package base classes that want the `Tool` machinery (`execute_action`, `action_set`, `@tool_action`), but it's incomplete relative to the Protocol (missing `goto`, `evaluate_js`).

---

## Decision: Merge the Two Abstractions

The two abstractions should be merged into one. Three options:

### Option A — ABC only (remove Protocol)
- `BrowserTool(Tool)` ABC in `tools/browser.py` with all abstract methods: `goto`, `evaluate_js`, `page_obs`, `noop`, `session`
- Concrete impls must subclass it → enforcement at definition time + `Tool` machinery for free
- **Downside:** `cube-browser-tool` must inherit from `cube-standard` (hard dependency)

### Option B — Protocol only (remove ABC)
- Single `AbstractBrowserTool` Protocol in `tools/browser.py` for both type annotations and external contract
- `cube-browser-tool` stays decoupled — no forced inheritance
- **Downside:** no enforcement at definition time, weaker `isinstance()` checks

### Option C — ABC without `Tool` inheritance
- `BrowserTool` is a standalone ABC in `tools/browser.py` (not extending `Tool`)
- Impls can still use `Tool` machinery by also subclassing `Tool`, but it's not forced
- Middle ground: enforced interface + decoupled from `Tool` hierarchy

### Recommendation

Given that concrete implementations live in a separate package, **Option B (Protocol only)** is the most architecturally consistent choice. `tools/browser.py` already uses this approach correctly — the fix is simply to remove `BrowserTool` from `tool.py` and ensure all tasks type against `AbstractBrowserTool`.
