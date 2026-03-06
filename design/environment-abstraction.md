# Environment Abstraction Design

## Problem

Cubes and harnesses (e.g. AgentLab2) should be completely independent packages. Currently they are not.

### The implicit coupling in MiniWob

`miniwob-cube`'s `pyproject.toml` does not list `agentlab2` as a dependency, but the runtime coupling is real:

- `MiniWobTask.reset()` calls `self.tool.goto()`, `self.tool.evaluate_js()`, `self.tool.page_obs()`
- `MiniWobTask.finished()` calls `self.tool.evaluate_js()`
- None of these methods exist on cube's base classes
- `MiniWobTaskConfig` asserts: `"env_config must be set to either BrowsergymConfig or PlaywrightConfig"` — naming agentlab2 classes by string

MiniWob cannot run without an agentlab2 tool being injected. The dependency direction is wrong.

```text
CURRENT (wrong):
  miniwob-cube --[runtime dependency]--> agentlab2

CORRECT:
  miniwob-cube --> cube-standard (only)
  agentlab2    --> miniwob-cube  (to provide a concrete implementation)
```

---

## Naming note: Tool vs Environment

AL2 calls its browser backends `BrowsergymTool` and `PlaywrightTool`. Cube-standard went through
a rename from `Tool` → `Environment`. In practice, the distinction is mostly philosophical:
from the agent's perspective these are "tools it uses"; from the task's perspective they are
"the environment it runs in." Both framings are valid, and the rename touches a lot of code for
limited conceptual gain. This document uses "tool/environment" interchangeably and does not
mandate which name the codebase settles on.

---

## Key insight: the abstract is domain-level, not benchmark-level

The first instinct when fixing the coupling is to have each cube define its own
`AbstractXxxEnvironment`. But that forces every benchmark designer to write both an abstract
and a concrete class — an unrealistic burden, especially when many benchmarks share the same
backend (browser, terminal, etc.).

The right level for the abstract is the **domain**, not the benchmark:

```text
AbstractBrowserTool  ←  defined once in cube-standard
  task-internal:  goto(url), evaluate_js(js), page_obs()
  agent-facing:   browser_click, browser_type, browser_hover, ...
```

A web benchmark like MiniWob simply **declares** that its tasks require an `AbstractBrowserTool`.
It does not implement one. Only a genuinely novel domain (robot arm, GUI app, terminal) needs a
new abstract+concrete pair.

---

## What benchmark designers write

```python
# miniwob_cube/task.py

from cube.tools.browser import AbstractBrowserTool

class MiniWobTask(Task):
    def reset(self, tool: AbstractBrowserTool):
        tool.goto("https://...")
        return tool.page_obs()

    def finished(self, tool: AbstractBrowserTool) -> bool:
        return tool.evaluate_js("reward()")
```

No agentlab2 import. No assert on concrete class names. The dependency arrow is correct.

---

## Two-layer model

```text
Layer 1 – cube-standard (owns the contracts)
  AbstractBrowserTool:
    • task-internal:   goto(url), evaluate_js(js), page_obs()
    • agent-facing:    browser_click, browser_type, ...
  DefaultBrowserTool:
    • thin Playwright wrapper; ships with cube-standard (see decision below)

Layer 2 – Harness (provides optimized implementations)
  BrowsergymTool  implements AbstractBrowserTool  (full BrowserGym stack)
  PlaywrightTool  implements AbstractBrowserTool  (lightweight, sync Playwright)
```

Note: dynamic action addition by agent users was considered but is deferred — it adds complexity
without a concrete use case at this stage.

---

## Where does DefaultBrowserTool live? — the decision

### Initial thinking: keep cube-standard lean (Option B)

The simplest approach is to put no concrete implementation in cube-standard at all.
Cube-standard defines only `AbstractBrowserTool`; harnesses like AL2 provide everything concrete.
Cube-standard stays a pure standards package with no `playwright` dependency.

This was appealing until stress testing entered the picture.

### The stress-test requirement rules it out

Cube-standard ships `cube.testing.run_stress_test`, which is designed to run with only
`pip install cube my_cube` — no harness required:

```bash
cube stress-test my_cube
```

The mini harness calls `task_config.make()`, which instantiates a tool and runs a full episode.
For MiniWob that tool is a browser. If the only concrete `BrowserTool` lives in AL2, the stress
test cannot run without AL2 — defeating the entire point of the compliance check.

The stress test is the concrete use case that answers "is standalone runnability a hard
requirement?" — **yes**.

### Decision: DefaultBrowserTool ships in cube-standard (Option A)

```text
cube-standard:  AbstractBrowserTool + DefaultBrowserTool (thin Playwright wrapper)
miniwob-cube:   depends on cube-standard only; runs standalone, stress-test passes
agentlab2:      provides BrowsergymTool as optimized drop-in
```

The cost is that cube-standard gains a `playwright` dependency. This is acceptable: Playwright
is stable, widely available, and the alternative (a separate `cube-browser` package) adds a
third package to maintain for limited benefit.

---

## Summary of changes needed

| What | Where | Change |
|---|---|---|
| Define `AbstractBrowserTool` | `cube-standard/src/cube/tools/browser.py` | New file |
| Define `DefaultBrowserTool` | `cube-standard/src/cube/tools/browser.py` | New class (thin Playwright wrapper) |
| Remove forced `tool_config` assert | `miniwob_cube/task.py` | Use `AbstractBrowserTool` type instead |
| Remove agentlab2 class name strings | `miniwob_cube/task.py` | No more `"BrowsergymConfig or PlaywrightConfig"` in assert |
| Provide optimized browser tool | `agentlab2` | `BrowsergymTool` implements `AbstractBrowserTool` |
