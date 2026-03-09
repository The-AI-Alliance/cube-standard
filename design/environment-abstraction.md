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

Layer 2 – cube-tools/cube-browser-tool (provides implementations)
  BrowsergymTool  implements AbstractBrowserTool  (full BrowserGym stack)
  PlaywrightTool  implements AbstractBrowserTool  (lightweight, sync Playwright)
```

Note: dynamic action addition by agent users was considered but is deferred — it adds complexity
without a concrete use case at this stage.

---

## Where does AbstractBrowserTool live? — the decision

Three options were considered:

- **Option A** — `AbstractBrowserTool` + `DefaultBrowserTool` (thin Playwright wrapper) both in `cube-standard`. Simple, but adds a `playwright` hard dependency to cube-standard.
- **Option B** — `cube-standard` defines only `AbstractBrowserTool`; harnesses provide everything concrete. Cube-standard stays dependency-free, but the stress test cannot run without a harness installed.
- **Option C** — `cube-standard` defines only `AbstractBrowserTool`; a dedicated `cube-browser-tool` package provides one or more concrete implementations. Web benchmark cubes list `cube-browser-tool` as an **optional** dependency.

### Decision: Option C

```text
cube-standard/
  src/cube/                  ← AbstractBrowserTool (Protocol only, no playwright dependency)
  cube-tools/
    cube-browser-tool/       ← BrowsergymTool, PlaywrightTool — concrete implementations
      pyproject.toml         ← depends on cube-standard + playwright/browsergym
      src/cube_browser_tool/

miniwob-cube:  depends on cube-standard (required) + cube-browser-tool (optional extra)
agentlab2:     continues to provide its own optimized tools, also satisfying the Protocol
```

**Why Protocol, not ABC?**

`AbstractBrowserTool` is an *external-facing contract* meant to be implemented by third parties.
Any class with the right methods satisfies it without importing from `cube-standard` or inheriting
from its class hierarchy. `@runtime_checkable` is added so `isinstance(tool, AbstractBrowserTool)`
works in tests and debug contexts. The existing `AbstractTool(ABC)` in `tool.py` remains the
right pattern for cube's *internal* tool framework.

### Optional dependency in web benchmark cubes

```toml
# cube-miniwob/pyproject.toml
[project]
dependencies = ["cube-standard"]

[project.optional-dependencies]
browser = ["cube-browser-tool"]
```

- Regular users (bring their own tool): `pip install cube-miniwob`
- Quick start / stress test: `pip install cube-miniwob[browser]`

The stress test CLI (`cube stress-test my-cube`) raises a clear `ImportError` with install
instructions if `cube-browser-tool` is not present and no tool is provided.

---

## Summary of changes

### cube-standard repo

| What | Where | Status |
| --- | --- | --- |
| Define `AbstractBrowserTool` / `AsyncAbstractBrowserTool` | `cube-standard/src/cube/tools/browser.py` | Done |
| Helpful `ImportError` in debug/stress runner | `cube-standard/src/cube/testing.py` | Done |
| Define `BrowsergymTool`, `PlaywrightTool` | `cube-standard/cube-tools/cube-browser-tool/` | Follow-up PR |

### AgentLab2 repo (miniwob-cube)

| What | Where | Status |
| --- | --- | --- |
| `MiniWobTask.tool` typed as `AbstractBrowserTool` | `miniwob_cube/task.py` | Done |
| Remove agentlab2 class name strings from assert | `miniwob_cube/task.py` | Done |
| Add optional extra once `cube-browser-tool` exists | `miniwob-cube/pyproject.toml` → `[project.optional-dependencies] browser = ["cube-browser-tool"]` | Follow-up PR |
