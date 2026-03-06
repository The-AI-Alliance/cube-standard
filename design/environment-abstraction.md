# Environment Abstraction Design

## Problem

Cubes and harnesses (e.g. AgentLab2) should be completely independent packages. Currently they are not.

### The implicit coupling in MiniWob

`miniwob-cube`'s `pyproject.toml` does not list `agentlab2` as a dependency, but the runtime coupling is real:

- `MiniWobTask.reset()` calls `self.tool.goto()`, `self.tool.evaluate_js()`, `self.tool.page_obs()`
- `MiniWobTask.finished()` calls `self.tool.evaluate_js()`
- None of these methods exist on cube's `AbstractTool`
- `MiniWobTaskConfig` asserts: `"tool_config must be set to either BrowsergymConfig or PlaywrightConfig"` — naming agentlab2 classes by string

MiniWob cannot run without an agentlab2 tool being injected. The dependency direction is wrong.

```
CURRENT (wrong):
  miniwob-cube --[runtime dependency]--> agentlab2

CORRECT:
  miniwob-cube --> cube-standard (only)
  agentlab2    --> miniwob-cube  (to provide an optimized implementation)
```

---

## Three-Layer Model

There are three distinct sources of actions:

```
Layer 1 – Cube (owns the contract)
  Defines AbstractXxxEnvironment:
    • task-internal methods:  goto(url), evaluate_js(js), page_obs()    [not in agent's action_set]
    • agent-facing actions:   @tool_action browser_click, browser_type, ...
  Provides DefaultXxxEnvironment:
    • minimal working implementation (e.g. raw Playwright, requests)
    • cube works standalone, no harness required

Layer 2 – Harness (provides optimized implementations)
  BrowsergymEnv satisfies AbstractMiniWobEnvironment
  PlaywrightEnv  satisfies AbstractMiniWobEnvironment
  These are drop-in replacements, not requirements.

Layer 3 – Agent user (adds custom actions at runtime)
  Extra functions (env, **kwargs) -> result registered on the task
  Exposed in action_set, dispatched separately from env built-ins
```

---

## Desired Requirements (from `examples/test.py`)

```
1. benchmark -> task -> env comes with its own default actions.
2. harness has existing env actions that can manipulate env.
3. Agent users can define their own env actions.
```

### Requirement 1 — Cube provides default actions

The cube defines its own abstract environment AND ships a default concrete implementation:

```python
# miniwob_cube/environment.py

class AbstractMiniWobEnvironment(AbstractTool):
    """Contract that any MiniWob environment must satisfy."""

    # Task-internal (used by MiniWobTask, NOT in agent action_set)
    @abstractmethod
    def goto(self, url: str) -> str: ...
    @abstractmethod
    def evaluate_js(self, js: str) -> Any: ...
    @abstractmethod
    def page_obs(self) -> Observation: ...

    # Agent-facing actions
    @tool_action
    @abstractmethod
    def browser_click(self, bid: str) -> str: ...

    @tool_action
    @abstractmethod
    def browser_type(self, bid: str, text: str) -> str: ...
    # ...


class DefaultMiniWobEnvironment(AbstractMiniWobEnvironment, Tool):
    """Works out of the box with raw Playwright — no agentlab2 needed."""
    # simple but functional implementation
```

`MiniWobTaskConfig.make()` defaults to `DefaultMiniWobEnvironment`. The assert requiring agentlab2 is removed.

### Requirement 2 — Harness provides an optimized drop-in

The harness implements the cube's interface. The dependency arrow is now correct: agentlab2 imports from miniwob-cube, never the other way.

```python
# agentlab2/cubes/miniwob_adapter.py

from miniwob_cube.environment import AbstractMiniWobEnvironment
from agentlab2.tools.browsergym import BrowsergymTool

class MiniWobBrowsergymEnv(BrowsergymTool, AbstractMiniWobEnvironment):
    """Drop-in: gives MiniWob tasks the full BrowserGym stack."""
    pass
```

The harness user passes `tool_config=MiniWobBrowsergymConfig()` to override the default. If they don't, the default impl is used.

### Requirement 3 — Agent users add custom actions

The existing `@tool_action` + subclassing already handles this for users who own the environment class. For users who want to add actions without subclassing (e.g. injecting a standalone function):

```python
# examples/test.py pattern:
class MyTool():
    def add2(env: MyEnv):
        env.state += 2
```

This needs a first-class concept on `Task`: a list of **extra actions** — callables of the form `(env, **kwargs) -> result` registered separately from the environment's built-in methods.

```python
class ExtraAction(TypedBaseModel):
    schema: ActionSchema
    fn: Callable  # (env, **kwargs) -> result

class Task:
    extra_actions: list[ExtraAction] = []

    @property
    def action_set(self):
        return self.filter_actions(
            self.tool.action_set + [ea.schema for ea in self.extra_actions]
        )

    def step(self, action):
        # dispatch to env or extra_actions
        if action.name in {a.name for a in self.tool.action_set}:
            result = self.tool.execute_action(action)
        else:
            ea = next(ea for ea in self.extra_actions if ea.schema.name == action.name)
            result = ea.fn(self.tool, **action.arguments)
```

---

## Naming

`Tool` → `Environment` (or `AbstractEnvironment`, `EnvironmentConfig`, `Environment`)

This reflects what the class actually is: the stateful environment the agent interacts with, not the agent's tool. The rename is independent of the design changes above and can be done separately.

---

## Summary of Changes Needed

| What | Where | Change |
|---|---|---|
| Define `AbstractMiniWobEnvironment` | `miniwob_cube/environment.py` | New file |
| Define `DefaultMiniWobEnvironment` | `miniwob_cube/environment.py` | New class (raw Playwright) |
| Remove forced `tool_config` assert | `miniwob_cube/task.py` | Default to `DefaultMiniWobEnvConfig` |
| Define `MiniWobBrowsergymEnv` | `agentlab2/cubes/miniwob_adapter.py` | New adapter class |
| Add `extra_actions` to `Task` | `cube-standard/src/cube/task.py` | New field + dispatch logic |
| Rename `Tool` → `Environment` | `cube-standard/src/cube/tool.py` | Rename (separate PR) |
