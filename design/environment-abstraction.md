# Environment Abstraction Design

## Problem

Cubes and harnesses (e.g. AgentLab2) should be completely independent packages. Currently they are not.

### The implicit coupling in MiniWob

`miniwob-cube`'s `pyproject.toml` does not list `agentlab2` as a dependency, but the runtime coupling is real:

- `MiniWobTask.reset()` calls `self.tool.goto()`, `self.tool.evaluate_js()`, `self.tool.page_obs()`
- `MiniWobTask.finished()` calls `self.tool.evaluate_js()`
- None of these methods exist on cube's `AbstractEnvironment`
- `MiniWobTaskConfig` asserts: `"env_config must be set to either BrowsergymConfig or PlaywrightConfig"` — naming agentlab2 classes by string

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
    • agent-facing actions:   @environment_action browser_click, browser_type, ...
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

class AbstractMiniWobEnvironment(AbstractEnvironment):
    """Contract that any MiniWob environment must satisfy."""

    # Task-internal (used by MiniWobTask, NOT in agent action_set)
    @abstractmethod
    def goto(self, url: str) -> str: ...
    @abstractmethod
    def evaluate_js(self, js: str) -> Any: ...
    @abstractmethod
    def page_obs(self) -> Observation: ...

    # Agent-facing actions
    @environment_action
    @abstractmethod
    def browser_click(self, bid: str) -> str: ...

    @environment_action
    @abstractmethod
    def browser_type(self, bid: str, text: str) -> str: ...
    # ...


class DefaultMiniWobEnvironment(AbstractMiniWobEnvironment, Environment):
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

The harness user passes `env_config=MiniWobBrowsergymConfig()` to override the default. If they don't, the default impl is used.

### Requirement 3 — Agent users add custom actions

The existing `@environment_action` + subclassing already handles this for users who own the environment class. For users who want to inject extra tools without subclassing (e.g. a standalone calculator or code-execution tool alongside a browser task), the harness expresses them as serializable configs.

`TaskConfig` carries both the main env config and any extra tool configs. All configs are serializable so they cross worker boundaries safely. `Task.model_post_init` instantiates them, mirroring the existing `env_config` → `_env` pattern:

```python
class TaskConfig(ABC, TypedBaseModel):
    task_id: str
    seed: int | None = None
    env_config: EnvironmentConfig | None = None
    extra_tool_configs: list[EnvironmentConfig] = []   # ← new field

class Task(TypedBaseModel, ABC):
    # serializable fields
    extra_tool_configs: list[EnvironmentConfig] = []

    # non-serializable runtime state (set during model_post_init)
    _env: AbstractEnvironment | None = PrivateAttr(default=None)
    _extra_tools: list[AbstractEnvironment] = PrivateAttr(default_factory=list)  # ← new

    def model_post_init(self, __context):
        # ... existing container + env setup ...
        self._extra_tools = [cfg.make() for cfg in self.extra_tool_configs]

    @property
    def action_set(self):
        extra = [schema for tool in self._extra_tools for schema in tool.action_set]
        return self.filter_actions(self.env.action_set + extra)

    def step(self, action):
        # dispatch to main env or extra tools
        if action.name in {a.name for a in self.env.action_set}:
            result = self.env.execute_action(action)
        else:
            tool = next(
                t for t in self._extra_tools
                if action.name in {a.name for a in t.action_set}
            )
            result = tool.execute_action(action)
```

---

## Summary of Changes Needed

| What | Where | Change |
|---|---|---|
| Define `AbstractMiniWobEnvironment` | `miniwob_cube/environment.py` | New file |
| Define `DefaultMiniWobEnvironment` | `miniwob_cube/environment.py` | New class (raw Playwright) |
| Remove forced `tool_config` assert | `miniwob_cube/task.py` | Default to `DefaultMiniWobEnvConfig` |
| Define `MiniWobBrowsergymEnv` | `agentlab2/cubes/miniwob_adapter.py` | New adapter class |
| Add `extra_tool_configs` to `TaskConfig` and `Task` | `cube-standard/src/cube/task.py` | New field on both + `_extra_tools` PrivateAttr + dispatch logic in `step()` |
