# cube-browser-tool

Concrete browser tool implementations for [cube-standard](../../README.md) benchmarks.

`cube-standard` defines the [`AbstractBrowserTool`](../../src/cube/tools/browser.py)
protocol. This package provides ready-to-use implementations:

| Class | Backend | Phase |
|---|---|---|
| `SyncPlaywrightTool` | Playwright (sync) | ✅ done |
| `AsyncPlaywrightTool` | Playwright (async) | 🔜 Phase 2 |
| `BgymTool` | BrowserGym | 🔜 Phase 3 |

Web benchmark cubes (MiniWob, WorkArena, WebArena, …) declare this package as an
**optional** dependency — bring your own tool, or install this one for a quick start.

## Getting started

```bash
# Install the package and its dev dependencies, then download Chromium
make install

# Run tests (integration tests are skipped automatically if Playwright is not installed)
make test
```

## Usage

```python
from cube_browser_tool import PlaywrightConfig, SyncPlaywrightTool

# Create a tool from config
config = PlaywrightConfig(headless=True, use_screenshot=True, use_html=True)
tool = config.make()

# Task-internal navigation
tool.reset()
tool.goto("https://example.com")
goal = tool.evaluate_js("() => document.title")

# Capture the current page state
obs = tool.page_obs()

# Execute an agent action
from cube.core import Action
action = Action(name="browser_click", arguments={"selector": "#submit-btn"})
result = tool.execute_action(action)  # returns Observation (with page_obs appended)

# Inspect available actions
for schema in tool.action_set:
    print(schema.name, schema.description)

tool.close()
```

## Using with a web benchmark cube

```python
from cube_browser_tool import PlaywrightConfig

# Pass the config to a benchmark; the cube calls tool.goto(), tool.evaluate_js(), etc.
from miniwob_cube import MiniWobBenchmark

benchmark = MiniWobBenchmark(tool_config=PlaywrightConfig(headless=True))
for task_config in benchmark.get_task_configs():
    task = task_config.make()
    obs, info = task.reset()
    ...
    task.close()
```

## Action spaces

`BrowserActionSpace` and `BidBrowserActionSpace` are exported for implementing
custom tools that share the same action contract:

```python
from cube_browser_tool import BrowserActionSpace
from cube.tool import Tool

class MyCustomTool(Tool, BrowserActionSpace):
    def browser_click(self, selector: str) -> None:
        ...  # your implementation
    # implement the remaining abstract methods
```

## Optional BrowserGym support (Phase 3)

```bash
pip install cube-browser-tool[bgym]
```

```python
from cube_browser_tool.bgym_tool import BgymToolConfig, BgymTool
```
