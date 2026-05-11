# cube-tools

Optional tool implementations for [cube-standard](../README.md).

`cube-standard` defines some `Protocol` for each benchmark domain but ships no concrete tool implementation. This folder contains packages that implement these protocols and can be installed independently.

For instance, web browsing benchmarks (MiniWob, WorkArena, WebArena) can use the `cube-browser-tool` package, which provides `BrowsergymTool` and `PlaywrightTool` -- both satisfying the `AbstractBrowserTool` protocol defined in `cube-standard`.

## Packages

| Package | PyPI name | Description |
| --- | --- | --- |
| [`cube-bash-tool/`](cube-bash-tool/) | `cube-bash-tool` | Container-backed bash tool for SWE-style benchmarks |
| [`cube-browser-tool/`](cube-browser-tool/) | `cube-browser-tool` | BrowserGym and Playwright concrete browser tools |
| [`cube-computer-tool/`](cube-computer-tool/) | `cube-computer-tool` | Generic desktop computer tool for VM-based benchmarks |
| [`cube-web-tool/`](cube-web-tool/) | `cube-web-tool` | Web search (Brave) and web fetch+extract tools |

## Usage

### Example usage for web benchmark cubes

Web benchmark cubes (MiniWob, WorkArena, WebArena, …) declare `cube-browser-tool` as an
**optional** dependency:

```bash
# Just the benchmark — bring your own tool
pip install cube-miniwob

# Benchmark + bundled browser tool (quick start, stress test)
pip install cube-miniwob[browser]
```

### Example usage for VM-based desktop benchmark cubes

Desktop benchmark cubes (OSWorld, …) use `cube-computer-tool` with a live VM handle:

```python
from cube_computer_tool import ComputerConfig, ActionSpace

# computer_13: 13 mouse/keyboard primitives
config = ComputerConfig(action_space=ActionSpace.COMPUTER_13)
tool = config.make(vm=vm)  # vm is a cube.vm.VM handle

# pyautogui: execute Python/pyautogui code in the VM
config = ComputerConfig(action_space=ActionSpace.PYAUTOGUI)
tool = config.make(vm=vm)

# Deferred VM attach (for deferred-launch patterns)
tool = config.make()
tool.attach_vm(vm)
```

## Adding a new tool package

1. Create a new subdirectory here (e.g. `cube-terminal-tool/`).
2. Add a `pyproject.toml` with `cube-standard` as a dependency.
3. Implement the relevant protocol from `cube-standard` (`AbstractBrowserTool` for web benchmarks) in your package.
4. Add a row to the table above.
