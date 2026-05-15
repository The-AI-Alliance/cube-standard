# cube-tools

Optional tool implementations for [cube-standard](../README.md).

`cube-standard` defines some `Protocol` for each benchmark domain but ships no concrete tool implementation. This folder contains packages that implement these protocols and can be installed independently.

For instance, web browsing benchmarks (MiniWob, WorkArena, WebArena) can use the `cube-browser-tool` package, which provides `BrowsergymTool` and `PlaywrightTool` -- both satisfying the `AbstractBrowserTool` protocol defined in `cube-standard`.

## When does a tool belong here?

A generalist tool gets its own `cube-tools/cube-<name>-tool/` sub-package when it pulls a non-trivial runtime dep (Playwright, BrowserGym, PyAutoGUI, MCP SDK, …); otherwise it stays in `cube-standard/src/cube/tools/`. Cube-specific tools live in their own cube package. See [tool/spec.md § Packaging conventions](../openspec/specs/tool/spec.md#packaging-conventions).

## Packages

| Package | PyPI name | Description |
| --- | --- | --- |
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

Desktop benchmark cubes (OSWorld, …) use `cube-computer-tool`. The VM is
provisioned through an `InfraConfig` (`VMResourceConfig` → `LocalInfraConfig` /
`cube-infra-aws` / `cube-infra-azure`), which returns a `ResourceHandle`; the tool
connects to the in-VM guest agent via `attach_endpoint(handle.endpoint)`:

```python
from cube_computer_tool import ComputerConfig, ActionSpace

# computer_13: 13 mouse/keyboard primitives
config = ComputerConfig(action_space=ActionSpace.COMPUTER_13)

# pyautogui: execute Python/pyautogui code in the VM
# config = ComputerConfig(action_space=ActionSpace.PYAUTOGUI)

# The tool is constructed without a live connection, then attached once the VM
# is launched (deferred-launch pattern that fits the InfraConfig lifecycle).
tool = config.make()
tool.attach_endpoint(handle.endpoint)  # handle = infra.launch(vm_resource)
```

## Adding a new tool package

1. Create a new subdirectory here (e.g. `cube-terminal-tool/`).
2. Add a `pyproject.toml` with `cube-standard` as a dependency.
3. Implement the relevant protocol from `cube-standard` (`AbstractBrowserTool` for web benchmarks) in your package.
4. Add a row to the table above.

If you're authoring a brand-new abstract tool base (the next `BrowserTool` / `TerminalTool`), the abstract carries the task-side contract only — no `@tool_action` methods, no enumerated action space. See [tool/spec.md § Contracts for implementers](../openspec/specs/tool/spec.md).
