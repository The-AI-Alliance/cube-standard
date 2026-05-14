# Shared packages — reuse before reinventing

Before designing any custom tool or resource, **scan the directories first** — new packages may have been added since this catalog was written:

- `cube-tools/` — agent-facing tool packages (list subdirs to see what's available)
- `cube-resources/` — shared infrastructure packages (browser sessions, VMs, cloud infra)

Read each subdir's `README.md` or `pyproject.toml` to understand what it provides. The catalog below covers the most established packages; treat it as a starting map, not a complete index.

## cube-tools/ — agent-facing actions

| Package | Concrete classes | Use when | How to adopt |
|---------|------------------|----------|--------------|
| `cube-browser-tool` | `SyncPlaywrightTool`, `PlaywrightConfig`; `BgymTool` (optional) | Agent clicks / types / navigates web pages. Implements `AbstractBrowserTool`. | Import `PlaywrightConfig(headless=True, ...)` and pass as the `tool_config` on your benchmark. Subclass only for benchmark-specific actions on top of the 7 standard browser actions. |
| `cube-chat-tool` | (see package) | Chat-style task: send / receive / report-infeasible messages. | Import and use directly. Subclass to customize the action schema. |
| `cube-computer-tool` | `ComputerConfig`, `ActionSpace.COMPUTER_13`, `ActionSpace.PYAUTOGUI` | Agent needs mouse + keyboard + screenshot against a VM (desktop automation). | `ComputerConfig(action_space=ActionSpace.COMPUTER_13).make(vm=vm)`. VM handle comes from the benchmark's resource layer. |

## cube-resources/ — shared infrastructure

| Package | Concrete classes | Use when | How to adopt |
|---------|------------------|----------|--------------|
| `cube-browser-playwright` | `PlaywrightSessionConfig`, `PlaywrightSession` | Running Chromium browser session (served via CDP URL). Typically paired with `cube-browser-tool`. | `PlaywrightSessionConfig(headless=True).make()` in `Benchmark._setup()`. Expose `cdp_url` via `self._runtime_context`. |
| `cube-chat` | (see package) | `ChatSession` backing for `cube-chat-tool`. | Parallel pattern to `cube-browser-playwright`. |
| `cube-vm-backend` | `LocalQEMUVMBackend`, `LocalDockerVMBackend` | Need a local VM for desktop automation (qcow2 images). | For OSWorld-style cubes: subclass `LocalDockerVMBackend` and override `ensure_resource()` to auto-download your base image. Use `LocalQEMUVMBackend` for fastest Linux runs. |
| `cube-infra-aws` | `AWSInfraConfig` | Cube needs remote cloud VMs (qcow2 → AMI). | `AWSInfraConfig(region=...)` then `infra.provision(resource)` + `infra.launch(resource)`. Requires AWS creds via boto3 default chain. |
| `cube-infra-azure` | `AzureInfraConfig` | Same as above, Azure flavor. | Requires resource group / storage / VNet / NSG / Compute Gallery first — see the package README. |

## Decision tree (phase 2)

1. Agent acts on web pages? → `cube-browser-tool` + `cube-browser-playwright`.
2. Mouse + keyboard on a desktop? → `cube-computer-tool`, backed by `cube-vm-backend` (local) or `cube-infra-aws/azure` (cloud).
3. Chat-style task? → `cube-chat-tool` + `cube-chat`.
4. None of the above? → custom tool. Subclass `Tool` (or `AsyncTool`); use `@tool_action`.

## Subclassing patterns

Add custom actions on top of a standard tool:

```python
from cube_browser_tool import SyncPlaywrightTool
from cube.tool import tool_action

class MyBenchmarkBrowserTool(SyncPlaywrightTool):
    @tool_action
    def submit_answer(self, answer: str) -> str:
        ...
```

Custom provisioning for a resource:

```python
from cube_vm_backend import LocalDockerVMBackend

class MyCubeVMBackend(LocalDockerVMBackend):
    def ensure_resource(self):
        # auto-download your base qcow2 image here
        ...
```

## Where new tool code lives (cube-developer rule)

If you build a **cube-specific** tool (one whose only consumer is your benchmark), it lives in your cube package — typically by subclassing a generalist tool from `cube-tools/`. If you build something **generalist** (reusable across cubes), it belongs in `cube-standard` — either in `src/cube/tools/` (no extra deps) or as a new `cube-tools/cube-<name>-tool/` sub-package (heavy deps). **Never put tool code in `cube-harness`.** See [`openspec/specs/tool/spec.md` § Packaging conventions](../../../../openspec/specs/tool/spec.md#packaging-conventions) for the authoritative rule. When in doubt, raise it during the Reflect phase.

## Anti-patterns

- Don't fork these packages into the user's cube. Use them as dependencies.
- Don't reimplement `AbstractBrowserTool` — inherit from `SyncPlaywrightTool` or compose.
- Don't build your own VM backend unless you truly need a different hypervisor.
- Don't put generalist tool code in `cube-harness`. It belongs in `cube-standard`.
