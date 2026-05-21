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
| `cube-computer-tool` | `ComputerConfig`, `ActionSpace.COMPUTER_13`, `ActionSpace.PYAUTOGUI` | Agent needs mouse + keyboard + screenshot against a VM (desktop automation). | `t = ComputerConfig(action_space=ActionSpace.COMPUTER_13).make()` then `t.attach_endpoint(handle.endpoint)`. The handle comes from `InfraConfig.launch(vm_resource)`. |

## cube-resources/ — shared infrastructure

| Package | Concrete classes | Use when | How to adopt |
|---------|------------------|----------|--------------|
| `cube-browser-playwright` | `PlaywrightSessionConfig`, `PlaywrightSession` | Running Chromium browser session (served via CDP URL). Typically paired with `cube-browser-tool`. | `PlaywrightSessionConfig(headless=True).make()` in `Benchmark._setup()`. Expose `cdp_url` via `self._runtime_context`. |
| `cube-chat` | (see package) | `ChatSession` backing for `cube-chat-tool`. | Parallel pattern to `cube-browser-playwright`. |
| `cube.infra_local` (built into `cube-standard`) | `LocalInfraConfig` | Need a local VM for desktop automation (qcow2 images). | Declare a `VMResourceConfig(name=..., source_url=...)`; `LocalInfraConfig().provision(resource)` downloads/converts the qcow2 and `.launch(resource)` boots a copy-on-write overlay VM. For OSWorld-style cubes just set `source_url` to the base image. |
| `cube-infra-aws` | `AWSInfraConfig` | Cube needs remote cloud VMs (qcow2 → AMI). | `AWSInfraConfig(region=...)` then `infra.provision(resource)` + `infra.launch(resource)` with a `VMResourceConfig`. Requires AWS creds via boto3 default chain. |
| `cube-infra-azure` | `AzureInfraConfig` | Same as above, Azure flavor. | Requires resource group / storage / VNet / NSG / Compute Gallery first — see the package README. |

## Decision tree (phase 2)

1. Agent acts on web pages? → `cube-browser-tool` + `cube-browser-playwright`.
2. Mouse + keyboard on a desktop? → `cube-computer-tool`, with the VM provisioned via a `VMResourceConfig` + `LocalInfraConfig` (local) or `cube-infra-aws/azure` (cloud).
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

Declare a VM resource (provisioning is handled by the injected `InfraConfig`):

```python
from cube.resource import VMResourceConfig

vm_resource = VMResourceConfig(
    name="my-cube-vm",
    source_url="https://.../base.qcow2.zip",  # auto-downloaded by infra.provision()
)
```

## Where new tool code lives

Cube-specific tools live in your cube package. Generalist tools live in `cube-standard` — `src/cube/tools/` (no extra deps) or a `cube-tools/cube-*-tool/` sub-package (heavy deps), never in `cube-harness`. See [tool/spec.md § Packaging conventions](../../../../openspec/specs/tool/spec.md#packaging-conventions).

## Anti-patterns

- Don't fork these packages into the user's cube. Use them as dependencies.
- Don't reimplement `AbstractBrowserTool` — inherit from `SyncPlaywrightTool` or compose.
- Don't write a custom VM provisioner — declare a `VMResourceConfig` and reuse `LocalInfraConfig` / `cube-infra-aws` / `cube-infra-azure`. Only implement a new `InfraConfig` if you truly need a different cloud/hypervisor.
