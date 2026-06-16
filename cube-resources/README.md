# cube-resources

Optional resource implementations for [cube-standard](../README.md).

`cube-standard` defines abstract contracts (`BrowserConfig`, `BrowserSession`, `InfraConfig`) for shared infrastructure, but ships no concrete cloud implementation. This folder contains packages that implement these contracts and can be installed independently.

For instance, web benchmarks that need a running browser can use the `cube-browser-playwright` package, which provides `PlaywrightSessionConfig` and `PlaywrightSession` — both satisfying the `BrowserConfig` / `BrowserSession` abstractions defined in `cube-standard`. VM-based benchmarks declare a `VMResourceConfig` and provision it through an `InfraConfig` — `LocalInfraConfig` (built into `cube-standard`) for local QEMU/qcow2, or `cube-infra-aws` / `cube-infra-azure` for cloud.

## What is a Cube Resource?

A **Resource** is a piece of shared infrastructure (e.g. a running browser instance), as opposed to a **Tool** (which executes agent actions against that infrastructure).

The pattern is: **Config → Session**

- `BrowserConfig` — a serializable factory. Call `make()` to launch a browser and get a live handle.
- `BrowserSession` — the live handle. Exposes `cdp_url` (Chrome DevTools Protocol URL) and `stop()`.

This separation enables three use cases:

1. **Cross-process sharing** — serialize the config, pass `cdp_url` to a Ray worker or subprocess, and reconnect via `pw.chromium.connect_over_cdp(session.cdp_url)`.
2. **Cross-backend access** — the task sets up the page via Playwright; the tool can attach using a different backend (Puppeteer, raw CDP) through the same `cdp_url`.
3. **CUA (future)** — OS-level interaction (screenshot + keyboard/mouse) without a browser protocol. The session identifies the browser window at the OS level instead.

## Packages

| Package | PyPI name | Description |
| --- | --- | --- |
| [`cube-browser-playwright/`](cube-browser-playwright/) | `cube-browser-playwright` | Chromium browser session via Playwright |
| [`cube-chat/`](cube-chat/) | `cube-chat` | `ChatSession` backing for chat-style tasks |
| [`cube-infra-aws/`](cube-infra-aws/) | `cube-infra-aws` | AWS `InfraConfig` — provisions VMs (qcow2 → AMI) and Docker stacks |
| [`cube-infra-azure/`](cube-infra-azure/) | `cube-infra-azure` | Azure `InfraConfig` — provisions VMs (qcow2 → Gallery image) and Docker stacks |
| [`cube-infra-daytona/`](cube-infra-daytona/) | `cube-infra-daytona` | Daytona `InfraConfig` — container provisioning |
| [`cube-infra-modal/`](cube-infra-modal/) | `cube-infra-modal` | Modal `InfraConfig` — container provisioning |
| [`cube-infra-toolkit/`](cube-infra-toolkit/) | `cube-infra-toolkit` | Toolkit/EAI `InfraConfig` — container provisioning |

## Usage

### Launching a browser session

```python
from cube_browser_playwright import PlaywrightSessionConfig

config = PlaywrightSessionConfig(headless=True)
session = config.make()

# Use the CDP URL to attach from any backend
print(session.cdp_url)  # e.g. http://localhost:54321

# Direct Playwright access is also available
session.page.goto("https://example.com")

# Always stop the session when done
session.stop()
```

### Launching a VM

A benchmark declares **what** VM it needs with a `cube.resource.VMResourceConfig`; an
`InfraConfig` decides **how** to provision it. `LocalInfraConfig` (built into
`cube-standard`) boots a local QEMU VM from a qcow2 image; `cube-infra-aws` /
`cube-infra-azure` provision the same `VMResourceConfig` as a cloud VM. Either way,
`provision()` materializes the L1 image and `launch()` returns a `ResourceHandle`
whose `.endpoint` is the base URL of the in-VM HTTP guest agent.

#### Local QEMU (qcow2)

```python
from cube.infra_local import LocalInfraConfig
from cube.resource import VMResourceConfig

resource = VMResourceConfig(
    name="osworld-ubuntu-vm",
    source_url="https://huggingface.co/datasets/xlangai/ubuntu_osworld/resolve/main/Ubuntu.qcow2.zip",
)

infra = LocalInfraConfig()
infra.provision(resource)            # downloads + converts qcow2; idempotent (L1)

handle = infra.launch(resource)      # boots a copy-on-write overlay VM (L3)
print(handle.endpoint)               # e.g. http://127.0.0.1:54321

# Tear the VM down and release the overlay + tunnels
handle.close()
```

#### Cloud (AWS)

```python
from cube_infra_aws import AWSInfraConfig
from cube.resource import VMResourceConfig

resource = VMResourceConfig(
    name="osworld-ubuntu-vm",
    source_url="https://huggingface.co/datasets/xlangai/ubuntu_osworld/resolve/main/Ubuntu.qcow2.zip",
)

infra = AWSInfraConfig()             # region / VPC / S3 bucket auto-discovered
infra.provision(resource)            # qcow2 → VHD → AMI; idempotent (L1)

handle = infra.launch(resource)      # EC2 instance + SSH tunnel (L3)
print(handle.endpoint)               # e.g. http://localhost:54321
handle.close()
```

`cube-infra-azure` follows the same `provision()` / `launch()` shape — see its
README for the required resource group / storage / gallery prerequisites.

In practice benchmarks don't call `provision()` / `launch()` directly:
`BenchmarkConfig.make(infra)` provisions every entry in `config.resources` and the
task layer launches the per-task VM. Benchmarks that need the base qcow2 image
auto-downloaded (e.g. OSWorld) just set `source_url` on their `VMResourceConfig`.

## Adding a new resource package

1. Create a new subdirectory here (e.g. `cube-browser-cua/`).
2. Add a `pyproject.toml` with `cube-standard` as a dependency.
3. Implement the relevant abstract contract from `cube-standard` (`BrowserConfig` / `BrowserSession` for browser resources, `InfraConfig` for a new VM/container provisioner) in your package.
4. Add a row to the table above.
