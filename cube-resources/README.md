# cube-resources

Optional resource implementations for [cube-standard](../README.md).

`cube-standard` defines abstract contracts (`BrowserConfig`, `BrowserSession`) for shared infrastructure, but ships no concrete implementation. This folder contains packages that implement these contracts and can be installed independently.

For instance, web benchmarks that need a running browser can use the `cube-browser-playwright` package, which provides `PlaywrightSessionConfig` and `PlaywrightSession` — both satisfying the `BrowserConfig` / `BrowserSession` abstractions defined in `cube-standard`.

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
| [`cube-vm-backend/`](cube-vm-backend/) | `cube-vm-backend` | QEMU/KVM VM backend for desktop-automation benchmarks |

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

`cube-vm-backend` provides two backends. Pick the one that matches your platform:

| Backend | Class | Platform | Notes |
|---------|-------|----------|-------|
| QEMU | `LocalQEMUVMBackend` | Linux (KVM) | Runs the VM natively; fastest on Linux. |
| Docker | `LocalDockerVMBackend` | Linux | Runs QEMU inside `happysixd/osworld-docker`. Reset strategy: stop + remove container, start fresh (~30–60s). |

#### QEMU/KVM (Linux)

```python
from cube_vm_backend import LocalQEMUVMBackend
from cube.vm import VMConfig

backend = LocalQEMUVMBackend(path_to_vm="/path/to/base.qcow2", headless=True)
vm = backend.launch(VMConfig(screen_size=(1920, 1080)))

# vm.endpoint gives the base URL of the in-VM HTTP agent
print(vm.endpoint)  # e.g. http://localhost:5000

# Restore VM to its initial state (overlay reset, ~30s)
vm.restore_snapshot("initial")

# Stop the VM and clean up
vm.stop()
```

#### Docker

```python
from cube_vm_backend import LocalDockerVMBackend
from cube.vm import VMConfig

backend = LocalDockerVMBackend(
    path_to_vm="/path/to/base.qcow2",
    memory="4G",
    cpus=4,
)
vm = backend.launch(VMConfig(screen_size=(1920, 1080)))

# vm.endpoint gives the base URL of the in-VM HTTP agent
print(vm.endpoint)  # e.g. http://localhost:5000

# Reset: stops + removes the container, starts fresh (~30-60s)
vm.restore_snapshot("initial")

# Stop the container and release port reservations
vm.stop()
```

Benchmarks that need the base qcow2 image auto-downloaded (e.g. OSWorld) subclass `LocalDockerVMBackend` and override `ensure_resource()` instead of requiring `path_to_vm` to be set manually.

## Adding a new resource package

1. Create a new subdirectory here (e.g. `cube-browser-cua/`).
2. Add a `pyproject.toml` with `cube-standard` as a dependency.
3. Implement the relevant abstract contract from `cube-standard` (`BrowserConfig` / `BrowserSession` for browser resources, `VMBackend` / `VM` for VM resources) in your package.
4. Add a row to the table above.
