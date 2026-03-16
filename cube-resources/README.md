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
|---|---|---|
| [`cube-browser-playwright/`](cube-browser-playwright/) | `cube-browser-playwright` | Chromium browser session via Playwright |

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

## Adding a new resource package

1. Create a new subdirectory here (e.g. `cube-browser-cua/`).
2. Add a `pyproject.toml` with `cube-standard` as a dependency.
3. Implement the relevant abstract contract from `cube-standard` (`BrowserConfig` / `BrowserSession` for browser resources) in your package.
4. Add a row to the table above.
