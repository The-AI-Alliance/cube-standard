"""Abstract browser session contracts for cube-standard tools.

A BrowserSession / AsyncBrowserSession is a handle to a running browser instance,
designed to support three use cases:

  1. Cross-process (same computer): Pass the session to a Ray worker or subprocess.
     TODO: Implement __getstate__/__setstate__ to drop live objects and reconnect
     via cdp_url using pw.chromium.connect_over_cdp(). The cdp_url is already available.

  2. Cross-backend: The task sets up the environment via Playwright (e.g. WorkArena's
     setup(page)), while the tool acts via a different protocol (Puppeteer, raw CDP).
     The cdp_url is the shared reference — any backend can attach to it:
         pw.chromium.connect_over_cdp(session.cdp_url)       # Playwright
         connect(browserURL=session.cdp_url)                  # Puppeteer/Pyppeteer

  3. CUA (Computer Use Agent): The tool bypasses the browser protocol and acts at the
     OS level (screenshot + keyboard/mouse). No CDP needed; the session identifies the
     browser window at the OS level instead.
     TODO: CUASession — store PID and/or DISPLAY env var (Linux) / window handle (macOS).

BrowserConfig / AsyncBrowserConfig are the serializable factories for creating sessions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from cube.core import TypedBaseModel


class BrowserConfig(TypedBaseModel, ABC):
    """Abstract serializable config for a browser session.

    Call make() to launch a browser and get a live BrowserSession. The config holds
    all parameters needed to reproduce the launch and must be fully serializable.

    Subclasses:
    - PlaywrightSessionConfig: Chromium via Playwright (in cube-resources/cube-browser-playwright)
    # Future: CUAConfig — no browser protocol; just OS-level window metadata
    """

    @abstractmethod
    def make(self) -> "BrowserSession":
        """Launch a browser and return a live BrowserSession."""
        ...


class BrowserSession(ABC):
    """Abstract live browser session handle.

    Represents a running browser instance that can be shared across processes and
    backends. See the module docstring for the three design goals this abstraction serves.

    Implementations own the live browser resources and must implement stop().
    The cdp_url property is the cross-backend connection point — any Playwright-compatible
    or CDP-capable client can attach via pw.chromium.connect_over_cdp(session.cdp_url).

    Subclasses:
    - PlaywrightSession: owns Playwright objects directly; cdp_url always set (in cube-resources/cube-browser-playwright)
    # Future: CUASession — identified via OS process PID and/or Display env var
    """

    @property
    @abstractmethod
    def cdp_url(self) -> str | None:
        """The Chrome DevTools Protocol URL for this browser, or None if not available."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Close the browser and release all resources."""
        ...


class AsyncBrowserConfig(TypedBaseModel, ABC):
    """Abstract serializable config for an async browser session.

    Call make() to launch a browser and get a live AsyncBrowserSession. The config holds
    all parameters needed to reproduce the launch and must be fully serializable.

    Subclasses:
    - AsyncPlaywrightSessionConfig: Chromium via async Playwright (in cube-resources/cube-browser-playwright)
    """

    @abstractmethod
    async def make(self) -> "AsyncBrowserSession":
        """Launch a browser and return a live AsyncBrowserSession."""
        ...


class AsyncBrowserSession(ABC):
    """Abstract live async browser session handle.

    Same design goals as BrowserSession but all lifecycle methods are coroutines,
    matching the async Playwright API for high-throughput parallel data collection.

    Subclasses:
    - AsyncPlaywrightSession: owns async Playwright objects directly; cdp_url always set (in cube-resources/cube-browser-playwright)
    """

    @property
    @abstractmethod
    def cdp_url(self) -> str | None:
        """The Chrome DevTools Protocol URL for this browser, or None if not available."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Close the browser and release all resources."""
        ...
