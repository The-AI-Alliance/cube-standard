"""Abstract browser session contract for cube-standard tools.

A BrowserSession is a handle to a running browser instance, designed to support
three use cases:

  1. Cross-process (same computer): Pass the session to a Ray worker or subprocess.
     TODO: Implement __getstate__/__setstate__ to drop live objects and reconnect
     via cdp_url using pw.chromium.connect_over_cdp(). The cdp_url is already available.

  2. Cross-backend: The task sets up the environment via Playwright (e.g. WorkArena's
     setup(page)), while the tool acts via a different protocol (Puppeteer, raw CDP).
     PlaywrightSession.cdp_url is the shared reference — any backend can attach to it:
         pw.chromium.connect_over_cdp(session.cdp_url)       # Playwright
         connect(browserURL=session.cdp_url)                  # Puppeteer/Pyppeteer

  3. CUA (Computer Use Agent): The tool bypasses the browser protocol and acts at the
     OS level (screenshot + keyboard/mouse). No CDP needed; the session identifies the
     browser window at the OS level instead.
     TODO: CUASession — store PID and/or DISPLAY env var (Linux) / window handle (macOS).

BrowserConfig is the serializable factory for creating a BrowserSession.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from cube.core import TypedBaseModel

if TYPE_CHECKING:
    from playwright.sync_api import BrowserContext, Page


class BrowserConfig(TypedBaseModel, ABC):
    """Abstract serializable config for a browser session.

    Call make() to launch a browser and get a live BrowserSession. The config holds
    all parameters needed to reproduce the launch and must be fully serializable.

    Subclasses:
    - PlaywrightSessionConfig: Chromium via Playwright (in cube-resources)
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

    All sessions must implement get_playwright_session() — Playwright is the standard
    interface for browser interaction in this codebase. Non-Playwright backends (e.g.
    CUASession) connect lazily via CDP: pw.chromium.connect_over_cdp(self.cdp_url).

    Subclasses:
    - PlaywrightSession: owns Playwright objects directly; cdp_url always set (cube-resources)
    # Future: CUASession — identified via OS process PID and/or Display env var;
    #   get_playwright_session() connects via pw.chromium.connect_over_cdp(cdp_url)
    """

    @abstractmethod
    def get_playwright_session(self) -> tuple[Page, BrowserContext]:
        """Return a live Playwright (page, context) for this browser.

        For Playwright-native sessions this returns the live objects directly.
        For other backends (e.g. CUASession) this connects via CDP lazily:
            pw.chromium.connect_over_cdp(self.cdp_url)
        """
        ...

    @abstractmethod
    def stop(self) -> None:
        """Close the browser and release all resources."""
        ...
