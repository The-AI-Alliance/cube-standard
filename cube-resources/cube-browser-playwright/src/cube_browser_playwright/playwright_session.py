"""Playwright browser session implementation.

Provides PlaywrightSessionConfig and PlaywrightSession, the concrete implementation
of the BrowserConfig / BrowserSession abstractions defined in cube.tools.browser_session.

Chromium is always launched with --remote-debugging-port=0 so cdp_url is always
available for cross-backend access (Puppeteer, raw CDP, etc.).
"""

import logging
import tempfile
import time
from pathlib import Path

from cube.resources.browser_session import BrowserConfig, BrowserSession
from playwright.sync_api import BrowserContext, Page, Playwright, sync_playwright
from pydantic import Field

logger = logging.getLogger(__name__)


def _read_cdp_url(user_data_dir: str) -> str:
    """Read the CDP URL from Chrome's DevToolsActivePort file.

    Chrome writes this file to user_data_dir immediately after binding to the
    debug port. Using --remote-debugging-port=0 lets the OS assign a free port
    atomically, avoiding multiprocessing race conditions.
    """
    port_file = Path(user_data_dir) / "DevToolsActivePort"
    deadline = time.monotonic() + 2.0
    while not port_file.exists():
        if time.monotonic() > deadline:
            raise RuntimeError(f"Chrome did not write DevToolsActivePort to {user_data_dir!r}")
        time.sleep(0.05)
    port = int(port_file.read_text().splitlines()[0])
    return f"http://localhost:{port}"


class PlaywrightSessionConfig(BrowserConfig):
    """Serializable Playwright launch parameters.

    Call make() to start a Chromium browser and get a live PlaywrightSession.
    The browser is always launched with --remote-debugging-port so the returned
    session exposes a cdp_url for cross-backend access.
    """

    headless: bool = True
    viewport: dict[str, int] = Field(default_factory=lambda: {"width": 1280, "height": 720})
    slow_mo: int | None = None
    timeout: int | None = None
    locale: str | None = None
    timezone_id: str | None = None

    # Advanced Playwright options (rarely needed)
    resizeable_window: bool = False
    pw_chromium_kwargs: dict = Field(default_factory=dict)
    pw_context_kwargs: dict = Field(default_factory=dict)
    record_video_dir: str | None = None

    def make(self) -> "PlaywrightSession":
        """Launch a Chromium browser and return a live PlaywrightSession."""
        pw = sync_playwright().start()

        user_data_dir = tempfile.mkdtemp(prefix="cube_harness_")
        args = [
            f"--window-size={self.viewport['width']},{self.viewport['height']}" if self.resizeable_window else None,
            "--disable-features=OverlayScrollbars,ExtendedOverlayScrollbars",
            "--remote-debugging-port=0",
        ]
        context = pw.chromium.launch_persistent_context(
            user_data_dir,
            headless=self.headless,
            slow_mo=self.slow_mo,
            args=[arg for arg in args if arg is not None],
            ignore_default_args=["--hide-scrollbars"],
            no_viewport=True if self.resizeable_window else None,
            viewport=self.viewport if not self.resizeable_window else None,
            record_video_dir=Path(self.record_video_dir) / "task_video" if self.record_video_dir else None,
            record_video_size=self.viewport,
            locale=self.locale,
            timezone_id=self.timezone_id,
            **{**self.pw_chromium_kwargs, **self.pw_context_kwargs},
        )
        if self.timeout is not None:
            context.set_default_timeout(self.timeout)
        page = context.pages[0] if context.pages else context.new_page()
        cdp_url = _read_cdp_url(user_data_dir)
        return PlaywrightSession(playwright=pw, page=page, context=context, cdp_url=cdp_url)


class PlaywrightSession(BrowserSession):
    """Live Playwright browser session.

    Owns the Playwright instance, page, and context launched by PlaywrightSessionConfig.
    Always exposes a cdp_url for cross-backend access.
    """

    def __init__(self, playwright: Playwright, page: Page, context: BrowserContext, cdp_url: str) -> None:
        self._playwright: Playwright = playwright
        self._page: Page = page
        self._context: BrowserContext = context
        self.cdp_url: str = cdp_url

    def get_playwright_session(self) -> tuple[Page, BrowserContext]:
        """Return the live (page, context)."""
        return self._page, self._context

    def stop(self) -> None:
        """Close the context and release all Playwright resources."""
        try:
            self._context.close()
        except Exception as e:
            logger.warning(f"Error closing browser context: {e}")
        try:
            self._playwright.stop()
        except Exception as e:
            logger.warning(f"Error stopping playwright: {e}")
