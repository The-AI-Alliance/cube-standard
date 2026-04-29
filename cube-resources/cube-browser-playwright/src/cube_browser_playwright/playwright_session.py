"""Playwright browser session implementation.

Provides PlaywrightSessionConfig and PlaywrightSession, the concrete implementation
of the BrowserConfig / BrowserSession abstractions defined in cube.resources.browser_session.

Chromium is always launched with --remote-debugging-port=0 so cdp_url is always
available for cross-backend access (Puppeteer, raw CDP, etc.).
"""

import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from cube.resources.browser_session import BrowserConfig, BrowserSession
from playwright.sync_api import BrowserContext, Page, Playwright, sync_playwright
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Viewport(BaseModel):
    """Browser viewport dimensions."""

    width: int
    height: int


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
    content = port_file.read_text()
    try:
        port = int(content.splitlines()[0])
    except (IndexError, ValueError) as e:
        raise RuntimeError(
            f"Chrome wrote malformed DevToolsActivePort to {str(port_file)!r}; content={content!r}"
        ) from e
    return f"http://localhost:{port}"


class PlaywrightSessionConfig(BrowserConfig):
    """Serializable Playwright launch parameters.

    Call make() to start a Chromium browser and get a live PlaywrightSession.
    The browser is always launched with --remote-debugging-port so the returned
    session exposes a cdp_url for cross-backend access.
    """

    headless: bool = True
    viewport: Viewport = Field(default_factory=lambda: Viewport(width=1280, height=720))
    slow_mo: int | None = None
    timeout: int | None = None
    locale: str | None = None
    timezone_id: str | None = None

    # Advanced Playwright options (rarely needed)
    resizeable_window: bool = False
    pw_extra_kwargs: dict[str, Any] = Field(default_factory=dict)
    record_video_dir: str | None = None

    def make(self) -> "PlaywrightSession":
        """Launch a Chromium browser and return a live PlaywrightSession."""
        pw = sync_playwright().start()
        context = None
        user_data_dir = tempfile.mkdtemp(prefix="cube_harness_")
        try:
            args = [
                f"--window-size={self.viewport.width},{self.viewport.height}" if self.resizeable_window else None,
                "--disable-features=OverlayScrollbars,ExtendedOverlayScrollbars",
                "--remote-debugging-port=0",
            ]
            viewport_dict = {"width": self.viewport.width, "height": self.viewport.height}
            # Explicit params take precedence over pw_extra_kwargs.
            # Extra kwargs are merged first so that any key collision is won by the explicit value.
            explicit_kwargs: dict[str, Any] = {
                "headless": self.headless,
                "args": [arg for arg in args if arg is not None],
                "ignore_default_args": ["--hide-scrollbars"],
            }
            if self.slow_mo is not None:
                explicit_kwargs["slow_mo"] = self.slow_mo
            if self.resizeable_window:
                explicit_kwargs["no_viewport"] = True
            else:
                explicit_kwargs["viewport"] = viewport_dict
            if self.locale is not None:
                explicit_kwargs["locale"] = self.locale
            if self.timezone_id is not None:
                explicit_kwargs["timezone_id"] = self.timezone_id
            if self.record_video_dir is not None:
                explicit_kwargs["record_video_dir"] = Path(self.record_video_dir) / "task_video"
                explicit_kwargs["record_video_size"] = viewport_dict
            context = pw.chromium.launch_persistent_context(
                user_data_dir, **{**self.pw_extra_kwargs, **explicit_kwargs}
            )
            if self.timeout is not None:
                context.set_default_timeout(self.timeout)
            page = context.pages[0] if context.pages else context.new_page()
            cdp_url = _read_cdp_url(user_data_dir)
            return PlaywrightSession(
                playwright=pw, page=page, context=context, cdp_url=cdp_url, user_data_dir=user_data_dir
            )
        except Exception:
            if context is not None:
                try:
                    context.close()
                except Exception as e:
                    logger.error(
                        "Failed to close browser context during make() cleanup; resources may be leaked: %s", e
                    )
            try:
                pw.stop()
            except Exception as e:
                logger.error("Failed to stop playwright during make() cleanup; resources may be leaked: %s", e)
            try:
                shutil.rmtree(user_data_dir)
            except OSError as e:
                logger.warning("Failed to remove temp profile dir %r during make() cleanup: %s", user_data_dir, e)
            raise


class PlaywrightSession(BrowserSession):
    """Live Playwright browser session.

    Owns the Playwright instance, page, and context launched by PlaywrightSessionConfig.
    Always exposes a cdp_url for cross-backend access.
    """

    def __init__(
        self, playwright: Playwright, page: Page, context: BrowserContext, cdp_url: str, user_data_dir: str
    ) -> None:
        self._playwright: Playwright = playwright
        self._page: Page = page
        self._context: BrowserContext = context
        self._cdp_url: str = cdp_url
        self._user_data_dir: str = user_data_dir

    @property
    def cdp_url(self) -> str:
        return self._cdp_url

    @property
    def page(self) -> Page:
        return self._page

    @property
    def context(self) -> BrowserContext:
        return self._context

    @property
    def playwright(self) -> Playwright:
        return self._playwright

    def stop(self) -> None:
        """Close the context, release all Playwright resources, and remove the temp profile dir."""
        try:
            self._context.close()
        except Exception as e:
            # "Event loop is closed" is expected when stop() is called during interpreter
            # shutdown or after the sync-API event loop has already been torn down — not a leak.
            if "Event loop is closed" not in str(e):
                logger.error("Error closing browser context; resources may be leaked: %s", e)
        try:
            self._playwright.stop()
        except Exception as e:
            if "Event loop is closed" not in str(e):
                logger.error("Error stopping playwright; process may be leaked: %s", e)
        shutil.rmtree(self._user_data_dir, ignore_errors=True)
