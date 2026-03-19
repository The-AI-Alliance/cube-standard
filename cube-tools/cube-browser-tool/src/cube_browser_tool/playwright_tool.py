"""Sync Playwright browser tool for cube-standard benchmarks.

Provides ``PlaywrightConfig`` and ``SyncPlaywrightTool``, a concrete
implementation of the ``BrowserTool`` protocol defined in
``cube-standard/src/cube/tool.py``.

``AsyncPlaywrightTool`` will be added in Phase 2.
"""

import asyncio
import logging
import time
from io import BytesIO
from typing import Any, Literal

from cube.core import Action, Content, Observation, StepError
from cube.tool import AsyncTool, BrowserTool, ToolConfig
from cube_browser_playwright import PlaywrightSession, PlaywrightSessionConfig
from PIL import Image
from playwright.async_api import Error as AsyncError
from playwright.async_api import Page as AsyncPage
from playwright.async_api import async_playwright
from playwright.sync_api import Error as SyncError
from playwright.sync_api import Page as SyncPage
from pydantic import Field, field_validator

from cube_browser_tool._utils import flatten_axtree, prune_html
from cube_browser_tool.action_spaces import BrowserActionSpace

logger = logging.getLogger(__name__)


class PlaywrightConfig(ToolConfig):
    """Configuration for ``SyncPlaywrightTool``.

    Parameters
    ----------
    browser : PlaywrightSessionConfig
        Browser launch parameters (headless, viewport, etc.).
    max_wait : int
        Maximum number of seconds ``browser_wait`` may pause. Default ``60``.
    use_html : bool
        Include page HTML in ``page_obs()``. Default ``True``.
    use_axtree : bool
        Include accessibility tree in ``page_obs()``. Default ``False``.
    use_screenshot : bool
        Include a screenshot in ``page_obs()``. Default ``True``.
    prune_html : bool
        Strip noise from HTML before including it in ``page_obs()``. Default ``True``.
    """

    browser: PlaywrightSessionConfig = Field(default_factory=PlaywrightSessionConfig)
    max_wait: int = 60
    use_html: bool = True
    use_axtree: bool = False
    use_screenshot: bool = True
    prune_html: bool = True

    @field_validator("max_wait")
    @classmethod
    def _validate_max_wait(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_wait must be positive")
        return v

    def make(self, container=None) -> "SyncPlaywrightTool":
        session = self.browser.make()
        return SyncPlaywrightTool(config=self, session=session)


class SyncPlaywrightTool(BrowserTool, BrowserActionSpace):
    """Synchronous Playwright browser tool.

    Implements the ``BrowserTool`` contract (session, noop, page_obs) and
    ``BrowserActionSpace`` (CSS-selector browser actions). Browser lifecycle
    is delegated to an injected ``PlaywrightSession``.

    ``execute_action`` is overridden to append ``page_obs()`` after every
    successful action so the agent always receives the updated page state.

    Call ``reset()`` between episodes to get a fresh page without restarting
    the browser process. Call ``close()`` to release all Playwright resources.
    """

    def __init__(self, config: PlaywrightConfig, session: PlaywrightSession) -> None:
        self.config = config
        self._session = session

    @property
    def session(self) -> PlaywrightSession:
        return self._session

    @property
    def page(self) -> SyncPage:
        return self.session.page

    # ------------------------------------------------------------------
    # Tool lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Close the current page and open a fresh one (clears cookies and state)."""
        self._session.stop()
        self._session = self.config.browser.make()

    def close(self) -> None:
        """Release all Playwright resources via the session."""
        self._session.stop()

    # ------------------------------------------------------------------
    # Action dispatch override — appends page_obs() after every action
    # ------------------------------------------------------------------

    def execute_action(self, action: Action) -> Observation | StepError:
        result = super().execute_action(action)
        if isinstance(result, StepError):
            return result
        return result + self.page_obs()

    # ------------------------------------------------------------------
    # Task-internal methods (not agent-facing actions)
    # ------------------------------------------------------------------

    def goto(self, url: str) -> None:
        """Navigate to a URL and wait for the page to load.

        Parameters
        ----------
        url : str
            Fully qualified URL to navigate to.
        """
        self.page.goto(url)

    def evaluate_js(self, js: str) -> Any:
        """Evaluate a JavaScript expression in the page context.

        Parameters
        ----------
        js : str
            JavaScript expression or arrow function (e.g. ``"() => document.title"``).

        Returns
        -------
        Any
            Whatever the JS expression evaluates to (str, int, list, dict, bool, None).
        """
        result = self.page.evaluate(js)
        logger.debug("JS result: %s", result)
        return result

    def _wait_dom_loaded(self) -> None:
        for page in self.session.context.pages:
            try:
                page.wait_for_load_state("domcontentloaded", timeout=3000)
            except SyncError as e:
                logger.debug("page.wait_dom_loaded failed: %s", e)
            for frame in page.frames:
                try:
                    frame.wait_for_load_state("domcontentloaded", timeout=3000)
                except SyncError as e:
                    logger.debug("frame.wait_dom_loaded failed: %s", e)

    def page_obs(self) -> Observation:
        """Capture the current page state as an Observation.

        Content included depends on the config flags ``use_html``,
        ``use_axtree``, and ``use_screenshot``.
        """
        self._wait_dom_loaded()

        contents = []
        if self.config.use_html:
            html = self.page_html()
            name = "pruned_html" if self.config.prune_html else "html"
            data = prune_html(html) if self.config.prune_html else html
            contents.append(Content.from_data(data, name=name))
        if self.config.use_axtree:
            contents.append(Content.from_data(self.page_axtree(), name="axtree_txt"))
        if self.config.use_screenshot:
            contents.append(Content.from_data(self.page_screenshot(), name="screenshot"))
        return Observation(contents=contents)

    def page_html(self) -> str:
        """Return the raw HTML of the current page."""
        return self.page.content()

    def page_screenshot(self) -> Image.Image:
        """Return a PIL screenshot of the current page."""
        return Image.open(BytesIO(self.page.screenshot()))

    def page_axtree(self) -> str:
        """Return the accessibility tree of the current page as indented text."""
        return flatten_axtree(self.page.accessibility.snapshot())

    # ------------------------------------------------------------------
    # Agent-facing actions (implement BrowserActionSpace contract)
    # ------------------------------------------------------------------

    def browser_click(self, selector: str) -> None:
        """Click on an element specified by CSS selector.

        Parameters
        ----------
        selector : str
            CSS selector of the element to click.
        """
        self.page.click(selector, timeout=3000, strict=True)

    def browser_type(self, selector: str, text: str) -> None:
        """Type text into an element specified by CSS selector.

        Parameters
        ----------
        selector : str
            CSS selector of the element to type into.
        text : str
            Text to type.
        """
        self.page.type(selector, text)

    def browser_press_key(self, key: str) -> None:
        """Press a keyboard key.

        Parameters
        ----------
        key : str
            Key name as accepted by Playwright (e.g. 'Enter', 'Tab', 'Escape').
        """
        self.page.keyboard.press(key)

    def browser_hover(self, selector: str) -> None:
        """Hover over an element specified by CSS selector.

        Parameters
        ----------
        selector : str
            CSS selector of the element to hover over.
        """
        self.page.hover(selector, timeout=3000, strict=True)

    def browser_drag(self, from_selector: str, to_selector: str) -> None:
        """Drag an element to another element using CSS selectors.

        Parameters
        ----------
        from_selector : str
            CSS selector of the element to drag.
        to_selector : str
            CSS selector of the drop target.
        """
        from_elem = self.page.locator(from_selector)
        from_elem.hover(timeout=500)
        self.page.mouse.down()
        try:
            to_elem = self.page.locator(to_selector)
            to_elem.hover(timeout=500)
        except Exception:
            self.page.mouse.up()
            raise
        self.page.mouse.up()

    def browser_select_option(self, selector: str, value: str) -> None:
        """Select an option in a <select> element specified by CSS selector.

        Parameters
        ----------
        selector : str
            CSS selector of the <select> element.
        value : str
            Option value to select.
        """
        self.page.select_option(selector, value)

    def browser_mouse_click_xy(self, x: int, y: int) -> None:
        """Click at an absolute (x, y) coordinate on the page.

        Parameters
        ----------
        x : int
            Horizontal coordinate in pixels from the left edge of the viewport.
        y : int
            Vertical coordinate in pixels from the top edge of the viewport.
        """
        self.page.mouse.click(x, y, delay=100)

    def browser_scroll(self, selector: str, direction: Literal["up", "down", "left", "right"], amount: int) -> None:
        """Scroll an element in the specified direction.

        Parameters
        ----------
        selector : str
            CSS selector of the element to scroll.
        direction : {'up', 'down', 'left', 'right'}
            Direction to scroll.
        amount : int
            Number of pixels to scroll.
        """
        elem = self.page.locator(selector).first
        elem.scroll_into_view_if_needed()
        box = elem.bounding_box()
        if box is None:
            raise ValueError(
                f"browser_scroll: element '{selector}' has no bounding box (it may be hidden or have zero dimensions)."
            )
        self.page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        delta_x = {"left": -amount, "right": amount}.get(direction, 0)
        delta_y = {"up": -amount, "down": amount}.get(direction, 0)
        self.page.mouse.wheel(delta_x, delta_y)

    def browser_back(self) -> None:
        """Navigate back in the browser history."""
        self.page.go_back()

    def browser_forward(self) -> None:
        """Navigate forward in the browser history."""
        self.page.go_forward()

    def browser_wait(self, seconds: int) -> None:
        """Wait for a number of seconds before the next action.

        Parameters
        ----------
        seconds : int
            Number of seconds to wait (capped at the tool's ``max_wait``).
        """
        time.sleep(min(seconds, self.config.max_wait))

    def noop(self) -> None:
        """No-op: take no action and return the current page state."""
        pass


class AsyncPlaywrightTool(AsyncTool, BrowserActionSpace):
    """Fully asynchronous Playwright tool using playwright.async_api."""

    def __init__(self, config: PlaywrightConfig) -> None:
        super().__init__()
        self.config = config
        self._apw = None
        self._abrowser = None
        self._page: AsyncPage = None  # type: ignore

    async def initialize(self):
        self._apw = await async_playwright().start()
        self._abrowser = await self._apw.chromium.launch(chromium_sandbox=True, **self.config.browser.pw_extra_kwargs)
        self._page = await self._abrowser.new_page()

    async def execute_action(self, action: Action) -> Observation | StepError:
        result = await super().execute_action(action)
        if isinstance(result, StepError):
            return result
        try:
            result += await self.page_obs()
        except Exception as e:
            return StepError.from_exception(e)
        return result

    async def browser_press_key(self, key: str):
        """Press a key on the keyboard."""
        await self._page.keyboard.press(key)

    async def browser_type(self, selector: str, text: str):
        """Type text into the focused element."""
        await self._page.type(selector, text)

    async def browser_click(self, selector: str):
        """Click on a selector."""
        await self._page.click(selector, timeout=3000, strict=True)

    async def browser_drag(self, from_selector: str, to_selector: str):
        """Drag and drop from one selector to another."""
        from_elem = self._page.locator(from_selector)
        await from_elem.hover(timeout=500)
        await self._page.mouse.down()

        to_elem = self._page.locator(to_selector)
        await to_elem.hover(timeout=500)
        await self._page.mouse.up()

    async def browser_hover(self, selector: str):
        """Hover over a given element."""
        await self._page.hover(selector, timeout=3000, strict=True)

    async def browser_select_option(self, selector: str, value: str):
        """Select an option from a given element."""
        await self._page.select_option(selector, value)

    async def browser_mouse_click_xy(self, x: int, y: int):
        """Click at a given x, y coordinate using the mouse."""
        await self._page.mouse.click(x, y, delay=100)

    async def browser_wait(self, seconds: int):
        """Wait for a given number of seconds, up to max_wait."""
        await asyncio.sleep(min(seconds, self.config.max_wait))

    async def browser_back(self):
        """Navigate back in browser history."""
        await self._page.go_back()

    async def browser_forward(self):
        """Navigate forward in browser history."""
        await self._page.go_forward()

    async def noop(self):
        """No operation action."""
        pass

    async def evaluate_js(self, js: str):
        js_result = await self._page.evaluate(js)
        logger.info(f"JS result: {js_result}")
        return js_result

    async def goto(self, url: str):
        await self._page.goto(url)

    async def page_html(self) -> str:
        return await self._page.content()

    async def page_screenshot(self) -> Image.Image:
        scr_bytes = await self._page.screenshot()
        return Image.open(BytesIO(scr_bytes))

    async def page_axtree(self) -> str:
        axtree = await self._page.accessibility.snapshot()
        return flatten_axtree(axtree)

    async def _wait_dom_loaded(self) -> None:
        for context in self._abrowser.contexts:  # type: ignore
            for page in context.pages:
                try:
                    await page.wait_for_load_state("domcontentloaded", timeout=3000)
                except AsyncError as e:
                    logger.debug("page.wait_dom_loaded failed: %s", e)
                for frame in page.frames:
                    try:
                        await frame.wait_for_load_state("domcontentloaded", timeout=3000)
                    except AsyncError as e:
                        logger.debug("frame.wait_dom_loaded failed: %s", e)

    async def page_obs(self) -> Observation:
        await self._wait_dom_loaded()
        obs = Observation()
        if self.config.use_html:
            html = await self.page_html()
            if self.config.prune_html:
                obs.contents.append(Content.from_data(prune_html(html), name="pruned_html"))
            else:
                obs.contents.append(Content.from_data(html, name="html"))
        if self.config.use_axtree:
            obs.contents.append(Content.from_data(await self.page_axtree(), name="axtree_txt"))
        if self.config.use_screenshot:
            obs.contents.append(Content.from_data(await self.page_screenshot(), name="screenshot"))
        return obs

    async def close(self):
        await self._page.close()
        await self._abrowser.close()  # type: ignore
        await self._apw.stop()  # type: ignore
