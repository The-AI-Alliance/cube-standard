"""Sync and async Playwright browser tools for cube-standard benchmarks.

Provides ``PlaywrightConfig`` / ``SyncPlaywrightTool`` and
``AsyncPlaywrightConfig`` / ``AsyncPlaywrightTool``, concrete implementations
of the ``BrowserTool`` / ``AsyncBrowserTool`` abstractions.
"""

import asyncio
import logging
import time
from io import BytesIO
from typing import Any, Literal

from cube.container import Container
from cube.core import Action, Content, Observation, StepError
from cube.tool import AsyncToolConfig, ToolConfig
from cube.tools.browser import AsyncBrowserTool, BrowserTool
from cube_browser_playwright import (
    AsyncPlaywrightSession,
    AsyncPlaywrightSessionConfig,
    PlaywrightSession,
    PlaywrightSessionConfig,
)
from PIL import Image
from playwright.async_api import Error as AsyncError
from playwright.async_api import Page as AsyncPage
from playwright.sync_api import Error as SyncError
from playwright.sync_api import Page as SyncPage
from pydantic import Field, field_validator

from cube_browser_tool._utils import flatten_axtree, prune_html
from cube_browser_tool.action_spaces import AsyncBrowserActionSpace, BrowserActionSpace

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
    obs_max_retries: int = 3

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
        """Close the current page and open a fresh one (clears cookies and state).

        If stop() raises, make() is not called and self._session is left in an
        indeterminate state — the caller should treat the tool as unusable and
        discard it. If make() raises, self._session holds the stopped session
        and any subsequent call will fail.
        """
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
        try:
            obs = self.page_obs()
            if isinstance(result, StepError):
                # Prepend the error as text so the agent can see what went wrong
                # and retry, instead of killing the episode via StepError.
                error_content = Content.from_data(
                    f"Error executing action {action.name}: {result.exception_str}",
                    tool_call_id=action.id,
                )
                obs.contents.insert(0, error_content)
            if action.id and obs.contents:
                obs.contents[0].tool_call_id = action.id
            return obs
        except Exception as e:
            logger.exception("Error while capturing page_obs after action %s", action.name)
            return StepError.from_exception(e)

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
                if frame == page.main_frame:
                    continue
                try:
                    frame.wait_for_load_state("domcontentloaded", timeout=3000)
                except SyncError as e:
                    logger.debug("frame.wait_dom_loaded failed: %s", e)

    def _page_obs(self) -> Observation:
        """Capture the current page state as an Observation.

        Content included depends on the config flags ``use_html``,
        ``use_axtree``, and ``use_screenshot``.
        """
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

    def page_obs(self) -> Observation:
        """Capture the current page state as an Observation.

        Content included depends on the config flags ``use_html``,
        ``use_axtree``, and ``use_screenshot``.
        """
        for attempt in range(1, self.config.obs_max_retries + 1):
            self._wait_dom_loaded()
            try:
                return self._page_obs()
            except SyncError as e:
                if attempt == self.config.obs_max_retries:
                    raise e
                logger.warning(
                    "Observation extraction failed (attempt %s/%s): %s",
                    attempt,
                    self.config.obs_max_retries,
                    e,
                )
                time.sleep(0.5)

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


class AsyncPlaywrightConfig(AsyncToolConfig):
    """Configuration for ``AsyncPlaywrightTool``.

    Parameters
    ----------
    browser : AsyncPlaywrightSessionConfig
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
    obs_max_retries : int
        Number of times to retry observation extraction on error. Default ``3``.
    """

    browser: AsyncPlaywrightSessionConfig = Field(default_factory=AsyncPlaywrightSessionConfig)
    max_wait: int = 60
    use_html: bool = True
    use_axtree: bool = False
    use_screenshot: bool = True
    prune_html: bool = True
    obs_max_retries: int = 3

    @field_validator("max_wait")
    @classmethod
    def _validate_max_wait(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_wait must be positive")
        return v

    async def make(self, container: Container | None = None) -> "AsyncPlaywrightTool":
        session = await self.browser.make()
        return AsyncPlaywrightTool(config=self, session=session)


class AsyncPlaywrightTool(AsyncBrowserTool, AsyncBrowserActionSpace):
    """Fully asynchronous Playwright tool using playwright.async_api."""

    def __init__(self, config: AsyncPlaywrightConfig, session: AsyncPlaywrightSession) -> None:
        super().__init__()
        self.config = config
        self._session: AsyncPlaywrightSession = session

    @property
    def session(self) -> AsyncPlaywrightSession:
        return self._session

    @property
    def _page(self) -> AsyncPage:
        return self._session.page

    async def reset(self) -> None:
        """Close the current session and open a fresh one (clears cookies and state).

        If stop() raises, make() is not called and self._session is left in an
        indeterminate state — the caller should treat the tool as unusable and
        discard it. If make() raises, self._session holds the stopped session
        and any subsequent call will fail.
        """
        await self._session.stop()
        self._session = await self.config.browser.make()

    async def close(self) -> None:
        """Release all Playwright resources via the session."""
        await self._session.stop()

    async def execute_action(self, action: Action) -> Observation | StepError:
        result = await super().execute_action(action)
        try:
            obs = await self.page_obs()
            if isinstance(result, StepError):
                # Prepend the error as text so the agent can see what went wrong
                # and retry, instead of killing the episode via StepError.
                error_content = Content.from_data(
                    f"Error executing action {action.name}: {result.exception_str}",
                    tool_call_id=action.id,
                )
                obs.contents.insert(0, error_content)
            if action.id and obs.contents:
                obs.contents[0].tool_call_id = action.id
            return obs
        except Exception as e:
            logger.exception("Error while capturing page_obs after action %s", action.name)
            return StepError.from_exception(e)

    async def browser_press_key(self, key: str) -> None:
        """Press a keyboard key.

        Parameters
        ----------
        key : str
            Key name as accepted by Playwright (e.g. 'Enter', 'Tab', 'Escape').
        """
        await self._page.keyboard.press(key)

    async def browser_type(self, selector: str, text: str) -> None:
        """Type text into an element specified by CSS selector.

        Parameters
        ----------
        selector : str
            CSS selector of the element to type into.
        text : str
            Text to type.
        """
        await self._page.type(selector, text)

    async def browser_click(self, selector: str) -> None:
        """Click on an element specified by CSS selector.

        Parameters
        ----------
        selector : str
            CSS selector of the element to click.
        """
        await self._page.click(selector, timeout=3000, strict=True)

    async def browser_drag(self, from_selector: str, to_selector: str) -> None:
        """Drag an element to another element using CSS selectors.

        Parameters
        ----------
        from_selector : str
            CSS selector of the element to drag.
        to_selector : str
            CSS selector of the drop target.
        """
        from_elem = self._page.locator(from_selector)
        await from_elem.hover(timeout=500)
        await self._page.mouse.down()
        try:
            to_elem = self._page.locator(to_selector)
            await to_elem.hover(timeout=500)
        except Exception:
            await self._page.mouse.up()
            raise
        await self._page.mouse.up()

    async def browser_hover(self, selector: str) -> None:
        """Hover over an element specified by CSS selector.

        Parameters
        ----------
        selector : str
            CSS selector of the element to hover over.
        """
        await self._page.hover(selector, timeout=3000, strict=True)

    async def browser_select_option(self, selector: str, value: str) -> None:
        """Select an option in a <select> element specified by CSS selector.

        Parameters
        ----------
        selector : str
            CSS selector of the <select> element.
        value : str
            Option value to select.
        """
        await self._page.select_option(selector, value)

    async def browser_mouse_click_xy(self, x: int, y: int) -> None:
        """Click at an absolute (x, y) coordinate on the page.

        Parameters
        ----------
        x : int
            Horizontal coordinate in pixels from the left edge of the viewport.
        y : int
            Vertical coordinate in pixels from the top edge of the viewport.
        """
        await self._page.mouse.click(x, y, delay=100)

    async def browser_scroll(
        self, selector: str, direction: Literal["up", "down", "left", "right"], amount: int
    ) -> None:
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
        elem = self._page.locator(selector).first
        await elem.scroll_into_view_if_needed()
        box = await elem.bounding_box()
        if box is None:
            raise ValueError(
                f"browser_scroll: element '{selector}' has no bounding box (it may be hidden or have zero dimensions)."
            )
        await self._page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        delta_x = {"left": -amount, "right": amount}.get(direction, 0)
        delta_y = {"up": -amount, "down": amount}.get(direction, 0)
        await self._page.mouse.wheel(delta_x, delta_y)

    async def browser_wait(self, seconds: int) -> None:
        """Wait for a number of seconds before the next action.

        Parameters
        ----------
        seconds : int
            Number of seconds to wait (capped at the tool's ``max_wait``).
        """
        await asyncio.sleep(min(seconds, self.config.max_wait))

    async def browser_back(self) -> None:
        """Navigate back in browser history."""
        await self._page.go_back()

    async def browser_forward(self) -> None:
        """Navigate forward in browser history."""
        await self._page.go_forward()

    async def noop(self) -> None:
        """No operation action."""
        pass

    async def evaluate_js(self, js: str) -> Any:
        js_result = await self._page.evaluate(js)
        logger.debug("JS result: %s", js_result)
        return js_result

    async def goto(self, url: str) -> None:
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
        for page in self._session.context.pages:
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=3000)
            except AsyncError as e:
                logger.debug("page.wait_dom_loaded failed: %s", e)
            for frame in page.frames:
                if frame == page.main_frame:
                    continue
                try:
                    await frame.wait_for_load_state("domcontentloaded", timeout=3000)
                except AsyncError as e:
                    logger.debug("frame.wait_dom_loaded failed: %s", e)

    async def _page_obs(self) -> Observation:
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

    async def page_obs(self) -> Observation:
        for attempt in range(1, self.config.obs_max_retries + 1):
            await self._wait_dom_loaded()
            try:
                return await self._page_obs()
            except AsyncError as e:
                if attempt == self.config.obs_max_retries:
                    raise e
                logger.warning(
                    "Observation extraction failed (attempt %s/%s): %s",
                    attempt,
                    self.config.obs_max_retries,
                    e,
                )
                await asyncio.sleep(0.5)
