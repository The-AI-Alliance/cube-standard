"""Sync Playwright browser tool for cube-standard benchmarks.

Provides ``PlaywrightConfig`` and ``SyncPlaywrightTool``, a concrete
implementation of the ``AbstractBrowserTool`` protocol defined in
``cube-standard/src/cube/tools/browser.py``.

``AsyncPlaywrightTool`` will be added in Phase 2.
"""

import logging
import time
from io import BytesIO
from typing import Any, Literal

from cube.core import Action, Content, Observation, StepError
from cube.tool import Tool, ToolConfig
from PIL import Image
from playwright.sync_api import Page as SyncPage
from playwright.sync_api import sync_playwright
from pydantic import field_validator

from cube_browser_tool._utils import flatten_axtree, prune_html
from cube_browser_tool.action_spaces import BrowserActionSpace

logger = logging.getLogger(__name__)


class PlaywrightConfig(ToolConfig):
    """Configuration for ``SyncPlaywrightTool``.

    Parameters
    ----------
    headless : bool
        Run browser without a visible window. Default ``True``.
    viewport : dict
        Viewport size as ``{"width": int, "height": int}``.
    chromium_sandbox : bool
        Enable Chromium sandbox. Default ``True``. Set to ``False`` inside
        Docker containers.
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
    pw_kwargs : dict
        Extra keyword arguments forwarded to ``chromium.launch()``.
    """

    headless: bool = True
    viewport: dict = {"width": 1280, "height": 720}
    chromium_sandbox: bool = True
    max_wait: int = 60
    use_html: bool = True
    use_axtree: bool = False
    use_screenshot: bool = True
    prune_html: bool = True
    pw_kwargs: dict = {}

    @field_validator("max_wait")
    @classmethod
    def _validate_max_wait(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_wait must be positive")
        return v

    @field_validator("pw_kwargs")
    @classmethod
    def _validate_pw_kwargs(cls, v: dict) -> dict:
        overlap = {"headless", "chromium_sandbox"} & v.keys()
        if overlap:
            raise ValueError(f"pw_kwargs keys {overlap} shadow named config fields")
        return v

    def make(self, container=None) -> "SyncPlaywrightTool":
        return SyncPlaywrightTool(self)


class SyncPlaywrightTool(Tool, BrowserActionSpace):
    """Synchronous Playwright browser tool.

    Implements the ``cube.tools.browser.AbstractBrowserTool`` protocol via:
    - ``Tool`` (cube-standard): automatic ``action_set`` discovery and
      ``execute_action`` dispatch via the ``@tool_action`` decorator.
    - ``BrowserActionSpace``: abstract CSS-selector action contracts with
      pre-attached ``@tool_action`` registrations.

    ``execute_action`` is overridden to append ``page_obs()`` after every
    successful action so the agent always receives the updated page state.

    The Playwright context (browser + page) is created eagerly in ``__init__``
    and torn down in ``close()``. Call ``reset()`` between episodes to get a
    fresh page without restarting the browser process.
    """

    def __init__(self, config: PlaywrightConfig) -> None:
        self.config = config
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(
            headless=config.headless,
            chromium_sandbox=config.chromium_sandbox,
            **config.pw_kwargs,
        )
        self._page: SyncPage = self._browser.new_page(viewport=config.viewport)

    @property
    def page(self) -> SyncPage:
        return self._page

    # ------------------------------------------------------------------
    # Tool lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Close the current page and open a fresh one (clears cookies and state)."""
        self._page.close()
        self._page = self._browser.new_page(viewport=self.config.viewport)

    def close(self) -> None:
        """Release all Playwright resources (page, browser, playwright instance).

        Attempts all three cleanup steps regardless of individual failures.
        Raises ``RuntimeError`` listing all errors if any cleanup step failed.
        """
        errors = []
        for action, label in [
            (self._page.close, "page"),
            (self._browser.close, "browser"),
            (self._pw.stop, "playwright"),
        ]:
            try:
                action()
            except Exception as e:
                errors.append(f"{label}: {e}")
        if errors:
            raise RuntimeError(f"Errors during SyncPlaywrightTool.close(): {'; '.join(errors)}")

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
        self._page.goto(url)

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
        result = self._page.evaluate(js)
        logger.debug("JS result: %s", result)
        return result

    def page_obs(self) -> Observation:
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

    def page_html(self) -> str:
        """Return the raw HTML of the current page."""
        return self._page.content()

    def page_screenshot(self) -> Image.Image:
        """Return a PIL screenshot of the current page."""
        return Image.open(BytesIO(self._page.screenshot()))

    def page_axtree(self) -> str:
        """Return the accessibility tree of the current page as indented text."""
        return flatten_axtree(self._page.accessibility.snapshot())

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
        self._page.click(selector, timeout=3000, strict=True)

    def browser_type(self, selector: str, text: str) -> None:
        """Type text into an element specified by CSS selector.

        Parameters
        ----------
        selector : str
            CSS selector of the element to type into.
        text : str
            Text to type.
        """
        self._page.type(selector, text)

    def browser_press_key(self, key: str) -> None:
        """Press a keyboard key.

        Parameters
        ----------
        key : str
            Key name as accepted by Playwright (e.g. 'Enter', 'Tab', 'Escape').
        """
        self._page.keyboard.press(key)

    def browser_hover(self, selector: str) -> None:
        """Hover over an element specified by CSS selector.

        Parameters
        ----------
        selector : str
            CSS selector of the element to hover over.
        """
        self._page.hover(selector, timeout=3000, strict=True)

    def browser_drag(self, from_selector: str, to_selector: str) -> None:
        """Drag an element to another element using CSS selectors.

        Parameters
        ----------
        from_selector : str
            CSS selector of the element to drag.
        to_selector : str
            CSS selector of the drop target.
        """
        from_elem = self._page.locator(from_selector)
        from_elem.hover(timeout=500)
        self._page.mouse.down()
        try:
            to_elem = self._page.locator(to_selector)
            to_elem.hover(timeout=500)
        except Exception:
            self._page.mouse.up()
            raise
        self._page.mouse.up()

    def browser_select_option(self, selector: str, value: str) -> None:
        """Select an option in a <select> element specified by CSS selector.

        Parameters
        ----------
        selector : str
            CSS selector of the <select> element.
        value : str
            Option value to select.
        """
        self._page.select_option(selector, value)

    def browser_mouse_click_xy(self, x: int, y: int) -> None:
        """Click at an absolute (x, y) coordinate on the page.

        Parameters
        ----------
        x : int
            Horizontal coordinate in pixels from the left edge of the viewport.
        y : int
            Vertical coordinate in pixels from the top edge of the viewport.
        """
        self._page.mouse.click(x, y, delay=100)

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
        elem = self._page.locator(selector).first
        elem.scroll_into_view_if_needed()
        box = elem.bounding_box()
        if box is None:
            raise ValueError(
                f"browser_scroll: element '{selector}' has no bounding box (it may be hidden or have zero dimensions)."
            )
        self._page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        delta_x = {"left": -amount, "right": amount}.get(direction, 0)
        delta_y = {"up": -amount, "down": amount}.get(direction, 0)
        self._page.mouse.wheel(delta_x, delta_y)

    def browser_back(self) -> None:
        """Navigate back in the browser history."""
        self._page.go_back()

    def browser_forward(self) -> None:
        """Navigate forward in the browser history."""
        self._page.go_forward()

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
