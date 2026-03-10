"""Abstract action space contracts for browser tools.

Two ABCs define the canonical browser action sets:

    BrowserActionSpace    — CSS-selector-based (Playwright, Selenium, …)
    BidBrowserActionSpace — BID-based (BrowserGym, Phase 3)

Both use @tool_action on abstract methods so that subclass overrides inherit the
action registration without repeating the decorator.

Neither ABC includes ``goto`` — that is a task-internal method (called by task
logic during reset/setup) and is not exposed to the agent.

All docstrings follow NumPy format: the summary line becomes the LLM-facing action
description; ``Parameters`` sections populate the per-argument JSON schema
(parsed by cube-standard's ``function_to_dict``).
"""

from abc import ABC, abstractmethod

from cube.tool import tool_action


class BrowserActionSpace(ABC):
    """Abstract base for browser tools that locate elements via CSS selectors.

    Subclasses must implement every method. The ``@tool_action`` decoration on
    each abstract method is inherited by overrides, so concrete classes do not
    need to repeat the decorator.
    """

    @tool_action
    @abstractmethod
    def browser_click(self, selector: str) -> None:
        """Click on an element specified by CSS selector.

        Parameters
        ----------
        selector : str
            CSS selector of the element to click.
        """
        ...

    @tool_action
    @abstractmethod
    def browser_type(self, selector: str, text: str) -> None:
        """Type text into an element specified by CSS selector.

        Parameters
        ----------
        selector : str
            CSS selector of the element to type into.
        text : str
            Text to type.
        """
        ...

    @tool_action
    @abstractmethod
    def browser_press_key(self, key: str) -> None:
        """Press a keyboard key.

        Parameters
        ----------
        key : str
            Key name as accepted by Playwright (e.g. 'Enter', 'Tab', 'Escape').
        """
        ...

    @tool_action
    @abstractmethod
    def browser_hover(self, selector: str) -> None:
        """Hover over an element specified by CSS selector.

        Parameters
        ----------
        selector : str
            CSS selector of the element to hover over.
        """
        ...

    @tool_action
    @abstractmethod
    def browser_drag(self, from_selector: str, to_selector: str) -> None:
        """Drag an element to another element using CSS selectors.

        Parameters
        ----------
        from_selector : str
            CSS selector of the element to drag.
        to_selector : str
            CSS selector of the drop target.
        """
        ...

    @tool_action
    @abstractmethod
    def browser_select_option(self, selector: str, value: str) -> None:
        """Select an option in a <select> element specified by CSS selector.

        Parameters
        ----------
        selector : str
            CSS selector of the <select> element.
        value : str
            Option value to select.
        """
        ...

    @tool_action
    @abstractmethod
    def browser_mouse_click_xy(self, x: int, y: int) -> None:
        """Click at an absolute (x, y) coordinate on the page.

        Parameters
        ----------
        x : int
            Horizontal coordinate in pixels from the left edge of the viewport.
        y : int
            Vertical coordinate in pixels from the top edge of the viewport.
        """
        ...

    @tool_action
    @abstractmethod
    def browser_scroll(self, selector: str, direction: str, amount: int) -> None:
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
        ...

    @tool_action
    @abstractmethod
    def browser_back(self) -> None:
        """Navigate back in the browser history."""
        ...

    @tool_action
    @abstractmethod
    def browser_forward(self) -> None:
        """Navigate forward in the browser history."""
        ...

    @tool_action
    @abstractmethod
    def browser_wait(self, seconds: int) -> None:
        """Wait for a number of seconds before the next action.

        Parameters
        ----------
        seconds : int
            Number of seconds to wait (capped at the tool's ``max_wait``).
        """
        ...

    @tool_action
    @abstractmethod
    def noop(self) -> None:
        """No-op: take no action and return the current page state."""
        ...


class BidBrowserActionSpace(ABC):
    """Abstract base for browser tools that locate elements via BrowserGym BIDs.

    Same action set as ``BrowserActionSpace`` but element selectors use BIDs
    (Browser IDs injected by BrowserGym's ``_pre_extract``).

    Used by ``BgymTool`` (Phase 3).
    """

    @tool_action
    @abstractmethod
    def browser_click(self, bid: str) -> None:
        """Click on an element specified by BID.

        Parameters
        ----------
        bid : str
            BrowserGym BID of the element to click (e.g. 'a51').
        """
        ...

    @tool_action
    @abstractmethod
    def browser_type(self, bid: str, text: str) -> None:
        """Type text into an element specified by BID.

        Parameters
        ----------
        bid : str
            BrowserGym BID of the element to type into.
        text : str
            Text to type.
        """
        ...

    @tool_action
    @abstractmethod
    def browser_press_key(self, key: str) -> None:
        """Press a keyboard key.

        Parameters
        ----------
        key : str
            Key name as accepted by Playwright (e.g. 'Enter', 'Tab', 'Escape').
        """
        ...

    @tool_action
    @abstractmethod
    def browser_hover(self, bid: str) -> None:
        """Hover over an element specified by BID.

        Parameters
        ----------
        bid : str
            BrowserGym BID of the element to hover over.
        """
        ...

    @tool_action
    @abstractmethod
    def browser_drag(self, from_bid: str, to_bid: str) -> None:
        """Drag an element to another element using BIDs.

        Parameters
        ----------
        from_bid : str
            BrowserGym BID of the element to drag.
        to_bid : str
            BrowserGym BID of the drop target.
        """
        ...

    @tool_action
    @abstractmethod
    def browser_select_option(self, bid: str, value: str) -> None:
        """Select an option in a <select> element specified by BID.

        Parameters
        ----------
        bid : str
            BrowserGym BID of the <select> element.
        value : str
            Option value to select.
        """
        ...

    @tool_action
    @abstractmethod
    def browser_mouse_click_xy(self, x: int, y: int) -> None:
        """Click at an absolute (x, y) coordinate on the page.

        Parameters
        ----------
        x : int
            Horizontal coordinate in pixels from the left edge of the viewport.
        y : int
            Vertical coordinate in pixels from the top edge of the viewport.
        """
        ...

    @tool_action
    @abstractmethod
    def browser_scroll(self, bid: str, direction: str, amount: int) -> None:
        """Scroll an element in the specified direction.

        Parameters
        ----------
        bid : str
            BrowserGym BID of the element to scroll.
        direction : {'up', 'down', 'left', 'right'}
            Direction to scroll.
        amount : int
            Number of pixels to scroll.
        """
        ...

    @tool_action
    @abstractmethod
    def browser_back(self) -> None:
        """Navigate back in the browser history."""
        ...

    @tool_action
    @abstractmethod
    def browser_forward(self) -> None:
        """Navigate forward in the browser history."""
        ...

    @tool_action
    @abstractmethod
    def browser_wait(self, seconds: int) -> None:
        """Wait for a number of seconds before the next action.

        Parameters
        ----------
        seconds : int
            Number of seconds to wait (capped at the tool's ``max_wait``).
        """
        ...

    @tool_action
    @abstractmethod
    def noop(self) -> None:
        """No-op: take no action and return the current page state."""
        ...
