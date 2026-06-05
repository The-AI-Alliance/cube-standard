"""Browser tool abstract bases for web-based benchmark tasks.

cube-standard declares the contracts; concrete implementations live in
cube-tools/cube-browser-tool/ (or any compatible harness).

Two classes are provided — one sync, one async — mirroring the Playwright
sync/async API split:

    BrowserTool      — synchronous; primary path for cube-standard tasks.
    AsyncBrowserTool — asynchronous; for high-throughput parallel collection.

Web benchmark tasks (MiniWob, WorkArena, WebArena, …) type their tool as
BrowserTool and require no knowledge of the concrete implementation.
"""

from abc import abstractmethod
from typing import Any

from cube.core import Observation
from cube.resources.browser_session import AsyncBrowserSession, BrowserSession
from cube.tool import Tool


class BrowserTool(Tool):
    """Abstract base for browser tools used by web-based tasks (setup, validation, observation)."""

    @property
    @abstractmethod
    def session(self) -> BrowserSession: ...

    @abstractmethod
    def noop(self) -> None: ...

    @abstractmethod
    def goto(self, url: str) -> None: ...

    @abstractmethod
    def evaluate_js(self, js: str) -> Any: ...

    @abstractmethod
    def page_obs(self) -> Observation: ...


class AsyncBrowserTool(Tool):
    """Abstract base for async browser tools used by web-based tasks
    (setup, validation, observation).

    Subclasses `Tool` directly (was `AsyncTool` before the tool
    consolidation). All `@tool_action` methods on subclasses are
    `async def`; `Tool.async_execute_action` dispatches them
    natively, and `Tool.execute_action` bridges via thread+loop for
    sync callers.
    """

    @property
    @abstractmethod
    def session(self) -> AsyncBrowserSession: ...

    @abstractmethod
    async def noop(self) -> None: ...

    @abstractmethod
    async def goto(self, url: str) -> None: ...

    @abstractmethod
    async def evaluate_js(self, js: str) -> Any: ...

    @abstractmethod
    async def page_obs(self) -> Observation: ...
