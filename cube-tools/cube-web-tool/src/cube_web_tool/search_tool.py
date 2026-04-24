"""WebSearchTool for cube benchmarks."""

import logging
import os
from abc import abstractmethod

import httpx
from cube.tool import Tool, ToolConfig, tool_action

logger = logging.getLogger(__name__)

_USER_AGENT = "cube-web-tool/1.0"
_BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


class WebSearchToolConfig(ToolConfig):
    """Base configuration for web search tools."""


class WebSearchTool(Tool):
    """Abstract base for web search tools."""

    @tool_action
    @abstractmethod
    def web_search(self, query: str) -> str:
        """Search the web and return a list of results.

        Args:
            query: The search query string.
        """


class BraveWebSearchToolConfig(WebSearchToolConfig):
    """Configuration for BraveWebSearchTool."""

    brave_api_key: str | None = None
    num_search_results: int = 10

    def make(self, container=None) -> "BraveWebSearchTool":
        return BraveWebSearchTool(config=self)


class BraveWebSearchTool(WebSearchTool):
    """Tool providing Brave Search for cube benchmarks."""

    def __init__(self, config: BraveWebSearchToolConfig) -> None:
        self.config = config
        self._http_client = httpx.Client(timeout=30.0, headers={"User-Agent": _USER_AGENT})

    def close(self) -> None:
        self._http_client.close()

    def _resolve_brave_api_key(self) -> str:
        if self.config.brave_api_key is not None:
            return self.config.brave_api_key
        key = os.environ.get("BRAVE_API_KEY")
        if key is None:
            raise ValueError("BRAVE_API_KEY not set in config or environment")
        return key

    @tool_action
    def web_search(self, query: str) -> str:
        """Search the web and return a list of results.

        Args:
            query: The search query string.
        """
        try:
            api_key = self._resolve_brave_api_key()
        except ValueError as e:
            return f"Search error: {e}"

        params = {"q": query, "count": self.config.num_search_results}
        headers = {"X-Subscription-Token": api_key, "Accept": "application/json"}
        try:
            response = self._http_client.get(_BRAVE_SEARCH_URL, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.warning("web_search failed: %s", e)
            return f"Search error: {e}"

        results = data.get("web", {}).get("results", [])
        if not results:
            return "No results found."

        lines: list[str] = []
        for i, result in enumerate(results, start=1):
            title = result.get("title", "(no title)")
            url = result.get("url", "")
            description = result.get("description", "")
            lines.append(f"{i}. {title} - {url}\n   {description}")
        return "\n".join(lines)
