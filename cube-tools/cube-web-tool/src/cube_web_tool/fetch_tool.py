"""WebFetchTool for cube benchmarks."""

import logging

import httpx
import litellm
from cube.tool import Tool, ToolConfig, tool_action
from markitdown import MarkItDown

logger = logging.getLogger(__name__)

_USER_AGENT = "cube-web-tool/1.0"


class WebFetchToolConfig(ToolConfig):
    """Configuration for WebFetchTool."""

    query_llm_model: str | None
    max_fetch_chars: int = 20_000

    def make(self, container=None) -> "WebFetchTool":
        if self.query_llm_model is None:
            return WebFetchURLOnlyTool(config=self)
        return WebFetchWithQueryTool(config=self)


class WebFetchTool(Tool):
    """Base for web fetch tools. Owns HTTP/markdown plumbing but exposes no actions."""

    def __init__(self, config: WebFetchToolConfig) -> None:
        self.config = config
        self._http_client = httpx.Client(timeout=30.0, headers={"User-Agent": _USER_AGENT})
        self._markitdown = MarkItDown()

    def close(self) -> None:
        self._http_client.close()

    def _fetch_markdown(self, url: str) -> str:
        try:
            result = self._markitdown.convert_url(url)
            markdown = result.text_content
        except Exception as fetch_err:
            logger.warning("web_fetch markitdown failed for %s: %s", url, fetch_err)
            try:
                response = self._http_client.get(url)
                response.raise_for_status()
                markdown = response.text
            except Exception as e:
                return f"Fetch error: {e}"

        return markdown[: self.config.max_fetch_chars]


class WebFetchWithQueryTool(WebFetchTool):
    """Fetch a page and LLM-extract information relevant to a query."""

    @tool_action
    def web_fetch(self, url: str, query: str) -> str:
        """Fetch a web page, convert it to markdown, and extract information relevant to the query.

        Args:
            url: The URL to fetch.
            query: The research question to extract information for.
        """
        truncated = self._fetch_markdown(url)
        if truncated.startswith("Fetch error: "):
            return truncated

        try:
            completion = litellm.completion(
                model=self.config.query_llm_model,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Extract information relevant to the following question from the page content below.\n\n"
                            f"Question: {query}\n\n"
                            f"Page content:\n{truncated}"
                        ),
                    }
                ],
            )
            return completion.choices[0].message.content or truncated
        except Exception as e:
            logger.warning("web_fetch LLM extraction failed: %s", e)
            return truncated


class WebFetchURLOnlyTool(WebFetchTool):
    """Fetch a page and return the truncated markdown directly, without LLM extraction."""

    @tool_action
    def web_fetch(self, url: str) -> str:
        """Fetch a web page and return its content as truncated markdown.

        Args:
            url: The URL to fetch.
        """
        return self._fetch_markdown(url)
