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

    fetch_llm_model: str = "gpt-5.4-mini"
    max_fetch_chars: int = 20_000

    def make(self, container=None) -> "WebFetchTool":
        return WebFetchTool(config=self)


class WebFetchTool(Tool):
    """Tool for fetching web pages and extracting relevant content via LLM."""

    def __init__(self, config: WebFetchToolConfig) -> None:
        self.config = config
        self._http_client = httpx.Client(timeout=30.0, headers={"User-Agent": _USER_AGENT})
        self._markitdown = MarkItDown()

    def close(self) -> None:
        self._http_client.close()

    @tool_action
    def web_fetch(self, url: str, query: str) -> str:
        """Fetch a web page, convert it to markdown, and extract information relevant to the query.

        Args:
            url: The URL to fetch.
            query: The research question to extract information for.
        """
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

        truncated = markdown[: self.config.max_fetch_chars]

        try:
            completion = litellm.completion(
                model=self.config.fetch_llm_model,
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
