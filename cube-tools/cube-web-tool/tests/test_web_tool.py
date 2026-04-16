"""Tests for WebSearchTool and WebFetchTool."""

from cube_web_tool import BraveWebSearchToolConfig, WebFetchToolConfig


def test_brave_web_search_config_defaults() -> None:
    config = BraveWebSearchToolConfig()
    assert config.brave_api_key is None
    assert config.num_search_results == 10


def test_web_fetch_config_defaults() -> None:
    config = WebFetchToolConfig()
    assert config.fetch_llm_model == "gpt-5.4-mini"
    assert config.max_fetch_chars == 20_000


def test_brave_web_search_config_round_trip() -> None:
    config = BraveWebSearchToolConfig(brave_api_key="test-key", num_search_results=5)
    restored = BraveWebSearchToolConfig.model_validate(config.model_dump())
    assert restored.brave_api_key == "test-key"
    assert restored.num_search_results == 5


def test_web_fetch_config_round_trip() -> None:
    config = WebFetchToolConfig(fetch_llm_model="claude-3-haiku", max_fetch_chars=5000)
    restored = WebFetchToolConfig.model_validate(config.model_dump())
    assert restored.fetch_llm_model == "claude-3-haiku"
    assert restored.max_fetch_chars == 5000


def test_brave_web_search_tool_action_set() -> None:
    tool = BraveWebSearchToolConfig().make()
    names = {a.name for a in tool.action_set}
    assert "web_search" in names
    tool.close()


def test_web_fetch_tool_action_set() -> None:
    tool = WebFetchToolConfig().make()
    names = {a.name for a in tool.action_set}
    assert "web_fetch" in names
    tool.close()
