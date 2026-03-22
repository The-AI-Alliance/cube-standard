"""Tests for PlaywrightConfig and SyncPlaywrightTool.

Unit tests (no browser required) run by default.
Integration tests (require Playwright + Chromium) are marked with
``@pytest.mark.integration`` and excluded from the default ``pytest`` run.
Run them explicitly with: ``pytest -m integration``
"""

import pytest
from cube.core import Action, Observation, StepError
from cube.tool import BrowserTool
from cube_browser_playwright import PlaywrightSessionConfig, Viewport

from cube_browser_tool import BidBrowserActionSpace, BrowserActionSpace, PlaywrightConfig
from cube_browser_tool.playwright_tool import SyncPlaywrightTool

# ---------------------------------------------------------------------------
# Unit tests — no browser required
# ---------------------------------------------------------------------------

EXPECTED_ACTION_NAMES = {
    "browser_click",
    "browser_type",
    "browser_press_key",
    "browser_hover",
    "browser_drag",
    "browser_select_option",
    "browser_mouse_click_xy",
    "browser_scroll",
    "browser_back",
    "browser_forward",
    "browser_wait",
    "noop",
}


def test_playwright_config_defaults() -> None:
    config = PlaywrightConfig()
    assert config.browser.headless is True
    assert config.browser.viewport == Viewport(width=1280, height=720)
    assert config.use_html is True
    assert config.use_screenshot is True
    assert config.use_axtree is False
    assert config.prune_html is True
    assert config.max_wait == 60


def test_playwright_config_round_trip() -> None:
    config = PlaywrightConfig(
        browser=PlaywrightSessionConfig(headless=False, viewport=Viewport(width=800, height=600)),
        use_axtree=True,
    )
    data = config.model_dump()
    restored = PlaywrightConfig.model_validate(data)
    assert restored == config


def test_browser_action_space_is_abstract() -> None:
    """BrowserActionSpace cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BrowserActionSpace()  # type: ignore[abstract]


def test_bid_browser_action_space_is_abstract() -> None:
    """BidBrowserActionSpace cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BidBrowserActionSpace()  # type: ignore[abstract]


def test_max_wait_validator_rejects_zero() -> None:
    with pytest.raises(Exception, match="max_wait must be positive"):
        PlaywrightConfig(max_wait=0)


def test_max_wait_validator_rejects_negative() -> None:
    with pytest.raises(Exception, match="max_wait must be positive"):
        PlaywrightConfig(max_wait=-1)


def test_sync_playwright_tool_is_browser_tool() -> None:
    assert issubclass(SyncPlaywrightTool, BrowserTool)


# ---------------------------------------------------------------------------
# Integration tests — require a live Playwright/Chromium install
# ---------------------------------------------------------------------------

SIMPLE_PAGE = "data:text/html,<html><body><button id='btn'>Click me</button><p id='msg'>hello</p></body></html>"
SELECT_PAGE = "data:text/html,<html><body><select id='sel'><option value='a'>A</option><option value='b'>B</option></select></body></html>"
SCROLL_PAGE = "data:text/html,<html><body style='height:2000px'><div id='top'>top</div><div id='bottom' style='margin-top:1500px'>bottom</div></body></html>"


@pytest.mark.integration
def test_action_set_names() -> None:
    pytest.importorskip("playwright", reason="playwright not installed")
    tool = PlaywrightConfig(
        browser=PlaywrightSessionConfig(headless=True), use_html=True, use_screenshot=False, use_axtree=False
    ).make()
    try:
        names = {schema.name for schema in tool.action_set}
        assert EXPECTED_ACTION_NAMES == names
    finally:
        tool.close()


@pytest.mark.integration
def test_page_obs_returns_observation() -> None:
    pytest.importorskip("playwright", reason="playwright not installed")
    tool = PlaywrightConfig(
        browser=PlaywrightSessionConfig(headless=True), use_html=True, use_screenshot=False, use_axtree=False
    ).make()
    try:
        tool.goto(SIMPLE_PAGE)
        obs = tool.page_obs()
        assert isinstance(obs, Observation)
        assert len(obs.contents) > 0
    finally:
        tool.close()


@pytest.mark.integration
def test_page_obs_contains_html() -> None:
    pytest.importorskip("playwright", reason="playwright not installed")
    tool = PlaywrightConfig(
        browser=PlaywrightSessionConfig(headless=True), use_html=True, use_screenshot=False, use_axtree=False
    ).make()
    try:
        tool.goto(SIMPLE_PAGE)
        obs = tool.page_obs()
        combined = " ".join(c.data for c in obs.contents if hasattr(c, "data") and isinstance(c.data, str))
        assert "btn" in combined or "Click me" in combined
    finally:
        tool.close()


@pytest.mark.integration
def test_page_obs_respects_use_html_false() -> None:
    pytest.importorskip("playwright", reason="playwright not installed")
    tool = PlaywrightConfig(
        browser=PlaywrightSessionConfig(headless=True), use_html=False, use_screenshot=False, use_axtree=False
    ).make()
    try:
        tool.goto(SIMPLE_PAGE)
        obs = tool.page_obs()
        assert isinstance(obs, Observation)
        assert len(obs.contents) == 0
    finally:
        tool.close()


@pytest.mark.integration
def test_execute_action_appends_page_obs() -> None:
    pytest.importorskip("playwright", reason="playwright not installed")
    tool = PlaywrightConfig(
        browser=PlaywrightSessionConfig(headless=True), use_html=True, use_screenshot=False, use_axtree=False
    ).make()
    try:
        tool.goto(SIMPLE_PAGE)
        action = Action(name="noop", arguments={})
        result = tool.execute_action(action)
        assert isinstance(result, Observation)
        assert len(result.contents) >= 1
    finally:
        tool.close()


@pytest.mark.integration
def test_execute_action_returns_step_error_on_bad_selector() -> None:
    pytest.importorskip("playwright", reason="playwright not installed")
    tool = PlaywrightConfig(
        browser=PlaywrightSessionConfig(headless=True), use_html=True, use_screenshot=False, use_axtree=False
    ).make()
    try:
        tool.goto(SIMPLE_PAGE)
        action = Action(name="browser_click", arguments={"selector": "#does-not-exist"})
        result = tool.execute_action(action)
        assert isinstance(result, StepError)
    finally:
        tool.close()


@pytest.mark.integration
def test_reset_clears_page() -> None:
    pytest.importorskip("playwright", reason="playwright not installed")
    tool = PlaywrightConfig(
        browser=PlaywrightSessionConfig(headless=True), use_html=True, use_screenshot=False, use_axtree=False
    ).make()
    try:
        tool.goto(SIMPLE_PAGE)
        tool.reset()
        # After reset the page should be blank (about:blank), not the data URL
        obs = tool.page_obs()
        combined = " ".join(c.data for c in obs.contents if hasattr(c, "data") and isinstance(c.data, str))
        assert "Click me" not in combined
    finally:
        tool.close()


@pytest.mark.integration
def test_evaluate_js() -> None:
    pytest.importorskip("playwright", reason="playwright not installed")
    tool = PlaywrightConfig(
        browser=PlaywrightSessionConfig(headless=True), use_html=True, use_screenshot=False, use_axtree=False
    ).make()
    try:
        tool.goto(SIMPLE_PAGE)
        title = tool.evaluate_js("() => document.getElementById('msg').textContent")
        assert title == "hello"
    finally:
        tool.close()


@pytest.mark.integration
def test_browser_select_option() -> None:
    pytest.importorskip("playwright", reason="playwright not installed")
    tool = PlaywrightConfig(
        browser=PlaywrightSessionConfig(headless=True), use_html=True, use_screenshot=False, use_axtree=False
    ).make()
    try:
        tool.goto(SELECT_PAGE)
        action = Action(name="browser_select_option", arguments={"selector": "#sel", "value": "b"})
        result = tool.execute_action(action)
        assert isinstance(result, Observation)
        selected = tool.evaluate_js("() => document.getElementById('sel').value")
        assert selected == "b"
    finally:
        tool.close()


@pytest.mark.integration
def test_browser_wait_is_capped_at_max_wait() -> None:
    import time

    pytest.importorskip("playwright", reason="playwright not installed")
    tool = PlaywrightConfig(
        browser=PlaywrightSessionConfig(headless=True),
        use_html=False,
        use_screenshot=False,
        use_axtree=False,
        max_wait=1,
    ).make()
    try:
        start = time.monotonic()
        tool.execute_action(Action(name="browser_wait", arguments={"seconds": 9999}))
        elapsed = time.monotonic() - start
        assert elapsed < 3.0  # capped at max_wait=1, with 2s slack for CI
    finally:
        tool.close()


@pytest.mark.integration
def test_page_property_exposes_raw_page() -> None:
    pytest.importorskip("playwright", reason="playwright not installed")
    tool = PlaywrightConfig(
        browser=PlaywrightSessionConfig(headless=True), use_html=False, use_screenshot=False, use_axtree=False
    ).make()
    try:
        tool.goto(SIMPLE_PAGE)
        from playwright.sync_api import Page as SyncPage

        assert isinstance(tool.page, SyncPage)
    finally:
        tool.close()
