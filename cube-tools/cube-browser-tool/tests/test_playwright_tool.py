"""Tests for PlaywrightConfig and SyncPlaywrightTool.

Unit tests (no browser required) run by default.
Integration tests (require Playwright + Chromium) are marked with
``@pytest.mark.integration`` and excluded from the default test run.
Run them with: ``pytest -m integration``
"""

import pytest
from cube.core import Action, Observation, StepError

from cube_browser_tool import BrowserActionSpace, PlaywrightConfig

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


def test_playwright_config_defaults():
    config = PlaywrightConfig()
    assert config.headless is True
    assert config.viewport == {"width": 1280, "height": 720}
    assert config.use_html is True
    assert config.use_screenshot is True
    assert config.use_axtree is False
    assert config.prune_html is True
    assert config.max_wait == 60


def test_playwright_config_round_trip():
    config = PlaywrightConfig(headless=False, viewport={"width": 800, "height": 600}, use_axtree=True)
    data = config.model_dump()
    restored = PlaywrightConfig.model_validate(data)
    assert restored == config


def test_browser_action_space_is_abstract():
    """BrowserActionSpace cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BrowserActionSpace()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Integration tests — require a live Playwright/Chromium install
# ---------------------------------------------------------------------------

playwright = pytest.importorskip("playwright", reason="playwright not installed")

SIMPLE_PAGE = "data:text/html,<html><body><button id='btn'>Click me</button><p id='msg'>hello</p></body></html>"
SELECT_PAGE = "data:text/html,<html><body><select id='sel'><option value='a'>A</option><option value='b'>B</option></select></body></html>"


@pytest.fixture(scope="module")
def tool():
    t = PlaywrightConfig(headless=True, use_html=True, use_screenshot=False, use_axtree=False).make()
    yield t
    t.close()


@pytest.mark.integration
def test_action_set_names(tool):
    names = {schema.name for schema in tool.action_set}
    assert EXPECTED_ACTION_NAMES == names


@pytest.mark.integration
def test_page_obs_returns_observation(tool):
    tool.goto(SIMPLE_PAGE)
    obs = tool.page_obs()
    assert isinstance(obs, Observation)
    assert len(obs.contents) > 0


@pytest.mark.integration
def test_page_obs_contains_html(tool):
    tool.goto(SIMPLE_PAGE)
    obs = tool.page_obs()
    combined = " ".join(c.data for c in obs.contents if hasattr(c, "data") and isinstance(c.data, str))
    assert "btn" in combined or "Click me" in combined


@pytest.mark.integration
def test_execute_action_appends_page_obs(tool):
    tool.goto(SIMPLE_PAGE)
    action = Action(name="noop", arguments={})
    result = tool.execute_action(action)
    assert isinstance(result, Observation)
    # Should have at least one content item from page_obs
    assert len(result.contents) >= 1


@pytest.mark.integration
def test_execute_action_returns_step_error_on_bad_selector(tool):
    tool.goto(SIMPLE_PAGE)
    action = Action(name="browser_click", arguments={"selector": "#does-not-exist"})
    result = tool.execute_action(action)
    assert isinstance(result, StepError)


@pytest.mark.integration
def test_reset_clears_page(tool):
    tool.goto(SIMPLE_PAGE)
    tool.reset()
    # After reset the page should be blank (about:blank), not the data URL
    obs = tool.page_obs()
    combined = " ".join(c.data for c in obs.contents if hasattr(c, "data") and isinstance(c.data, str))
    assert "Click me" not in combined


@pytest.mark.integration
def test_evaluate_js(tool):
    tool.goto(SIMPLE_PAGE)
    title = tool.evaluate_js("() => document.getElementById('msg').textContent")
    assert title == "hello"


@pytest.mark.integration
def test_browser_select_option(tool):
    tool.goto(SELECT_PAGE)
    action = Action(name="browser_select_option", arguments={"selector": "#sel", "value": "b"})
    result = tool.execute_action(action)
    assert isinstance(result, Observation)
    selected = tool.evaluate_js("() => document.getElementById('sel').value")
    assert selected == "b"
