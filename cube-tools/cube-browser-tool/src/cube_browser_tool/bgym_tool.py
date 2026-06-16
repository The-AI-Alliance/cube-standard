"""BrowserGym-backed browser tool (optional dependency).

``BgymTool`` exposes BrowserGym's native action set on a Playwright page and
extracts BrowserGym observations (DOM / AXTree / screenshot). ``ExtraWebActionsTool``
adds ``keyboard_type_into`` and ``js_eval`` on top of the same session.

Requires the ``bgym`` extra::

    pip install cube-browser-tool[bgym]

Importing this module without ``browsergym-core`` installed raises a clear
``ImportError`` with the install hint.

Usage::

    from cube_browser_tool.bgym_tool import BgymToolConfig

    # axtree + screenshot (the canonical WorkArena observation):
    config = BgymToolConfig(use_html=False, use_axtree=True, use_screenshot=True)

    # bundled with keyboard_type_into / js_eval, flattened into one Toolbox:
    from cube_browser_tool.bgym_tool import ExtendedBrowserConfig
    config = ExtendedBrowserConfig(browser=BgymToolConfig(...))
"""

import json
import logging
import time
from typing import Any

import numpy as np
from cube.core import Action, ActionSchema, Content, Observation, StepError
from cube.tool import Tool, Toolbox, ToolConfig, tool_action
from cube.tools.browser import BrowserTool
from cube_browser_playwright.playwright_session import (
    PlaywrightSession,
    PlaywrightSessionConfig,
)
from PIL import Image
from playwright.sync_api import Error, Page
from pydantic import Field

try:
    from browsergym.core.action.base import execute_python_code
    from browsergym.core.action.highlevel import HighLevelActionSet
    from browsergym.core.action.utils import get_elem_by_bid
    from browsergym.core.constants import BROWSERGYM_ID_ATTRIBUTE, EXTRACT_OBS_MAX_TRIES
    from browsergym.core.observation import (
        MarkingError,
        _post_extract,
        _pre_extract,
        extract_dom_extra_properties,
        extract_dom_snapshot,
        extract_focused_element_bid,
        extract_merged_axtree,
        extract_screenshot,
    )
    from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html
except ImportError as e:  # pragma: no cover - exercised via packaging, not unit tests
    raise ImportError(
        "cube_browser_tool.bgym_tool requires the 'bgym' extra. Install it with: pip install cube-browser-tool[bgym]"
    ) from e

logger = logging.getLogger(__name__)


class BgymToolConfig(ToolConfig):
    """Configuration for the BrowserGym-style Playwright tool."""

    # Browser configuration (launch parameters)
    browser: PlaywrightSessionConfig = Field(default_factory=PlaywrightSessionConfig)

    # Action configuration
    action_subsets: list[str] = Field(default=["bid", "nav", "tab"])

    # Observation behavior
    tags_to_mark: str = "standard_html"  # "all" or "standard_html"
    pre_observation_delay: float = 0.5

    # Observation configuration
    use_html: bool = True
    use_axtree: bool = True
    use_screenshot: bool = True
    prune_html: bool = True

    # AXTree element attributes — requires extra_element_properties from the DOM snapshot
    axtree_with_visible: bool = False  # label visible elements (vis >= 0.5) as "visible"
    axtree_with_clickable: bool = False  # label clickable elements as "clickable"

    def make(self, container: Any = None) -> "BgymTool":
        return BgymTool(self)


class BgymTool(BrowserTool):
    """Browser tool using BrowserGym's action set on a Playwright Page.

    Exposes bgym's native actions (click, fill, scroll, ...) as tool actions.
    Pure browser — chat and infeasibility actions belong to ChatTool.
    """

    def __init__(self, config: BgymToolConfig) -> None:
        super().__init__()
        self.config = config
        self._action_set = HighLevelActionSet(subsets=config.action_subsets, multiaction=False)
        self._action_schemas: list[ActionSchema] | None = None
        self._session: PlaywrightSession | None = None
        self._last_obs: dict | None = None
        self._last_info: dict | None = None
        self._last_reward: float = 0.0
        self._last_terminated: bool = False

    # === Action set: built from bgym's HighLevelActionSet ===

    @property
    def action_set(self) -> list[ActionSchema]:
        if self._action_schemas is None:
            self._action_schemas = _build_action_schemas(self._action_set)
        return self._action_schemas

    # === Action execution: serialise Action -> bgym string -> execute ===

    def execute_action(self, action: Action) -> Observation | StepError:
        """Serialise an Action to a bgym action string, execute it, and return the observation."""
        action_str = _action_to_bgym_string(action)
        result = self._execute_bgym_step(action_str)
        obs = self.page_obs()
        return Observation(contents=[Content.from_data(result, tool_call_id=action.id)]) + obs

    # === BrowserTool interface ===

    @property
    def session(self) -> PlaywrightSession:
        if self._session is None:
            raise RuntimeError("Browser is not initialized. Call reset() first.")
        return self._session

    @property
    def page(self) -> Page:
        return self.session.page

    @property
    def last_reward(self) -> float:
        return self._last_reward

    @property
    def last_terminated(self) -> bool:
        return self._last_terminated

    def goto(self, url: str) -> None:
        self._execute_bgym_step(f'goto(url="{url}")')

    def noop(self) -> None:
        self._execute_bgym_step("noop()")

    def evaluate_js(self, js: str) -> Any:
        return self.page.evaluate(js)

    def page_obs(self) -> Observation:
        self._last_obs = self._extract_bgym_obs()
        self._last_info = {"source": "page_obs"}
        self._last_reward = 0.0
        self._last_terminated = False
        return self._bgym_obs_to_cube_obs(self._last_obs)

    # === Lifecycle ===

    def reset(self) -> None:
        self._close_runtime()
        self._create_runtime()
        self._wait_dom_loaded()
        self._last_obs = self._extract_bgym_obs()
        self._last_info = {"source": "reset"}
        self._last_reward = 0.0
        self._last_terminated = False

    def close(self) -> None:
        self._close_runtime()
        self._last_obs = None
        self._last_info = None
        self._last_reward = 0.0
        self._last_terminated = False

    def _create_runtime(self) -> None:
        self._session = self.config.browser.make()
        self._session.playwright.selectors.set_test_id_attribute(BROWSERGYM_ID_ATTRIBUTE)

    def _close_runtime(self) -> None:
        if self._session is not None:
            self.session.stop()
            self._session = None

    def _wait_dom_loaded(self) -> None:
        if self._session is None:
            return
        for page in self.session.context.pages:
            try:
                page.wait_for_load_state("domcontentloaded", timeout=1500)
            except Error:
                pass
            for frame in page.frames:
                # un necessary to wait for detached frames, and waiting on them raises a timeout error, so skip them
                if frame.is_detached():
                    continue
                try:
                    frame.wait_for_load_state("domcontentloaded", timeout=1500)
                except Error:
                    pass

    # === Core bgym step execution ===

    def _execute_bgym_step(self, action_str: str) -> str:
        """Execute a BrowserGym action string and return a result message.

        Captures three error channels:
        - Python exceptions (e.g. TimeoutError when element not found)
        - report_infeasible_instructions callback (bgym soft failures)
        - send_message_to_user callback (bgym task-completion signals)
        """
        logger.info(f"Execute bgym step: {action_str}")
        result = "Success"

        def send_message_to_user(_: str) -> None:
            assert False, "send_message_to_user should not be called"

        def report_infeasible_instructions(_: str) -> None:
            assert False, "report_infeasible_instructions should not be called"

        try:
            code = self._action_set.to_python_code(action_str)
            execute_python_code(
                code=code,
                page=self.page,
                send_message_to_user=send_message_to_user,
                report_infeasible_instructions=report_infeasible_instructions,
            )
            self._last_info = {
                "source": "action",
                "action": action_str,
                "action_error": "",
            }
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            self._last_info = {
                "source": "action",
                "action": action_str,
                "action_error": error_msg,
            }
            result = f"Failed: {error_msg}"

        self._last_obs = self._extract_bgym_obs()
        self._last_reward = 0.0
        self._last_terminated = False
        return result

    # === Observation extraction ===

    def _extract_bgym_obs(self) -> dict[str, Any]:
        page = self.page
        if self.config.pre_observation_delay > 0:
            time.sleep(self.config.pre_observation_delay)
        self._wait_dom_loaded()

        for retries_left in reversed(range(EXTRACT_OBS_MAX_TRIES)):
            try:
                _pre_extract(
                    page,
                    tags_to_mark=self.config.tags_to_mark,
                    lenient=(retries_left == 0),
                )
                dom = extract_dom_snapshot(page)
                axtree = extract_merged_axtree(page)
                focused_element_bid = extract_focused_element_bid(page)
                scale_factor = getattr(page, "_bgym_scale_factor", 1.0)
                need_extra = self.config.axtree_with_visible or self.config.axtree_with_clickable
                extra_properties = extract_dom_extra_properties(dom, scale_factor=scale_factor) if need_extra else {}
            except (Error, MarkingError):
                if retries_left > 0:
                    logger.warning(
                        f"Error extracting BrowserGym observation. Retrying ({retries_left}/{EXTRACT_OBS_MAX_TRIES})."
                    )
                    _post_extract(page)
                    time.sleep(0.5)
                    continue
                raise
            break

        _post_extract(page)
        obs: dict[str, Any] = {
            "dom_object": dom,
            "axtree_object": axtree,
            "extra_element_properties": extra_properties,
            "focused_element_bid": focused_element_bid,
            "last_action_error": (self._last_info.get("action_error", "") if self._last_info else ""),
        }
        if self.config.use_screenshot:
            obs["screenshot"] = extract_screenshot(page)
        return obs

    def _bgym_obs_to_cube_obs(self, bgym_obs: dict[str, Any]) -> Observation:
        """Convert BrowserGym observation dict to a cube Observation."""
        obs = Observation()

        extra_properties = bgym_obs.get("extra_element_properties", {})

        # HTML
        if self.config.use_html and "dom_object" in bgym_obs:
            dom_obj = bgym_obs["dom_object"]
            html_str = flatten_dom_to_str(dom_obj, extra_properties=extra_properties)
            if self.config.prune_html:
                html_str = prune_html(html_str)
            obs.contents.append(Content.from_data(html_str, name="pruned_html"))

        # Focused element (placed before axtree so axtree+screenshot remain last)
        if "focused_element_bid" in bgym_obs:
            focused_bid = bgym_obs["focused_element_bid"]
            if focused_bid:
                obs.contents.append(Content.from_data(focused_bid, name="focused_element"))

        # Accessibility tree
        if self.config.use_axtree and "axtree_object" in bgym_obs:
            axtree_obj = bgym_obs["axtree_object"]
            if axtree_obj:
                axtree_str = flatten_axtree_to_str(
                    axtree_obj,
                    extra_properties=extra_properties,
                    with_visible=self.config.axtree_with_visible,
                    with_clickable=self.config.axtree_with_clickable,
                )
                obs.contents.append(Content.from_data(axtree_str, name="axtree_txt"))

        # Screenshot
        if self.config.use_screenshot and "screenshot" in bgym_obs:
            screenshot = bgym_obs["screenshot"]
            if isinstance(screenshot, Image.Image):
                obs.contents.append(Content.from_data(screenshot, name="screenshot"))
            elif isinstance(screenshot, np.ndarray):
                screenshot_img = Image.fromarray(screenshot)
                obs.contents.append(Content.from_data(screenshot_img, name="screenshot"))

        # Last action error
        if "last_action_error" in bgym_obs:
            error = bgym_obs["last_action_error"]
            if error:
                obs.contents.append(Content.from_data(str(error), name="last_action_error"))

        # User messages from send_msg_to_user callback
        if self._last_info and self._last_info.get("user_messages"):
            for msg in self._last_info["user_messages"]:
                obs.contents.append(Content.from_data(msg, name="user_message"))

        return obs


class ExtraWebActionsTool(Tool):
    """Extra browser actions that complement BrowserGym's built-in set.

    Holds a reference to a BgymTool to share its page and observation
    extraction — both tools operate on the same browser session.

    Intentionally uses composition rather than inheritance: this tool adds
    actions on top of BgymTool but is not a BgymTool. It depends on BgymTool
    specifically (not a generic BrowserTool) because it accesses .page and
    .page_obs() which are BgymTool-specific.

    Experimental: if BrowserGym adds native keyboard-event support in a future
    release, keyboard_type_into can be removed and this class may become empty.
    """

    def __init__(self, browser: BgymTool) -> None:
        self._browser = browser

    def execute_action(self, action: Action) -> Observation | StepError:
        method = self.get_action_method(action)
        try:
            result = str(method(**action.arguments) or "Success")
        except Exception as e:
            return StepError.from_exception(e)
        action_obs = Observation(contents=[Content.from_data(result, tool_call_id=action.id)])
        return action_obs + self._browser.page_obs()

    @tool_action
    def keyboard_type_into(self, bid: str, text: str) -> str:
        """Type text into an element character-by-character, firing keyboard events per character.

        Use this instead of fill() for fields that show autocomplete suggestions or dynamic
        dropdowns as you type — fill() sets the value directly and bypasses keyboard events.
        After typing, call noop() to wait for suggestions to appear, then click the suggestion.
        """
        logger.info(f"keyboard_type_into: bid={bid!r} text={text!r}")
        try:
            get_elem_by_bid(self._browser.page, bid).press_sequentially(text, delay=50)
            return "Success"
        except Error as e:
            return f"Failed: {type(e).__name__}: {e}"

    @tool_action
    def js_eval(self, code: str, frame: str = "main") -> str:
        """Evaluate JavaScript in the browser and return the JSON-serialized result.

        Useful for inspecting DOM state, reading localStorage, checking field values,
        or diagnosing why an action isn't working as expected.

        Args:
            code: JavaScript expression to evaluate. The result is JSON-serialized.
                  Example: "document.title"
                  Example: "JSON.stringify(localStorage)"
                  Example: "g_form.getUniqueValue()"
            frame: Frame to evaluate in. "main" = top-level page. Any other string
                   is matched against iframe names (e.g. "gsft_main" for ServiceNow).
        """
        try:
            if frame == "main":
                target = self._browser.page
            else:
                target = next((f for f in self._browser.page.frames if f.name == frame), None)
                if target is None:
                    return f"Failed: frame {frame!r} not found"
            raw = target.evaluate(
                f"() => {{ try {{ return {code}; }} catch(e) {{ return 'JS error: ' + e.message; }} }}"
            )
            return json.dumps(raw, default=str)
        except Error as e:
            return f"Failed: {type(e).__name__}: {e}"


class ExtendedBrowserConfig(ToolConfig):
    """BrowserGym tool bundled with ExtraWebActionsTool, returned as a flat Toolbox.

    Drop-in replacement for BgymToolConfig when extra web actions are needed.
    When nested inside ToolboxConfig, the Toolbox is automatically flattened.
    """

    browser: BgymToolConfig = Field(default_factory=BgymToolConfig)

    def make(self, container: Any = None) -> Toolbox:
        bgym = self.browser.make(container)
        extra = ExtraWebActionsTool(bgym)
        return Toolbox([bgym, extra])


# === Module-level helpers ===


def _build_action_schemas(action_set: "HighLevelActionSet") -> list[ActionSchema]:
    """Convert bgym's HighLevelActionSet to a list of ActionSchema objects."""
    tool_descs = action_set.to_tool_description(api="openai")
    schemas = []
    for desc in tool_descs:
        # "type": "function" is at the top-level desc dict, not inside parameters.
        # parameters already has "type": "object" which Azure/OpenAI require — don't remove it.
        params = desc.get("parameters", {})
        name = desc["name"]
        schemas.append(ActionSchema(name=name, description=desc.get("description", name), parameters=params))
    return schemas


def _action_to_bgym_string(action: Action) -> str:
    """Serialise a cube Action into a BrowserGym action string like 'click(bid="a51")'."""
    args_parts = []
    for key, value in action.arguments.items():
        args_parts.append(f"{key}={repr(value)}")
    return f"{action.name}({', '.join(args_parts)})"
