"""cube-browser-tool — concrete browser tool implementations for cube-standard.

Phase 1 (sync Playwright):
    PlaywrightConfig    — serializable config; call .make() to get a SyncPlaywrightTool
    SyncPlaywrightTool  — synchronous Playwright tool satisfying AbstractBrowserTool

Phase 2 (async Playwright):
    AsyncPlaywrightTool — async variant (not yet implemented)

Phase 3 (BrowserGym, optional dep):
    BgymToolConfig, BgymTool — import from cube_browser_tool.bgym_tool;
    requires ``pip install cube-browser-tool[bgym]``

Action space ABCs (for implementing custom tools):
    BrowserActionSpace    — CSS-selector-based action contract
    BidBrowserActionSpace — BID-based action contract (BrowserGym)
"""

from cube_browser_tool.action_spaces import BidBrowserActionSpace, BrowserActionSpace
from cube_browser_tool.playwright_tool import PlaywrightConfig, SyncPlaywrightTool

__all__ = [
    "PlaywrightConfig",
    "SyncPlaywrightTool",
    "BrowserActionSpace",
    "BidBrowserActionSpace",
]
