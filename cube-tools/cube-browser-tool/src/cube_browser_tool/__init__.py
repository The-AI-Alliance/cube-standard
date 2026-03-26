"""cube-browser-tool — concrete browser tool implementations for cube-standard.

Sync Playwright:
    PlaywrightConfig    — serializable config; call .make() to get a SyncPlaywrightTool
    SyncPlaywrightTool  — synchronous Playwright tool satisfying BrowserTool

Async Playwright:
    AsyncPlaywrightConfig — serializable config; call .make() to get an AsyncPlaywrightTool
    AsyncPlaywrightTool   — async Playwright tool

BrowserGym (optional dep):
    BgymToolConfig, BgymTool — import from cube_browser_tool.bgym_tool;
    requires ``pip install cube-browser-tool[bgym]``

Action space ABCs (for implementing custom tools):
    BrowserActionSpace      — CSS-selector-based action contract (sync)
    AsyncBrowserActionSpace — CSS-selector-based action contract (async)
    BidBrowserActionSpace   — BID-based action contract (BrowserGym)
"""

from cube_browser_tool.action_spaces import AsyncBrowserActionSpace, BidBrowserActionSpace, BrowserActionSpace
from cube_browser_tool.playwright_tool import (
    AsyncPlaywrightConfig,
    AsyncPlaywrightTool,
    PlaywrightConfig,
    SyncPlaywrightTool,
)

__all__ = [
    "PlaywrightConfig",
    "SyncPlaywrightTool",
    "AsyncPlaywrightConfig",
    "AsyncPlaywrightTool",
    "BrowserActionSpace",
    "AsyncBrowserActionSpace",
    "BidBrowserActionSpace",
]
