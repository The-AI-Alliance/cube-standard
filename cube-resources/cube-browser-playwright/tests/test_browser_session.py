"""Tests for browser session abstractions and PlaywrightSession implementation."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from cube.resources.browser_session import BrowserConfig, BrowserSession

from cube_browser_playwright.playwright_session import PlaywrightSession, PlaywrightSessionConfig, _read_cdp_url


class TestReadCdpUrl:
    def test_returns_url_from_port_file(self, tmp_path: Path) -> None:
        (tmp_path / "DevToolsActivePort").write_text("9222\n/devtools/browser/abc\n")
        assert _read_cdp_url(str(tmp_path)) == "http://localhost:9222"

    def test_raises_on_timeout(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="DevToolsActivePort"):
            _read_cdp_url(str(tmp_path))  # file never written


class TestBrowserSession:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            BrowserSession()  # type: ignore[abstract]


class TestBrowserConfig:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            BrowserConfig()  # type: ignore[abstract]


def _make_session(**kwargs: MagicMock) -> PlaywrightSession:
    defaults: dict = {
        "playwright": MagicMock(),
        "page": MagicMock(),
        "context": MagicMock(),
        "cdp_url": "http://localhost:1234",
        "user_data_dir": "/tmp/fake_dir",
    }
    return PlaywrightSession(**{**defaults, **kwargs})


class TestPlaywrightSession:
    def test_cdp_url_is_stored(self) -> None:
        session = _make_session(cdp_url="http://localhost:1234")
        assert session.cdp_url == "http://localhost:1234"

    def test_get_playwright_session_returns_page_and_context(self) -> None:
        mock_page = MagicMock()
        mock_context = MagicMock()
        session = _make_session(page=mock_page, context=mock_context)
        page, context = session.get_playwright_session()
        assert page is mock_page
        assert context is mock_context

    def test_stop_closes_context(self) -> None:
        mock_context = MagicMock()
        session = _make_session(context=mock_context)
        session.stop()
        mock_context.close.assert_called_once()

    def test_stop_calls_playwright_stop_even_when_context_close_raises(self) -> None:
        mock_pw = MagicMock()
        mock_context = MagicMock()
        mock_context.close.side_effect = RuntimeError("context close error")
        session = _make_session(playwright=mock_pw, context=mock_context)
        session.stop()  # should not raise
        mock_pw.stop.assert_called_once()

    def test_stop_stops_playwright(self) -> None:
        mock_pw = MagicMock()
        session = _make_session(playwright=mock_pw)
        session.stop()
        mock_pw.stop.assert_called_once()

    def test_stop_removes_temp_dir(self, tmp_path: Path) -> None:
        profile_dir = tmp_path / "profile"
        profile_dir.mkdir()
        session = _make_session(user_data_dir=str(profile_dir))
        session.stop()
        assert not profile_dir.exists()


class TestPlaywrightSessionConfig:
    def _make_mock_pw(self) -> MagicMock:
        mock_page = MagicMock()
        mock_context = MagicMock()
        mock_context.pages = [mock_page]
        mock_pw = MagicMock()
        mock_pw.chromium.launch_persistent_context.return_value = mock_context
        return mock_pw

    def _make_sync_playwright_mock(self, mock_pw: MagicMock) -> MagicMock:
        mock_sp_cm = MagicMock()
        mock_sp_cm.start.return_value = mock_pw
        mock_sp = MagicMock(return_value=mock_sp_cm)
        return mock_sp

    def test_make_returns_playwright_session(self) -> None:
        mock_pw = self._make_mock_pw()
        mock_sp = self._make_sync_playwright_mock(mock_pw)
        with (
            patch("cube_browser_playwright.playwright_session.sync_playwright", mock_sp),
            patch("cube_browser_playwright.playwright_session._read_cdp_url", return_value="http://localhost:1234"),
        ):
            session = PlaywrightSessionConfig().make()
        assert isinstance(session, PlaywrightSession)

    def test_make_injects_remote_debugging_port_0(self) -> None:
        mock_pw = self._make_mock_pw()
        mock_sp = self._make_sync_playwright_mock(mock_pw)
        with (
            patch("cube_browser_playwright.playwright_session.sync_playwright", mock_sp),
            patch("cube_browser_playwright.playwright_session._read_cdp_url", return_value="http://localhost:1234"),
        ):
            PlaywrightSessionConfig().make()
        launch_args = mock_pw.chromium.launch_persistent_context.call_args[1]["args"]
        assert "--remote-debugging-port=0" in launch_args

    def test_make_cdp_url_comes_from_read_cdp_url(self) -> None:
        mock_pw = self._make_mock_pw()
        mock_sp = self._make_sync_playwright_mock(mock_pw)
        with (
            patch("cube_browser_playwright.playwright_session.sync_playwright", mock_sp),
            patch(
                "cube_browser_playwright.playwright_session._read_cdp_url", return_value="http://localhost:9876"
            ) as mock_read,
        ):
            session = PlaywrightSessionConfig().make()
        mock_read.assert_called_once()
        assert session.cdp_url == "http://localhost:9876"

    def test_make_passes_headless_flag(self) -> None:
        mock_pw = self._make_mock_pw()
        mock_sp = self._make_sync_playwright_mock(mock_pw)
        with (
            patch("cube_browser_playwright.playwright_session.sync_playwright", mock_sp),
            patch("cube_browser_playwright.playwright_session._read_cdp_url", return_value="http://localhost:1234"),
        ):
            PlaywrightSessionConfig(headless=False).make()
        assert mock_pw.chromium.launch_persistent_context.call_args[1]["headless"] is False
