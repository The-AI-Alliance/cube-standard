"""Tests for cube.container module-level helpers."""

from unittest.mock import MagicMock

import pytest

from cube.container import ExecResult, relocate_if_readonly


def _container(probe: ExecResult, copy: ExecResult) -> MagicMock:
    c = MagicMock()
    c.exec.side_effect = [probe, copy]
    return c


def test_relocate_returns_original_when_writable() -> None:
    c = MagicMock()
    c.exec.return_value = ExecResult(stdout="W", exit_code=0)
    assert relocate_if_readonly(c, "/app", "/tmp/app") == "/app"
    c.exec.assert_called_once()  # only the probe; no copy


def test_relocate_copies_and_returns_new_wd_on_success() -> None:
    c = _container(ExecResult(stdout="R", exit_code=0), ExecResult(exit_code=0))
    assert relocate_if_readonly(c, "/app", "/tmp/app") == "/tmp/app"


def test_relocate_raises_when_copy_fails_instead_of_returning_phantom_dir() -> None:
    """A failed `cp -a` must raise — never silently return a non-existent dir."""
    c = _container(
        ExecResult(stdout="R", exit_code=0),
        ExecResult(exit_code=1, stderr="cp: cannot stat '/app': No such file or directory"),
    )
    with pytest.raises(RuntimeError, match=r"relocate_if_readonly.*failed.*was not created"):
        relocate_if_readonly(c, "/app", "/tmp/app")
