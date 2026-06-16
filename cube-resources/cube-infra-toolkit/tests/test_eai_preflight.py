"""auto-fix(180): missing eai CLI → actionable ContainerLaunchError, not raw FileNotFoundError."""

import pytest
from cube_infra_toolkit.toolkit import _resolve_eai_account

from cube.container import ContainerLaunchError


def test_missing_eai_raises_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cube_infra_toolkit.toolkit.shutil.which", lambda _: None)
    with pytest.raises(ContainerLaunchError) as ei:
        _resolve_eai_account("eai", profile=None)
    msg = str(ei.value)
    assert "eai" in msg and "PATH" in msg and "not a pip dependency" in msg.lower() or "eai_path" in msg


def test_present_eai_passes_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    """When eai resolves, preflight is transparent (we stop before the real call
    by making the subprocess fail fast — proves the preflight didn't raise)."""
    monkeypatch.setattr("cube_infra_toolkit.toolkit.shutil.which", lambda _: "/usr/bin/eai")

    class _R:
        returncode = 1
        stdout = b""
        stderr = b"boom"

    monkeypatch.setattr("cube_infra_toolkit.toolkit.subprocess.run", lambda *a, **k: _R())
    with pytest.raises(ContainerLaunchError) as ei:
        _resolve_eai_account("eai", profile=None)
    assert "user get" in str(ei.value)  # got past preflight to the real call
