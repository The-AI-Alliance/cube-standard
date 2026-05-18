"""Regression tests for cube-standard #182 (auto-fix L1).

Invariant under test: ``ToolkitInfraConfig`` MUST reap orphaned EAI jobs
by reading cloud tags (``resource/spec.md`` inv. 3-4) — ``cleanup(run_id)``
kills only matching jobs and no-ops gracefully; ``cleanup_stale()`` kills
expired/old jobs by tag; ``launch()`` tags every job + sets a server-side
``--max-run-time``. No live cluster: ``_run_eai`` is mocked.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from cube_infra_toolkit.toolkit import (
    ToolkitInfraConfig,
    _parse_epoch,  # noqa: PLC2701
)

from cube.resource import DockerServiceConfig


def _patch_store_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("cube.provision_store._DEFAULT_STORE_DIR", tmp_path / "provisions")


def _infra() -> ToolkitInfraConfig:
    return ToolkitInfraConfig(profile="test", eai_path="eai", cube_data=None, default_ttl_seconds=3600)


def _job_line(jid: str, tags: dict[str, str | None], created: str = "2026-05-18T12:00:00Z") -> str:
    return json.dumps(
        {
            "id": jid,
            "created": created,
            "tags": [{"key": k, "value": v} if v is not None else {"key": k} for k, v in tags.items()],
        }
    )


def _cp(stdout: str = "", rc: int = 0, stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["eai"], rc, stdout, stderr)


def _mock_eai(listing: list[str]) -> tuple[list[list[str]], object]:
    """Return (kills, side_effect). side_effect serves a JSONL listing and
    records every `job kill <id>` invocation."""
    kills: list[list[str]] = []

    def side_effect(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if args[:2] == ["job", "ls"]:
            return _cp("\n".join(listing))
        if args[:2] == ["job", "kill"]:
            kills.append(args)
            return _cp()
        return _cp()

    return kills, side_effect


# ── _parse_epoch ──────────────────────────────────────────────────────────────


def test_parse_epoch() -> None:
    assert _parse_epoch("2026-05-18T12:36:49Z") == 1779107809
    assert _parse_epoch(None) is None
    assert _parse_epoch("not-a-date") is None


# ── cleanup(run_id) ───────────────────────────────────────────────────────────


def test_cleanup_kills_only_matching_run_id() -> None:
    listing = [
        _job_line("aaaaaaaa-0000-0000-0000-000000000001", {"cube_managed": None, "cube_run_id": "RUN-A"}),
        _job_line("bbbbbbbb-0000-0000-0000-000000000002", {"cube_managed": None, "cube_run_id": "RUN-B"}),
        _job_line("cccccccc-0000-0000-0000-000000000003", {"cube_managed": None, "cube_run_id": "RUN-A"}),
    ]
    kills, se = _mock_eai(listing)
    with patch("cube_infra_toolkit.toolkit._run_eai", side_effect=se):
        _infra().cleanup("RUN-A")
    killed = {k[2] for k in kills}
    assert killed == {
        "aaaaaaaa-0000-0000-0000-000000000001",
        "cccccccc-0000-0000-0000-000000000003",
    }


def test_cleanup_graceful_when_no_match() -> None:
    """resource/spec.md invariant 3: safe to call when nothing matches."""
    listing = [_job_line("aaaaaaaa-0000-0000-0000-000000000001", {"cube_managed": None, "cube_run_id": "RUN-X"})]
    kills, se = _mock_eai(listing)
    with patch("cube_infra_toolkit.toolkit._run_eai", side_effect=se):
        _infra().cleanup("RUN-DOES-NOT-EXIST")  # must not raise
    assert kills == []


# ── cleanup_stale() ───────────────────────────────────────────────────────────


def test_cleanup_stale_kills_expired_by_tag() -> None:
    now = int(time.time())
    listing = [
        _job_line(
            "aaaaaaaa-0000-0000-0000-000000000001", {"cube_managed": None, "cube_expires_at": str(now - 100)}
        ),  # expired
        _job_line(
            "bbbbbbbb-0000-0000-0000-000000000002", {"cube_managed": None, "cube_expires_at": str(now + 9999)}
        ),  # fresh
    ]
    kills, se = _mock_eai(listing)
    with patch("cube_infra_toolkit.toolkit._run_eai", side_effect=se):
        killed = _infra().cleanup_stale()
    assert killed == ["aaaaaaaa-0000-0000-0000-000000000001"]


def test_cleanup_stale_kills_old_by_max_age() -> None:
    """No expiry tag, but older than max_age_seconds -> reaped."""
    listing = [
        _job_line("dddddddd-0000-0000-0000-000000000004", {"cube_managed": None}, created="2000-01-01T00:00:00Z"),
    ]
    kills, se = _mock_eai(listing)
    with patch("cube_infra_toolkit.toolkit._run_eai", side_effect=se):
        killed = _infra().cleanup_stale(max_age_seconds=3600)
    assert killed == ["dddddddd-0000-0000-0000-000000000004"]


def test_cleanup_stale_spares_fresh() -> None:
    now = int(time.time())
    listing = [
        _job_line("eeeeeeee-0000-0000-0000-000000000005", {"cube_managed": None, "cube_expires_at": str(now + 9999)}),
    ]
    kills, se = _mock_eai(listing)
    with patch("cube_infra_toolkit.toolkit._run_eai", side_effect=se):
        killed = _infra().cleanup_stale()
    assert killed == []
    assert kills == []


# ── list_active(run_id) ───────────────────────────────────────────────────────


def test_list_active_filters_by_run_id() -> None:
    listing = [
        _job_line("aaaaaaaa-0000-0000-0000-000000000001", {"cube_managed": None, "cube_run_id": "RUN-A"}),
        _job_line("bbbbbbbb-0000-0000-0000-000000000002", {"cube_managed": None, "cube_run_id": "RUN-B"}),
    ]
    _, se = _mock_eai(listing)
    with patch("cube_infra_toolkit.toolkit._run_eai", side_effect=se):
        handles = _infra().list_active("RUN-B")
    assert [h.id for h in handles] == ["bbbbbbbb-0000-0000-0000-000000000002"]
    assert handles[0].run_id == "RUN-B"


# ── launch() tagging invariant ────────────────────────────────────────────────


def test_launch_tags_job_and_sets_max_run_time(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_store_path(monkeypatch, tmp_path)
    captured: list[list[str]] = []

    def se(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if args[:2] == ["job", "new"]:
            captured.append(args)
            return _cp(rc=1, stderr="stop here — argv captured")  # abort before _wait_for_running
        return _cp()

    infra = _infra()  # default_ttl_seconds=3600
    resource = DockerServiceConfig(name="t", scope="task", docker_images=["alpine:3"])
    with patch("cube_infra_toolkit.toolkit._run_eai", side_effect=se):
        with pytest.raises(Exception):  # noqa: B017,PT011 — rc!=0 -> ContainerLaunchError
            infra.launch(resource)

    assert captured, "eai job new was never invoked"
    argv = captured[0]
    joined = " ".join(argv)
    assert "--tag cube_managed" in joined
    assert "cube_run_id=" in joined
    assert "cube_expires_at=" in joined
    # server-side hard TTL == effective_ttl (infra.default_ttl_seconds=3600)
    assert "--max-run-time" in argv
    assert argv[argv.index("--max-run-time") + 1] == "3600"
