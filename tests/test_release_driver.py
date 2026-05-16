"""Mocked simulation of scripts/release.py's --execute path.

The dry-run/planning path is validated against the live repos; this covers
the side-effecting control flow (tag push → PyPI wait → tier gating →
idempotent skip → block/timeout → partial-run summary) WITHOUT touching git
or PyPI, so the never-run-for-real --execute path has real coverage.

git / PyPI / sleep are all stubbed. No network, no subprocess.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Load scripts/release.py as a module (it's a script, not a package member).
# It must be registered in sys.modules before exec_module so @dataclass can
# resolve cls.__module__ under `from __future__ import annotations`.
_SPEC = importlib.util.spec_from_file_location(
    "release_driver", Path(__file__).resolve().parents[1] / "scripts" / "release.py"
)
release = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = release
_SPEC.loader.exec_module(release)


def _pkg(dist: str, tier: int) -> "release.Package":
    return release.Package(
        dist=dist,
        repo="cube-standard",
        pyproject=Path(f"/fake/{dist}/pyproject.toml"),
        tag_prefix=dist,
        tier=tier,
    )


def _plan(dist: str, tier: int, version: str = "1.0.0") -> "release.Plan":
    return release.Plan(pkg=_pkg(dist, tier), current=version, latest_tag=None, state="RELEASE", reason="")


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(release, "sleep", lambda *_: None)


# --------------------------------------------------------------------------- #
# do_release — the irreversible bit
# --------------------------------------------------------------------------- #


def test_fresh_release_pushes_tag_and_waits(monkeypatch, no_sleep, capsys):
    calls: list[tuple] = []
    monkeypatch.setattr(release, "_git_ok", lambda *a: False)  # tag does not exist
    monkeypatch.setattr(release, "_git", lambda repo, *a: calls.append(a) or "deadbeefcafe")
    # not on PyPI on first check, then appears
    seq = iter([False, True])
    monkeypatch.setattr(release, "_pypi_has", lambda *_: next(seq))

    status = release.do_release(_plan("cube-standard", 1), Path("/fake"), pypi_timeout_s=60)

    assert status == "released"
    assert ("tag", "cube-standard/v1.0.0", "deadbeefcafe") in calls
    assert ("push", "origin", "cube-standard/v1.0.0") in calls
    out = capsys.readouterr().out
    assert "[tag] pushing cube-standard/v1.0.0" in out
    assert "[done] cube-standard==1.0.0 published" in out


def test_already_published_is_idempotent_skip(monkeypatch, no_sleep, capsys):
    monkeypatch.setattr(release, "_git_ok", lambda *a: False)
    monkeypatch.setattr(release, "_git", lambda repo, *a: "head000")
    monkeypatch.setattr(release, "_pypi_has", lambda *_: True)  # already there

    status = release.do_release(_plan("cube-web-tool", 3), Path("/fake"), pypi_timeout_s=60)

    assert status == "skipped"
    assert "[skip] cube-web-tool==1.0.0 already on PyPI" in capsys.readouterr().out


def test_tag_at_head_then_publishes_counts_as_skip(monkeypatch, no_sleep):
    # Tag already exists pointing at HEAD (a prior run pushed it), not yet on PyPI.
    monkeypatch.setattr(release, "_git_ok", lambda *a: True)
    monkeypatch.setattr(release, "_git", lambda repo, *a: "samehead")  # HEAD == tagged commit
    seq = iter([False, True])
    monkeypatch.setattr(release, "_pypi_has", lambda *_: next(seq))

    status = release.do_release(_plan("cube-standard", 1), Path("/fake"), pypi_timeout_s=60)
    # tag wasn't pushed this run → skipped, even though it landed on PyPI now
    assert status == "skipped"


def test_tag_at_different_commit_blocks(monkeypatch):
    monkeypatch.setattr(release, "_git_ok", lambda *a: True)

    def fake_git(repo, *a):
        return "headAAA" if a[:2] == ("rev-parse", "HEAD") else "tagBBB"

    monkeypatch.setattr(release, "_git", fake_git)
    monkeypatch.setattr(release, "_pypi_has", lambda *_: False)

    with pytest.raises(SystemExit, match=r"\[BLOCKED\].*different commit"):
        release.do_release(_plan("cube-standard", 1), Path("/fake"), pypi_timeout_s=60)


def test_pypi_timeout_blocks_with_tag_pushed_note(monkeypatch, no_sleep):
    monkeypatch.setattr(release, "_git_ok", lambda *a: False)
    monkeypatch.setattr(release, "_git", lambda repo, *a: "head111")
    monkeypatch.setattr(release, "_pypi_has", lambda *_: False)  # never appears
    t = iter([1000, 1001, 9999])  # deadline = 1000+0; next time() already past
    monkeypatch.setattr(release, "time", lambda: next(t))

    with pytest.raises(SystemExit, match=r"\[BLOCKED\].*WAS pushed"):
        release.do_release(_plan("cube-standard", 1), Path("/fake"), pypi_timeout_s=0)


# --------------------------------------------------------------------------- #
# execute — tier ordering + summary (incl. partial/aborted run)
# --------------------------------------------------------------------------- #


def test_execute_runs_tiers_in_order_and_summarizes(monkeypatch, capsys):
    order: list[str] = []

    def fake_do_release(pl, repo_path, timeout):
        order.append(f"{pl.pkg.dist}@t{pl.pkg.tier}")
        return "released"

    monkeypatch.setattr(release, "do_release", fake_do_release)
    plans = [_plan("cubes-x", 5), _plan("cube-standard", 1), _plan("cube-browser-tool", 3)]

    release.execute(plans, {"cube-standard": Path("/fake")}, pypi_timeout_s=60)

    assert order == ["cube-standard@t1", "cube-browser-tool@t3", "cubes-x@t5"]  # tier-ascending
    out = capsys.readouterr().out
    assert "[summary] released  : cube-standard==1.0.0, cube-browser-tool==1.0.0, cubes-x==1.0.0" in out
    assert "[summary] remaining : (none)" in out


def test_execute_abort_prints_partial_summary(monkeypatch, capsys):
    def fake_do_release(pl, repo_path, timeout):
        if pl.pkg.dist == "cube-browser-tool":
            raise SystemExit("[BLOCKED] cube-browser-tool==1.0.0: PyPI timeout")
        return "released"

    monkeypatch.setattr(release, "do_release", fake_do_release)
    plans = [_plan("cube-standard", 1), _plan("cube-browser-tool", 3), _plan("cubes-x", 5)]

    with pytest.raises(SystemExit, match=r"\[BLOCKED\]"):
        release.execute(plans, {"cube-standard": Path("/fake")}, pypi_timeout_s=60)

    out = capsys.readouterr().out
    # tier 1 done before the tier-3 failure; tier-3 + tier-5 remain; idempotency note shown
    assert "[summary] released  : cube-standard==1.0.0" in out
    assert "cube-browser-tool==1.0.0" in out  # listed under remaining
    assert "cubes-x==1.0.0" in out
    assert "re-run is safe: idempotent" in out
