"""Cross-repo release driver for cube-standard + cube-harness.

============================================================================
STATUS: NOT FULLY TESTED. The dry-run / planning path (preflight, manifest,
version-vs-tag, BLOCKED detection) is validated against the live repos. The
``--execute`` path (tag push → PyPI wait → tier gating) has NEVER run for
real — it is covered only by a mocked simulation test
(``tests/test_release_driver.py``); no real tag has been pushed and nothing
has been published through it. First real use: run the default dry-run,
eyeball the plan, then ``--execute --only <one package>`` supervised, tier
by tier. It is idempotent — re-running after a partial release skips
completed packages and resumes at the next tier.
============================================================================

Releases are tag-driven and per-package: pushing ``<prefix>/v<version>`` makes
each repo's ``release.yml`` build + publish that package to PyPI. The packages
form a dependency graph spanning two repos, so they must be published in
topological order — otherwise a downstream package lands on PyPI pinning an
upstream version that isn't up yet, and ``pip install`` breaks for the window
in between.

This script automates that runbook. It does NOT modify ``release.yml`` (the
publish path is untouched) — it only decides *what* to tag, in *what order*,
pushes the tags tier by tier, and waits for each tier to appear on PyPI before
starting the next.

Philosophy: **dry-run by default; never guess.** Any ambiguity (dirty repo,
version not bumped, tag already at a different commit, PyPI timeout) stops the
run with a precise, actionable message rather than doing something clever. The
worst case is "the script did 95% and told you exactly what's left."

Usage::

    # Show the ordered plan, no side effects (default):
    python scripts/release.py

    # Actually push tags + wait for PyPI, tier by tier:
    python scripts/release.py --execute

    # Restrict to specific packages:
    python scripts/release.py --only cube-standard --only cube-browser-tool

Topological tiers (lower publishes first):

    1  cube-standard            (cube-standard repo)
    2  cube-resources/*         (cube-standard repo)  — depend on cube-standard
    3  cube-tools/*             (cube-standard repo)  — depend on cube-standard[, cube-resources]
    4  cube-harness             (cube-harness repo)   — depends on 1–3
    5  cubes/*                  (cube-harness repo)   — depend on 1 + 3
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from time import sleep, time

import requests

try:
    from packaging.version import InvalidVersion, Version
except ModuleNotFoundError:  # pragma: no cover - packaging is ubiquitous in build envs
    print(
        "ERROR: this script needs `packaging` (near-universal build dep). Install it: pip install packaging",
        file=sys.stderr,
    )
    raise SystemExit(2) from None

PYPI_POLL_INTERVAL_S = 15


@dataclass(frozen=True)
class Package:
    """One releasable package: where it lives and how its tag is shaped."""

    dist: str  # PyPI distribution name (the `name` in pyproject)
    repo: str  # "cube-standard" | "cube-harness"
    pyproject: Path  # absolute path to its pyproject.toml
    tag_prefix: str  # tag = f"{tag_prefix}/v{version}"
    tier: int  # publish order; lower first


@dataclass
class Plan:
    """Per-package decision the dry-run prints and --execute acts on."""

    pkg: Package
    current: str  # version in pyproject
    latest_tag: str | None  # latest released version for this tag_prefix
    state: str  # RELEASE | UP-TO-DATE | BLOCKED
    reason: str = ""  # populated for BLOCKED / informational


# --------------------------------------------------------------------------- #
# git / shell helpers
# --------------------------------------------------------------------------- #


def _git(repo: Path, *args: str) -> str:
    """Run a git command in ``repo`` and return stripped stdout (raises on error)."""
    out = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def _git_ok(repo: Path, *args: str) -> bool:
    """Run a git command, return True iff it exited 0 (no raise)."""
    return (
        subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
        ).returncode
        == 0
    )


def _read_version(pyproject: Path) -> str:
    with pyproject.open("rb") as fh:
        data = tomllib.load(fh)
    return data["project"]["version"]


def _read_name(pyproject: Path) -> str:
    with pyproject.open("rb") as fh:
        data = tomllib.load(fh)
    return data["project"]["name"]


def _latest_tag_version(repo: Path, tag_prefix: str) -> str | None:
    """Highest released version for ``<tag_prefix>/v*`` (PEP 440 ordering), or None."""
    raw = _git(repo, "tag", "--list", f"{tag_prefix}/v*")
    versions: list[Version] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        ver_str = line[len(tag_prefix) + 2 :]  # strip "<prefix>/v"
        try:
            versions.append(Version(ver_str))
        except InvalidVersion:
            continue
    if not versions:
        return None
    return str(max(versions))


def _commits_touching_since(repo: Path, rel_path: str, tag: str) -> int:
    """Count commits touching ``rel_path`` since ``tag`` (0 if tag missing)."""
    if not _git_ok(repo, "rev-parse", "--verify", f"{tag}^{{commit}}"):
        return 0
    out = _git(repo, "rev-list", "--count", f"{tag}..HEAD", "--", rel_path)
    return int(out or "0")


def _pypi_has(dist: str, version: str) -> bool:
    try:
        r = requests.get(f"https://pypi.org/pypi/{dist}/json", timeout=20)
    except requests.RequestException:
        return False
    if r.status_code != 200:
        return False
    return version in r.json().get("releases", {})


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #


def build_manifest(cube_standard: Path, cube_harness: Path) -> list[Package]:
    """Discover every releasable package across both repos, with its tier."""
    pkgs: list[Package] = []

    # Tier 1 — cube-standard core
    pkgs.append(
        Package(
            dist=_read_name(cube_standard / "pyproject.toml"),
            repo="cube-standard",
            pyproject=cube_standard / "pyproject.toml",
            tag_prefix="cube-standard",
            tier=1,
        )
    )

    # Tier 2 — cube-resources/*  | Tier 3 — cube-tools/*
    for sub, tier in (("cube-resources", 2), ("cube-tools", 3)):
        for pp in sorted((cube_standard / sub).glob("*/pyproject.toml")):
            d = pp.parent.name
            pkgs.append(
                Package(
                    dist=_read_name(pp),
                    repo="cube-standard",
                    pyproject=pp,
                    tag_prefix=f"{sub}/{d}",
                    tier=tier,
                )
            )

    # Tier 4 — cube-harness core
    pkgs.append(
        Package(
            dist=_read_name(cube_harness / "pyproject.toml"),
            repo="cube-harness",
            pyproject=cube_harness / "pyproject.toml",
            tag_prefix="cube-harness",
            tier=4,
        )
    )

    # Tier 5 — cubes/*
    for pp in sorted((cube_harness / "cubes").glob("*/pyproject.toml")):
        d = pp.parent.name
        pkgs.append(
            Package(
                dist=_read_name(pp),
                repo="cube-harness",
                pyproject=pp,
                tag_prefix=f"cubes/{d}",
                tier=5,
            )
        )

    return pkgs


# --------------------------------------------------------------------------- #
# preflight + planning
# --------------------------------------------------------------------------- #


def preflight(repo: Path, ref: str) -> list[str]:
    """Return a list of blocking problems for ``repo`` (empty == OK)."""
    problems: list[str] = []
    if not (repo / ".git").exists():
        return [f"{repo} is not a git repository"]
    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    if branch != ref:
        problems.append(f"{repo.name}: on '{branch}', expected '{ref}' (release ref)")
    if _git(repo, "status", "--porcelain"):
        problems.append(f"{repo.name}: working tree not clean — commit/stash first")
    _git_ok(repo, "fetch", "origin", ref, "--quiet")
    if _git_ok(repo, "rev-parse", "--verify", f"origin/{ref}"):
        local = _git(repo, "rev-parse", "HEAD")
        remote = _git(repo, "rev-parse", f"origin/{ref}")
        if local != remote:
            problems.append(f"{repo.name}: HEAD != origin/{ref} — push/pull so the tag matches the published commit")
    return problems


def plan_package(pkg: Package, repo_path: Path) -> Plan:
    cur = _read_version(pkg.pyproject)
    latest = _latest_tag_version(repo_path, pkg.tag_prefix)
    rel = pkg.pyproject.parent.relative_to(repo_path).as_posix() or "."

    if latest is None:
        if _pypi_has(pkg.dist, cur):
            return Plan(pkg, cur, None, "UP-TO-DATE", f"{pkg.dist}=={cur} already on PyPI (no tag found)")
        return Plan(pkg, cur, None, "RELEASE", "first release for this tag prefix")

    try:
        cur_v, latest_v = Version(cur), Version(latest)
    except InvalidVersion as e:
        return Plan(pkg, cur, latest, "BLOCKED", f"unparseable version: {e}")

    if cur_v == latest_v:
        # Already tagged at this version. Are there unreleased commits anyway?
        n = _commits_touching_since(repo_path, rel, f"{pkg.tag_prefix}/v{latest}")
        if n > 0:
            return Plan(
                pkg,
                cur,
                latest,
                "BLOCKED",
                f"{n} commit(s) touch {rel} since {pkg.tag_prefix}/v{latest} "
                f"but version is still {cur} — bump the version in {pkg.pyproject.name}",
            )
        return Plan(pkg, cur, latest, "UP-TO-DATE", f"released at v{latest}, no changes since")

    if cur_v < latest_v:
        return Plan(pkg, cur, latest, "BLOCKED", f"pyproject {cur} < latest tag {latest} — version went backwards")

    return Plan(pkg, cur, latest, "RELEASE", f"{latest} → {cur}")


# --------------------------------------------------------------------------- #
# execution
# --------------------------------------------------------------------------- #


def do_release(plan: Plan, repo_path: Path, pypi_timeout_s: int) -> str:
    """Tag + push one package, block until it's on PyPI. Idempotent.

    Returns ``"released"`` (tag pushed this run) or ``"skipped"`` (already
    published / tagged at HEAD before this run). Raises ``SystemExit`` with a
    ``[BLOCKED]`` message on any anomaly. All progress lines are prefixed with
    a greppable ``[token]`` so a supervising agent can read the outcome.
    """
    pkg, version = plan.pkg, plan.current
    tag = f"{pkg.tag_prefix}/v{version}"
    head = _git(repo_path, "rev-parse", "HEAD")
    label = f"{pkg.dist}=={version}"

    tag_existed = False
    if _git_ok(repo_path, "rev-parse", "--verify", f"{tag}^{{commit}}"):
        tagged = _git(repo_path, "rev-list", "-n", "1", tag)
        if tagged != head:
            raise SystemExit(
                f"[BLOCKED] {label}: tag {tag} already exists at {tagged[:9]} but HEAD is "
                f"{head[:9]}. Release was cut from a different commit, or the version needs "
                f"bumping. Refusing to move the tag — resolve manually, then re-run (idempotent)."
            )
        tag_existed = True
        print(f"[tag] {tag} already at HEAD — skipping push")
    else:
        print(f"[tag] pushing {tag} @ {head[:9]}")
        _git(repo_path, "tag", tag, head)
        _git(repo_path, "push", "origin", tag)

    if _pypi_has(pkg.dist, version):
        print(f"[skip] {label} already on PyPI")
        return "skipped"

    print(f"[wait] {label} — polling PyPI (release.yml building; timeout {pypi_timeout_s}s)")
    deadline = time() + pypi_timeout_s
    while time() < deadline:
        sleep(PYPI_POLL_INTERVAL_S)
        if _pypi_has(pkg.dist, version):
            print(f"[done] {label} published")
            return "skipped" if tag_existed else "released"
    raise SystemExit(
        f"[BLOCKED] {label}: timed out after {pypi_timeout_s}s waiting on PyPI. The tag {tag} "
        f"WAS pushed — check the release.yml run for {pkg.repo}. Re-run this script once it's up; "
        f"it is idempotent and will resume at the remaining tiers."
    )


def execute(to_release: list[Plan], repo_paths: dict[str, Path], pypi_timeout_s: int) -> None:
    """Release every plan tier by tier; a tier must be fully on PyPI before the next.

    The SUMMARY prints in ``finally`` so an aborted partial run is still legible
    (``do_release`` raises ``SystemExit`` on any block/timeout, which propagates
    after the summary).
    """
    released: list[str] = []
    skipped: list[str] = []
    try:
        for tier in sorted({pl.pkg.tier for pl in to_release}):
            tier_plans = [pl for pl in to_release if pl.pkg.tier == tier]
            print(f"\n== tier {tier}: {len(tier_plans)} package(s) ==")
            for pl in tier_plans:
                print(f"[release] {pl.pkg.dist}=={pl.current} ({pl.latest_tag or 'first'} -> {pl.current}) tier={tier}")
                status = do_release(pl, repo_paths[pl.pkg.repo], pypi_timeout_s)
                (released if status == "released" else skipped).append(f"{pl.pkg.dist}=={pl.current}")
    finally:
        _print_summary(to_release, released, skipped)


def _print_summary(to_release: list[Plan], released: list[str], skipped: list[str]) -> None:
    """Final, greppable post-condition — printed even on abort, so a partial run is legible."""
    done = set(released) | set(skipped)
    remaining = [f"{pl.pkg.dist}=={pl.current}" for pl in to_release if f"{pl.pkg.dist}=={pl.current}" not in done]
    print("\nSUMMARY")
    print(f"  [summary] released  : {', '.join(released) or '(none)'}")
    print(f"  [summary] skipped   : {', '.join(skipped) or '(none)'}  (already tagged/published)")
    print(f"  [summary] remaining : {', '.join(remaining) or '(none)'}")
    if remaining:
        print("  [summary] re-run is safe: idempotent — completed packages are skipped, resumes at the next tier")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--cube-standard",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="path to the cube-standard repo (default: this script's repo)",
    )
    ap.add_argument(
        "--cube-harness",
        type=Path,
        default=None,
        help="path to the cube-harness repo (default: sibling ../cube-harness)",
    )
    ap.add_argument("--ref", default="main", help="branch each tag points at (default: main — the release ref)")
    ap.add_argument("--only", action="append", default=[], metavar="DIST", help="restrict to these dist names")
    ap.add_argument("--execute", action="store_true", help="actually push tags + wait for PyPI (default: dry run)")
    ap.add_argument("--pypi-timeout", type=int, default=900, help="seconds to wait per package for PyPI (default 900)")
    args = ap.parse_args()

    cube_standard = args.cube_standard.resolve()
    cube_harness = (args.cube_harness or cube_standard.parent / "cube-harness").resolve()
    repo_paths = {"cube-standard": cube_standard, "cube-harness": cube_harness}

    print(f"cube-standard : {cube_standard}")
    print(f"cube-harness  : {cube_harness}")
    print(f"release ref   : {args.ref}")
    print(f"mode          : {'EXECUTE' if args.execute else 'dry-run (no side effects)'}\n")

    # Preflight both repos.
    problems = preflight(cube_standard, args.ref) + preflight(cube_harness, args.ref)
    if problems:
        for p in problems:
            print(f"[BLOCKED] preflight: {p}")
        raise SystemExit(1)

    manifest = build_manifest(cube_standard, cube_harness)
    if args.only:
        manifest = [p for p in manifest if p.dist in set(args.only)]
        if not manifest:
            raise SystemExit(f"--only matched no packages. Known: {sorted(p.dist for p in manifest)}")

    plans = [plan_package(p, repo_paths[p.repo]) for p in manifest]

    # Print the ordered plan.
    blocked = [pl for pl in plans if pl.state == "BLOCKED"]
    to_release = [pl for pl in plans if pl.state == "RELEASE"]
    print("PLAN (topological order):\n")
    for tier in sorted({pl.pkg.tier for pl in plans}):
        print(f"  ── tier {tier} ──")
        for pl in [p for p in plans if p.pkg.tier == tier]:
            mark = {"RELEASE": "▶ RELEASE ", "UP-TO-DATE": "· uptodate", "BLOCKED": "✗ BLOCKED "}[pl.state]
            detail = f"{pl.pkg.tag_prefix}/v{pl.current}" if pl.state == "RELEASE" else pl.reason
            print(f"    {mark} {pl.pkg.dist:<28} {detail}")
    print()

    if blocked:
        print("Resolve these before releasing:")
        for pl in blocked:
            print(f"[BLOCKED] {pl.pkg.dist}: {pl.reason}")
        raise SystemExit(1)

    if not to_release:
        print("[done] nothing to release — every package is up to date")
        return

    if not args.execute:
        print(f"[dry-run] {len(to_release)} package(s) would be released, tier by tier.")
        print("[dry-run] re-run with --execute to push tags and wait for PyPI between tiers.")
        return

    execute(to_release, repo_paths, args.pypi_timeout)

    print("\n[done] all tiers published")
    print(
        "[next] post-release dev bump (recommendation A): for every package just released, "
        "bump its `dev` pyproject version to the next pre-release so dev never impersonates "
        "the published version (cube-standard: rcN -> rc(N+1)). Open a small "
        "`chore: bump dev versions` PR. Left manual on purpose — touching dev needs a "
        "signed-off PR, not a tag push."
    )


if __name__ == "__main__":
    main()
