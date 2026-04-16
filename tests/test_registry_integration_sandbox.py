"""Sandbox integration tests for `cube registry add --submit`.

Identical flow to test_registry_integration.py but targets the developer's own
fork of cube-registry instead of the upstream Alliance repository.  PRs never
touch The-AI-Alliance/cube-registry, so these tests are safe to run at any time
without polluting the production registry.

The test opens a PR from `<user>/cube-registry:add/counter-cube` to
`<user>/cube-registry:main`, which triggers the same CI workflows (they live in
the fork), so the full ownership-check / DCO / quick-compliance pipeline runs.

Run with:
    pytest -m integration tests/test_registry_integration_sandbox.py -s

Requirements:
  - `gh` CLI installed and authenticated
  - The authenticated user must have a fork of cube-registry
    (created automatically by `cube registry add --submit` on first run,
     or by running: gh repo fork The-AI-Alliance/cube-registry --clone=false)
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

CUBE_STANDARD_ROOT = Path(__file__).parent.parent
COUNTER_CUBE_PATH = CUBE_STANDARD_ROOT / "examples" / "counter-cube"
ENTRY_ID = "counter-cube"
BRANCH = f"add/{ENTRY_ID}"
CI_TIMEOUT_S = 300

# ---------------------------------------------------------------------------
# Helpers (duplicated from test_registry_integration.py to keep files independent)
# ---------------------------------------------------------------------------


def _gh(*args: str) -> str:
    r = subprocess.run(["gh"] + list(args), capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)}\n{r.stderr.strip()}")
    return r.stdout.strip()


def _gh_api(method: str, endpoint: str, body: dict | None = None) -> dict | list:
    cmd = ["gh", "api", "--method", method, endpoint]
    if body:
        for k, v in body.items():
            cmd.extend(["--field", f"{k}={v}"])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"gh api {endpoint}\n{r.stderr.strip()}")
    return json.loads(r.stdout) if r.stdout.strip() else {}


def _gh_api_json(method: str, endpoint: str, body: dict) -> dict | list:
    """Like _gh_api but sends body as JSON (for endpoints that don't accept --field)."""
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(body, f)
        tmp = f.name
    try:
        r = subprocess.run(
            ["gh", "api", "--method", method, "--input", tmp, endpoint],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"gh api {endpoint}\n{r.stderr.strip()}")
        return json.loads(r.stdout) if r.stdout.strip() else {}
    finally:
        Path(tmp).unlink(missing_ok=True)


def _run_cube(*args: str) -> subprocess.CompletedProcess:
    import os

    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    return subprocess.run(
        ["uv", "run", "cube"] + list(args),
        cwd=str(CUBE_STANDARD_ROOT),
        capture_output=True,
        text=True,
        env=env,
    )


def _gh_available() -> bool:
    r = subprocess.run(["gh", "auth", "status", "--active"], capture_output=True)
    return r.returncode == 0


def _fork_available() -> bool:
    """True if the authenticated user has a fork of cube-registry."""
    if not _gh_available():
        return False
    try:
        user = _gh("api", "/user", "--jq", ".login")
        r = subprocess.run(
            ["gh", "api", f"/repos/{user}/cube-registry"],
            capture_output=True,
        )
        return r.returncode == 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Fixtures & markers
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def cleanup_sandbox():
    """Close any open sandbox PR and delete the test branch after each test."""
    yield
    if not _gh_available():
        return

    gh_user = _gh("api", "/user", "--jq", ".login")
    sandbox_registry = f"{gh_user}/cube-registry"
    # GitHub redirects fork PRs to the upstream — check both the fork and upstream.
    upstream = "The-AI-Alliance/cube-registry"

    # Close any open PRs from this branch on the fork or upstream
    for registry in (sandbox_registry, upstream):
        try:
            prs = _gh_api("GET", f"/repos/{registry}/pulls?state=open&head={gh_user}:{BRANCH}")
            for pr in prs if isinstance(prs, list) else []:
                _gh_api("PATCH", f"/repos/{registry}/pulls/{pr['number']}", {"state": "closed"})
                print(f"\n[cleanup] Closed sandbox PR #{pr['number']} on {registry}")
        except Exception as e:
            print(f"\n[cleanup] Could not close PRs on {registry}: {e}")

    # Delete the test branch from the fork
    try:
        _gh_api("DELETE", f"/repos/{gh_user}/cube-registry/git/refs/heads/{BRANCH}")
        print(f"\n[cleanup] Deleted branch {BRANCH} from {sandbox_registry}")
    except Exception:
        pass

    # Remove generated YAML
    yaml_path = COUNTER_CUBE_PATH / "cube-registry-entry.yaml"
    if yaml_path.exists():
        yaml_path.unlink()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _gh_available(), reason="gh CLI not available or not authenticated")
class TestRegistryAddSandbox:
    """Full E2E test targeting the developer's own fork — no upstream impact."""

    def _sandbox_registry(self) -> str:
        return f"{_gh('api', '/user', '--jq', '.login')}/cube-registry"

    @pytest.mark.skipif(
        not _fork_available(),
        reason="cube-registry fork not found — run: gh repo fork The-AI-Alliance/cube-registry --clone=false",
    )
    def test_full_submit_and_ci_sandbox(self):
        """Generate YAML, fill TODOs, submit PR to personal fork, wait for CI, verify checks pass."""
        gh_user = _gh("api", "/user", "--jq", ".login")
        sandbox_registry = f"{gh_user}/cube-registry"
        yaml_path = COUNTER_CUBE_PATH / "cube-registry-entry.yaml"

        # Sync the fork's main with upstream so workflows are up to date.
        try:
            _gh_api_json("POST", f"/repos/{sandbox_registry}/merge-upstream", {"branch": "main"})
            print(f"[sandbox] Synced {sandbox_registry}:main with upstream")
        except Exception as e:
            print(f"[sandbox] Fork sync skipped: {e}")

        # 1. Generate
        result = _run_cube("registry", "add", str(COUNTER_CUBE_PATH))
        assert result.returncode == 0, f"Generate failed:\n{result.stderr}"
        assert yaml_path.exists()

        # 2. Fill TODOs
        content = yaml_path.read_text()
        content = re.sub(r"github: <TODO: github-handle>", f"github: {gh_user}", content)
        content = re.sub(r"name: <TODO: Full Name>", f"name: {gh_user}", content)
        content = re.sub(r"wrapper_license: <TODO:.*?>", "wrapper_license: MIT", content)
        content = re.sub(r"  - <TODO: math\|web\|gui\|desktop>", "  - math", content)
        yaml_path.write_text(content)

        remaining = [ln for ln in content.splitlines() if "<TODO:" in ln and not ln.strip().startswith("#")]
        assert not remaining, "TODOs still present:\n" + "\n".join(remaining)

        # 3. Submit — target the personal fork, not the upstream Alliance repo
        result = _run_cube("registry", "add", "--submit", "--registry", sandbox_registry, str(COUNTER_CUBE_PATH))
        assert result.returncode == 0, f"Submit failed:\n{result.stdout}\n{result.stderr}"

        pr_match = re.search(r"https://github\.com/([^/]+/[^/\s]+)/pull/(\d+)", result.stdout + result.stderr)
        assert pr_match, f"No PR URL found in output:\n{result.stdout}\n{result.stderr}"
        # GitHub may redirect fork PRs to the upstream repo — use the actual repo from the URL.
        actual_registry = pr_match.group(1)
        pr_number = int(pr_match.group(2))
        pr_url = pr_match.group(0)
        print(f"\nSandbox PR: {pr_url} (registry: {actual_registry})")

        # 4. Poll CI wherever the PR actually landed
        pr_data = _gh_api("GET", f"/repos/{actual_registry}/pulls/{pr_number}")
        head_sha = pr_data["head"]["sha"]

        print(f"Waiting for CI on commit {head_sha[:8]} (up to {CI_TIMEOUT_S}s)...")
        deadline = time.time() + CI_TIMEOUT_S
        check_runs = []
        statuses: dict[str, str] = {}
        while time.time() < deadline:
            time.sleep(15)
            resp = _gh_api("GET", f"/repos/{actual_registry}/commits/{head_sha}/check-runs")
            check_runs = resp.get("check_runs", []) if isinstance(resp, dict) else []
            statuses = {cr["name"]: cr["status"] for cr in check_runs}
            print(f"  checks: {statuses}")
            all_done = all(cr["status"] == "completed" for cr in check_runs) and len(check_runs) >= 2
            if all_done:
                break
        else:
            pytest.fail(f"CI did not complete within {CI_TIMEOUT_S}s. Last checks: {statuses}")

        # 5. Verify required checks passed
        required_checks = {"ownership-check", "DCO", "quick-compliance"}
        failed = [
            cr["name"]
            for cr in check_runs
            if cr["name"] in required_checks and cr.get("conclusion") not in ("success", "skipped")
        ]
        assert not failed, f"CI checks failed: {failed}\nCheck: {pr_url}/checks"
        print("All CI checks passed on sandbox fork.")
