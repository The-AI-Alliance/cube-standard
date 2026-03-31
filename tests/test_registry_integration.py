"""Integration tests for `cube registry add`.

These tests exercise the full end-to-end flow:
  1. `cube registry add <path>` generates a YAML entry with TODO markers.
  2. The test fills in all TODO fields.
  3. `cube registry add --submit <path>` forks cube-registry, creates a branch,
     uploads the YAML, and opens a PR.
  4. The test polls GitHub until the quick-compliance CI check completes.
  5. Teardown closes the PR and deletes the branch to keep the registry clean.

Run with:
    pytest -m integration tests/test_registry_integration.py -s

Requirements:
  - `gh` CLI installed and authenticated (`gh auth login`)
  - cube-harness checked out as a sibling of cube-standard
    (i.e. ../cube-harness/cubes/arithmetic-cube must exist)
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

CUBE_STANDARD_ROOT = Path(__file__).parent.parent
# counter-cube: ships with cube-standard, not yet in the registry → open submission (passes ownership-check)
COUNTER_CUBE_PATH = CUBE_STANDARD_ROOT / "examples" / "counter-cube"
# arithmetic-cube: used for YAML generation tests only (no GitHub)
ARITHMETIC_CUBE_PATH = CUBE_STANDARD_ROOT.parent / "cube-harness" / "cubes" / "arithmetic-cube"
REGISTRY = "The-AI-Alliance/cube-registry"
ENTRY_ID = "counter-cube"
BRANCH = f"add/{ENTRY_ID}"
CI_TIMEOUT_S = 300  # 5 minutes max for CI to complete

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gh(*args: str) -> str:
    """Run a gh command, return stdout. Raises on non-zero exit."""
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


def _run_cube(*args: str) -> subprocess.CompletedProcess:
    """Run `cube` via uv from cube-standard root."""
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


def _cube_harness_available() -> bool:
    return ARITHMETIC_CUBE_PATH.exists()


# ---------------------------------------------------------------------------
# Fixtures & markers
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.integration


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: end-to-end tests that hit GitHub")


@pytest.fixture(autouse=True)
def cleanup_pr_and_branch():
    """Ensure PR and fork branch are cleaned up even if the test fails."""
    yield
    if not _gh_available():
        return

    gh_user = _gh("api", "/user", "--jq", ".login")

    # Close any open PRs from this branch
    try:
        prs = _gh_api("GET", f"/repos/{REGISTRY}/pulls?state=open&head={gh_user}:{BRANCH}")
        for pr in prs if isinstance(prs, list) else []:
            _gh_api("PATCH", f"/repos/{REGISTRY}/pulls/{pr['number']}", {"state": "closed"})
            print(f"\n[cleanup] Closed PR #{pr['number']}")
    except Exception as e:
        print(f"\n[cleanup] Could not close PRs: {e}")

    # Delete fork branch
    try:
        _gh_api("DELETE", f"/repos/{gh_user}/cube-registry/git/refs/heads/{BRANCH}")
        print(f"\n[cleanup] Deleted branch {BRANCH}")
    except Exception:
        pass  # branch may not exist

    # Remove generated YAML files
    for yaml_path in [
        COUNTER_CUBE_PATH / "cube-registry-entry.yaml",
        ARITHMETIC_CUBE_PATH / "cube-registry-entry.yaml",
    ]:
        if yaml_path.exists():
            yaml_path.unlink()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _cube_harness_available(), reason="cube-harness not found at ../cube-harness")
class TestRegistryAdd:
    def test_generate_yaml_creates_file_with_correct_fields(self):
        """cube registry add generates a YAML with auto-detected fields and TODO markers."""
        result = _run_cube("registry", "add", str(ARITHMETIC_CUBE_PATH))

        assert result.returncode == 0, f"Exit {result.returncode}\n{result.stderr}"

        yaml_path = ARITHMETIC_CUBE_PATH / "cube-registry-entry.yaml"
        assert yaml_path.exists(), "cube-registry-entry.yaml was not created"

        content = yaml_path.read_text()
        assert "id: arithmetic-cube" in content
        assert 'version: "0.1.0"' in content
        assert "package: arithmetic-cube" in content
        assert "dev_install_url:" in content
        assert "cube-harness" in content  # detected from git remote
        assert "<TODO:" in content  # some fields need manual filling

    def test_generate_yaml_todo_count(self):
        """Exactly the expected TODO fields are left for the developer to fill."""
        _run_cube("registry", "add", str(ARITHMETIC_CUBE_PATH))
        content = (ARITHMETIC_CUBE_PATH / "cube-registry-entry.yaml").read_text()
        todos = [ln for ln in content.splitlines() if "<TODO:" in ln and not ln.strip().startswith("#")]
        # Expect: github handle, full name, wrapper_license, tags
        assert len(todos) == 4, f"Expected 4 TODOs, got {len(todos)}:\n" + "\n".join(todos)

    def test_submit_blocked_when_todos_remain(self):
        """--submit exits non-zero if TODOs are still present."""
        _run_cube("registry", "add", str(ARITHMETIC_CUBE_PATH))
        result = _run_cube("registry", "add", "--submit", str(ARITHMETIC_CUBE_PATH))
        assert result.returncode != 0
        assert "TODO" in result.stderr or "TODO" in result.stdout

    def test_rerun_preserves_edited_file(self):
        """Running --submit uses the existing file without regenerating it."""
        _run_cube("registry", "add", str(ARITHMETIC_CUBE_PATH))
        yaml_path = ARITHMETIC_CUBE_PATH / "cube-registry-entry.yaml"
        # Inject a sentinel comment
        yaml_path.write_text(yaml_path.read_text() + "\n# SENTINEL\n")
        # Running again without --submit should regenerate (overwrite)
        _run_cube("registry", "add", str(ARITHMETIC_CUBE_PATH))
        assert "SENTINEL" not in yaml_path.read_text(), "Generate should overwrite existing file"

    @pytest.mark.skipif(not _gh_available(), reason="gh CLI not available or not authenticated")
    def test_full_submit_and_ci(self):
        """Generate YAML for counter-cube, fill TODOs, submit PR, wait for CI, verify checks pass."""
        # counter-cube is not yet in the registry → open submission (ownership-check passes)
        yaml_path = COUNTER_CUBE_PATH / "cube-registry-entry.yaml"

        # 1. Generate
        result = _run_cube("registry", "add", str(COUNTER_CUBE_PATH))
        assert result.returncode == 0, f"Generate failed:\n{result.stderr}"
        assert yaml_path.exists()

        # 2. Fill TODOs
        gh_user = _gh("api", "/user", "--jq", ".login")
        content = yaml_path.read_text()
        content = re.sub(r"github: <TODO: github-handle>", f"github: {gh_user}", content)
        content = re.sub(r"name: <TODO: Full Name>", f"name: {gh_user}", content)
        content = re.sub(r"wrapper_license: <TODO:.*?>", "wrapper_license: MIT", content)
        content = re.sub(r"  - <TODO: math\|web\|gui\|desktop>", "  - math", content)
        yaml_path.write_text(content)

        remaining_todos = [ln for ln in content.splitlines() if "<TODO:" in ln and not ln.strip().startswith("#")]
        assert not remaining_todos, f"TODOs still present:\n" + "\n".join(remaining_todos)

        # 3. Submit
        result = _run_cube("registry", "add", "--submit", str(COUNTER_CUBE_PATH))
        assert result.returncode == 0, f"Submit failed:\n{result.stdout}\n{result.stderr}"

        pr_match = re.search(r"https://github\.com/[^\s]+/pull/(\d+)", result.stdout + result.stderr)
        assert pr_match, f"No PR URL found in output:\n{result.stdout}\n{result.stderr}"
        pr_number = int(pr_match.group(1))
        pr_url = pr_match.group(0)
        print(f"\nPR created: {pr_url}")

        # 4. Poll for CI checks (ownership-check and quick-compliance)
        pr_data = _gh_api("GET", f"/repos/{REGISTRY}/pulls/{pr_number}")
        head_sha = pr_data["head"]["sha"]

        print(f"Waiting for CI on commit {head_sha[:8]} (up to {CI_TIMEOUT_S}s)...")
        deadline = time.time() + CI_TIMEOUT_S
        check_runs = []
        while time.time() < deadline:
            time.sleep(15)
            resp = _gh_api("GET", f"/repos/{REGISTRY}/commits/{head_sha}/check-runs")
            check_runs = resp.get("check_runs", []) if isinstance(resp, dict) else []
            statuses = {cr["name"]: cr["status"] for cr in check_runs}
            conclusions = {cr["name"]: cr["conclusion"] for cr in check_runs if cr["conclusion"]}
            print(f"  checks: {statuses}")
            all_done = all(cr["status"] == "completed" for cr in check_runs) and len(check_runs) >= 2
            if all_done:
                break
        else:
            pytest.fail(f"CI did not complete within {CI_TIMEOUT_S}s. Last status: {statuses}")

        # 5. Verify required checks passed (exclude infra jobs like "Enable auto-merge")
        required_checks = {"ownership-check", "DCO", "quick-compliance"}
        failed = [
            cr["name"]
            for cr in check_runs
            if cr["name"] in required_checks
            and cr.get("conclusion") not in ("success", "skipped")
        ]
        assert not failed, f"CI checks failed: {failed}\nCheck: {pr_url}/checks"
        print(f"All CI checks passed.")
