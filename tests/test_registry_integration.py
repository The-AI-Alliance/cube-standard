"""Integration tests for `cube registry add`.

Tests the YAML generation flow (no GitHub required) and delegates the full
E2E submit/CI test to test_registry_integration_sandbox.py.

Run with:
    pytest -m integration tests/test_registry_integration.py -s

Requirements:
  - cube-harness checked out as a sibling of cube-standard
    (i.e. ../cube-harness/cubes/arithmetic-cube must exist)
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

CUBE_STANDARD_ROOT = Path(__file__).parent.parent
# arithmetic-cube: used for YAML generation tests (no GitHub)
ARITHMETIC_CUBE_PATH = CUBE_STANDARD_ROOT.parent / "cube-harness" / "cubes" / "arithmetic-cube"
COUNTER_CUBE_PATH = CUBE_STANDARD_ROOT / "examples" / "counter-cube"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _cube_harness_available() -> bool:
    return ARITHMETIC_CUBE_PATH.exists()


# ---------------------------------------------------------------------------
# Fixtures & markers
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.integration


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: end-to-end tests that hit GitHub")


@pytest.fixture(autouse=True)
def cleanup_yaml():
    """Remove generated YAML files after each test."""
    yield
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
        """Running without --submit regenerates (overwrites) the existing file."""
        _run_cube("registry", "add", str(ARITHMETIC_CUBE_PATH))
        yaml_path = ARITHMETIC_CUBE_PATH / "cube-registry-entry.yaml"
        yaml_path.write_text(yaml_path.read_text() + "\n# SENTINEL\n")
        _run_cube("registry", "add", str(ARITHMETIC_CUBE_PATH))
        assert "SENTINEL" not in yaml_path.read_text(), "Generate should overwrite existing file"
