"""
End-to-end smoke test: start server.py, run client.py, check exit code.

Starts the benchmark server as a subprocess on a fixed port, waits for it
to bind, then runs the client and asserts clean exit with the expected reward.

Run from examples/counter-cube-remote/:

    uv run pytest tests/ -v
"""

import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).parent.parent  # examples/counter-cube-remote/
_SERVER_PORT = 8765
_SERVER_STARTUP_SECONDS = 3.0


@pytest.fixture
def benchmark_server():
    """Start server.py on _SERVER_PORT, yield, then terminate."""
    proc = subprocess.Popen(
        [sys.executable, str(HERE / "server.py"), "--port", str(_SERVER_PORT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(_SERVER_STARTUP_SECONDS)
    yield proc
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.mark.parametrize("script", ["client_sdk.py", "client_raw.py"])
def test_client_runs_full_episode(benchmark_server, script):
    assert benchmark_server.poll() is None, "benchmark server exited before client started"
    result = subprocess.run(
        [sys.executable, str(HERE / script), f"http://127.0.0.1:{_SERVER_PORT}"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"{script} exited with code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert '"reward": 1.0' in result.stdout
