"""Integration tests for ToolkitContainerBackend against a real EAI Toolkit cluster.

Requires: ``eai`` CLI installed and authenticated.
"""

import os
import shutil
import subprocess
import time
import urllib.request

import pytest

pytest.importorskip("tenacity")  # skip whole module if tenacity not installed

pytestmark = pytest.mark.integration

from cube.backends.toolkit import ToolkitContainerBackend  # noqa: E402
from cube.container import ContainerConfig  # noqa: E402
from tests.backends.test_harness import (  # noqa: E402
    log,
    make_container_common_tests,
    make_container_health_check_tests,
)

if shutil.which("eai") is None:
    pytest.skip("eai CLI not installed", allow_module_level=True)

_probe = ["eai", "user", "get"]
if _EAI_PROFILE := os.environ.get("EAI_PROFILE"):
    _probe += ["--profile", _EAI_PROFILE]

try:
    _result = subprocess.run(_probe, capture_output=True, text=True, timeout=10)
    if _result.returncode != 0:
        raise RuntimeError(_result.stderr.strip())
except Exception as _e:
    pytest.skip(f"eai endpoint not reachable: {_e}", allow_module_level=True)

backend = ToolkitContainerBackend(timeout_seconds=600, profile=_EAI_PROFILE)
spec = ContainerConfig(image="python:3.12-slim")

_all_tests = make_container_common_tests(backend, spec) + make_container_health_check_tests(backend, spec)


@pytest.mark.parametrize("name,fn", _all_tests, ids=[t[0] for t in _all_tests])
def test_container(name, fn):
    fn()


def test_port_forwarding():
    port_spec = ContainerConfig(image="python:3.12-slim", ports=[8080])
    container = backend.launch(port_spec)
    try:
        container.exec("python -m http.server 8080 </dev/null >/dev/null 2>&1 & disown")
        time.sleep(3)

        host_port = container.forward_port(8080)
        assert isinstance(host_port, int) and host_port > 0
        log(f"host_port={host_port}")

        url = container.get_url(8080)
        assert url == f"http://localhost:{host_port}"
        log(f"url={url}")

        resp = urllib.request.urlopen(url, timeout=10)
        assert resp.status == 200
        log(f"HTTP GET {url} -> {resp.status}")
    finally:
        container.stop()
