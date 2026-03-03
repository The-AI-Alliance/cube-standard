"""Integration tests for ToolkitContainerBackend against a real EAI Toolkit cluster.

Requires: ``eai`` CLI installed and authenticated.
"""

import shutil
import time
import urllib.request

import pytest

if shutil.which("eai") is None:
    pytest.skip("eai CLI not installed", allow_module_level=True)

from test_harness import log, make_container_common_tests, make_container_health_check_tests

from cube.backends.toolkit import ToolkitContainerBackend
from cube.container import ContainerConfig

backend = ToolkitContainerBackend(timeout_seconds=600)
spec = ContainerConfig(image="python:3.12-slim")

_all_tests = make_container_common_tests(backend, spec) + make_container_health_check_tests(backend, spec)


@pytest.mark.parametrize("name,fn", _all_tests, ids=[t[0] for t in _all_tests])
def test_container(name, fn):
    fn()


def test_port_forwarding():
    port_spec = ContainerConfig(image="python:3.12-slim", ports=[8080])
    container = backend.launch(port_spec)
    try:
        container.exec("python -m http.server 8080 &")
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
