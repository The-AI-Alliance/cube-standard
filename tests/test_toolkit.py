#!/usr/bin/env python3
"""Integration tests for ToolkitContainerBackend against a real EAI Toolkit cluster.

Requires: ``eai`` CLI installed and authenticated.

Usage:  uv run python tests/test_toolkit.py
"""

import time
import urllib.request

from test_harness import log, make_container_common_tests, make_container_health_check_tests, run_all

from cube.backends.toolkit import ToolkitContainerBackend
from cube.container import ContainerConfig

BACKEND_KWARGS = {"timeout_seconds": 600}
backend = ToolkitContainerBackend(**BACKEND_KWARGS)
spec = ContainerConfig(image="python:3.12-slim")

tests = make_container_common_tests(backend, spec)
tests += make_container_health_check_tests(backend, spec)


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


tests.append(("port forwarding", test_port_forwarding))

if __name__ == "__main__":
    run_all("ToolkitContainerBackend integration tests", tests)
