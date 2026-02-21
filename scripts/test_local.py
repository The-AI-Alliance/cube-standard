#!/usr/bin/env python3
"""Integration tests for LocalContainerBackend against a real Docker daemon.

Usage:  uv run scripts/test_local.py
"""

import time
import urllib.request

from test_harness import log, make_health_check_tests, make_tests, run_all

from cube.backends.local import LocalContainerBackend
from cube.container import ContainerConfig, ContainerError, ContainerLaunchError

BACKEND_KWARGS = {"timeout_seconds": 60}
backend = LocalContainerBackend(**BACKEND_KWARGS)
spec = ContainerConfig(image="alpine:latest")

tests = make_tests(backend, spec)
tests += make_health_check_tests(LocalContainerBackend, spec, BACKEND_KWARGS)

def test_port_forwarding():
    port_spec = ContainerConfig(image="python:3.12-slim", ports=[8080])
    container = backend.launch(port_spec)
    try:
        container.exec("python -m http.server 8080 &")
        time.sleep(2)

        host_port = container.forward_port(8080)
        assert isinstance(host_port, int) and host_port > 0
        log(f"host_port={host_port}")

        url = container.get_url(8080)
        assert url == f"http://localhost:{host_port}"
        log(f"url={url}")

        resp = urllib.request.urlopen(url, timeout=5)
        assert resp.status == 200
        log(f"HTTP GET {url} -> {resp.status}")
    finally:
        container.stop()


tests.append(("port forwarding", test_port_forwarding))


def test_forward_unexposed_port():
    container = backend.launch(spec)
    try:
        try:
            container.forward_port(9999)
            assert False, "should have raised ContainerError"
        except ContainerError:
            log("correctly raised ContainerError for unexposed port")
    finally:
        container.stop()


tests.append(("forward unexposed port", test_forward_unexposed_port))


def test_disk_override_not_supported():
    bad_spec = ContainerConfig(image="alpine:latest", disk_gb=20.0)
    try:
        backend.launch(bad_spec)
        assert False, "should have raised ContainerLaunchError"
    except ContainerLaunchError:
        log("correctly rejected unsupported disk_gb override")


tests.append(("reject disk_gb override", test_disk_override_not_supported))

if __name__ == "__main__":
    run_all("LocalContainerBackend integration tests", tests)
