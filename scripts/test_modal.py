#!/usr/bin/env python3
"""Integration tests for ModalContainerBackend against the real Modal API.

Requires: modal token set --token-id <id> --token-secret <secret>

Usage:  uv run scripts/test_modal.py
"""

import time
import urllib.request
from urllib.parse import urlparse

from test_harness import log, make_health_check_tests, make_tests, run_all

from cube.backends.modal import ModalContainerBackend
from cube.container import ContainerConfig, ContainerLaunchError

BACKEND_KWARGS = {"timeout_seconds": 600, "app_name": "cube-test"}
backend = ModalContainerBackend(**BACKEND_KWARGS)
spec = ContainerConfig(image="python:3.12-slim")

tests = make_tests(backend, spec)
tests += make_health_check_tests(ModalContainerBackend, spec, BACKEND_KWARGS)

def test_tunnel_url():
    port_spec = ContainerConfig(image="python:3.12-slim", ports=[8080])
    container = ModalContainerBackend(**BACKEND_KWARGS).launch(port_spec)
    try:
        container.exec("python -m http.server 8080 &")
        time.sleep(3)

        url = container.get_url(8080)
        assert url.startswith("http"), f"unexpected url: {url}"
        log(f"tunnel url={url}")
        parsed = urlparse(url)
        expected_port = parsed.port or (443 if parsed.scheme == "https" else 80)
        forwarded = container.forward_port(8080)
        assert forwarded == expected_port
        log(f"forwarded_port={forwarded}")

        try:
            resp = urllib.request.urlopen(url, timeout=10)
            log(f"HTTP GET -> {resp.status}")
        except Exception as exc:
            log(f"HTTP GET attempt: {exc}")
    finally:
        container.stop()


tests.append(("tunnel URL", test_tunnel_url))


def test_disk_override_not_supported():
    bad_spec = ContainerConfig(image="python:3.12-slim", disk_gb=20.0)
    try:
        backend.launch(bad_spec)
        assert False, "should have raised ContainerLaunchError"
    except ContainerLaunchError:
        log("correctly rejected unsupported disk_gb override")


tests.append(("reject disk_gb override", test_disk_override_not_supported))

if __name__ == "__main__":
    run_all("ModalContainerBackend integration tests", tests)
