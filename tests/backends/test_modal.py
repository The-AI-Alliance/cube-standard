"""Integration tests for ModalContainerBackend against the real Modal API.

Requires: modal token set --token-id <id> --token-secret <secret>
"""

import time
import urllib.request
from urllib.parse import urlparse

import pytest

from cube.backends.modal import ModalContainerBackend
from cube.container import ContainerConfig
from tests.backends.test_harness import log, make_container_common_tests, make_container_health_check_tests

modal = pytest.importorskip("modal")

try:
    import modal.config as _mc

    cfg = _mc.config
    if not (cfg.get("token_id") and cfg.get("token_secret")):
        raise ValueError("no token")
except Exception as _e:
    pytest.skip(f"Modal token not configured: {_e}", allow_module_level=True)

BACKEND_KWARGS = {"timeout_seconds": 600, "app_name": "cube-test"}
backend = ModalContainerBackend(**BACKEND_KWARGS)
spec = ContainerConfig(image="python:3.12-slim")

_all_tests = make_container_common_tests(backend, spec) + make_container_health_check_tests(backend, spec)


@pytest.mark.parametrize("name,fn", _all_tests, ids=[t[0] for t in _all_tests])
def test_container(name, fn):
    fn()


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
