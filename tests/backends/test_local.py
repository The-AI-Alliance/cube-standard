"""Integration tests for LocalContainerBackend against a real Docker daemon."""

import time
import urllib.request

import pytest

pytest.importorskip("docker")  # skip whole module if docker not installed
pytest.importorskip("tenacity")  # local backend also requires tenacity

pytestmark = pytest.mark.integration

import docker  # noqa: E402 — must come after pytest.importorskip

from cube.backends.local import LocalContainerBackend  # noqa: E402
from cube.container import ContainerConfig, ContainerError  # noqa: E402
from tests.backends.test_harness import (  # noqa: E402
    log,
    make_container_common_tests,
    make_container_health_check_tests,
)

try:
    docker.DockerClient().ping()
except Exception as _e:
    pytest.skip(f"Docker daemon not reachable: {_e}", allow_module_level=True)


backend = LocalContainerBackend(timeout_seconds=60)
spec = ContainerConfig(image="alpine:latest")

_all_tests = make_container_common_tests(backend, spec) + make_container_health_check_tests(backend, spec)


@pytest.mark.parametrize("name,fn", _all_tests, ids=[t[0] for t in _all_tests])
def test_container(name, fn):
    fn()


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


def test_forward_unexposed_port():
    container = backend.launch(spec)
    try:
        with pytest.raises(ContainerError):
            container.forward_port(9999)
        log("correctly raised ContainerError for unexposed port")
    finally:
        container.stop()
