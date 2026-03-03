"""Integration tests for DaytonaContainerBackend against the real Daytona API.

Requires DAYTONA_API_KEY in the environment (or in a .env file).
"""

import os

import pytest
from dotenv import load_dotenv

from cube.backends.daytona import DaytonaContainerBackend
from cube.container import ContainerConfig, ContainerError
from tests.backends.test_harness import log, make_container_common_tests, make_container_health_check_tests

daytona = pytest.importorskip("daytona")

load_dotenv()

if not os.environ.get("DAYTONA_API_KEY"):
    pytest.skip("DAYTONA_API_KEY not set", allow_module_level=True)

backend = DaytonaContainerBackend(
    api_key=os.environ["DAYTONA_API_KEY"],
    timeout_seconds=300,
    ephemeral=True,
    auto_stop_minutes=5,
    auto_delete_minutes=3,
)
spec = ContainerConfig(image="python:3.12-slim")

_all_tests = make_container_common_tests(backend, spec) + make_container_health_check_tests(backend, spec)


@pytest.mark.parametrize("name,fn", _all_tests, ids=[t[0] for t in _all_tests])
def test_container(name, fn):
    fn()


def test_declared_ports_enforced():
    port_spec = ContainerConfig(image="python:3.12-slim", ports=[8080])
    container = backend.launch(port_spec)
    try:
        with pytest.raises(ContainerError):
            container.get_url(9090)
        log("correctly rejected undeclared port access")
    finally:
        container.stop()
