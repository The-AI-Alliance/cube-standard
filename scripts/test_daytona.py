#!/usr/bin/env python3
"""Integration tests for DaytonaContainerBackend against the real Daytona API.

Requires DAYTONA_API_KEY in the environment (or in a .env file).

Usage:  PYTHONPATH=scripts uv run python scripts/test_daytona.py
"""

import os
import sys

from dotenv import load_dotenv
from test_harness import log, make_health_check_tests, make_tests, run_all

from cube.backends.daytona import DaytonaContainerBackend
from cube.container import ContainerConfig, ContainerError

load_dotenv()

API_KEY = os.environ.get("DAYTONA_API_KEY")
if not API_KEY:
    print("ERROR: DAYTONA_API_KEY not set")
    sys.exit(1)
print(f"Using API key: {API_KEY[:10]}...{API_KEY[-4:]}")

BACKEND_KWARGS = {
    "api_key": API_KEY,
    "timeout_seconds": 300,
    "ephemeral": True,
    "auto_stop_minutes": 5,
    "auto_delete_minutes": 3,
}
backend = DaytonaContainerBackend(**BACKEND_KWARGS)
spec = ContainerConfig(image="python:3.12-slim")

tests = make_tests(backend, spec)
tests += make_health_check_tests(DaytonaContainerBackend, spec, BACKEND_KWARGS)


def test_declared_ports_enforced():
    port_spec = ContainerConfig(image="python:3.12-slim", ports=[8080])
    container = backend.launch(port_spec)
    try:
        try:
            container.get_url(9090)
            assert False, "should have raised ContainerError"
        except ContainerError:
            log("correctly rejected undeclared port access")
    finally:
        container.stop()


tests.append(("declared ports enforced", test_declared_ports_enforced))

if __name__ == "__main__":
    run_all("DaytonaContainerBackend integration tests", tests)
