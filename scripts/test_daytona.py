#!/usr/bin/env python3
"""Integration tests for DaytonaContainerBackend against the real Daytona API.

Usage:
    uv run scripts/test_daytona.py
    DAYTONA_API_KEY=dtn_... uv run scripts/test_daytona.py
"""

import os
import sys

from cube.backends.daytona import DaytonaContainerBackend  # noqa: E402
from cube.container import ContainerSpec  # noqa: E402
from dotenv import load_dotenv

from test_harness import make_health_check_tests, make_tests, run_all  # noqa: E402

# Load .env from AgentLab2 if DAYTONA_API_KEY not already set
if not os.environ.get("DAYTONA_API_KEY"):
    for env_path in [
        os.path.expanduser("~/Downloads/projects/servicenow/AgentLab2/.env"),
        os.path.join(os.path.dirname(__file__), "..", ".env"),
    ]:
        if os.path.exists(env_path):
            load_dotenv(env_path)
            print(f"Loaded env from {env_path}")
            break

API_KEY = os.environ.get("DAYTONA_API_KEY")
if not API_KEY:
    print("ERROR: DAYTONA_API_KEY not set. Set it in environment or in AgentLab2/.env")
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
spec = ContainerSpec(image="python:3.12-slim")

tests = make_tests(backend, spec)
tests += make_health_check_tests(DaytonaContainerBackend, spec, BACKEND_KWARGS)

if __name__ == "__main__":
    run_all("DaytonaContainerBackend integration tests", tests)
