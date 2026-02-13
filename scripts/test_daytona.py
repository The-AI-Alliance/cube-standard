#!/usr/bin/env python3
"""Integration tests for DaytonaContainerBackend against the real Daytona API.

Usage:
    uv run scripts/test_daytona.py
    DAYTONA_API_KEY=dtn_... uv run scripts/test_daytona.py
"""

import os
import sys

from dotenv import load_dotenv

from cube.backends.daytona import DaytonaContainerBackend
from cube.container import ContainerSpec

from test_harness import make_health_check_tests, make_tests, run_all

# Load .env if DAYTONA_API_KEY not already set
if not os.environ.get("DAYTONA_API_KEY"):
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"Loaded env from {env_path}")

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
