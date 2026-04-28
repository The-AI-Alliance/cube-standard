"""Benchmark discovery utilities — shared by the CUBE CLI and registry tooling."""

from __future__ import annotations

import importlib
import importlib.metadata
from typing import Any


def find_benchmark_class(package: str) -> tuple[Any | None, str]:
    """Resolve the BenchmarkConfig class for *package* via the ``cube.benchmarks`` entry point.

    Returns ``(BenchmarkConfigClass, "")`` on success or ``(None, error_message)`` on failure.
    """
    try:
        eps = importlib.metadata.entry_points(group="cube.benchmarks")
        matched = [ep for ep in eps if ep.name == package]
    except Exception as e:
        return None, f"Failed to query entry points: {e}"

    if not matched:
        return None, (
            f"Package '{package}' has no registered 'cube.benchmarks' entry point. "
            f"Add one to pyproject.toml:\n"
            f"  [project.entry-points.'cube.benchmarks']\n"
            f'  {package} = "your_module:YourBenchmarkConfig"'
        )

    try:
        cls = matched[0].load()
        return cls, ""
    except Exception as e:
        return None, f"Entry point '{matched[0].value}' failed to load: {e}"
