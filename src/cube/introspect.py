"""Benchmark discovery utilities — shared by the CUBE CLI and registry tooling."""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
from typing import Any


def find_benchmark_class(package: str) -> tuple[Any | None, str]:
    """Resolve the Benchmark class for *package*.

    Resolution order:
    1. ``cube.benchmarks`` entry point matching the package name (canonical CUBE mechanism).
    2. A class literally named ``Benchmark`` exported at the package top level.

    Returns ``(BenchmarkClass, "")`` on success or ``(None, error_message)`` on failure.
    """
    # 1. Try the cube.benchmarks entry point.
    try:
        eps = importlib.metadata.entry_points(group="cube.benchmarks")
        matched = [ep for ep in eps if ep.name == package]
        if matched:
            try:
                cls = matched[0].load()
                return cls, ""
            except Exception as e:
                return None, f"Entry point '{matched[0].value}' failed to load: {e}"
    except Exception:
        pass  # fall through to import-based discovery

    # TODO(standardization): this fallback exists because existing cubes don't all register
    # a [project.entry-points.'cube.benchmarks'] entry point in their pyproject.toml.
    # Once all cubes are standardized (see introspection_redesign.md §4), this block can be
    # replaced with a hard failure: "no entry point registered → reject".
    # 2. Import the package module and look for a class named Benchmark.
    try:
        mod = importlib.import_module(package.replace("-", "_"))
    except ImportError as e:
        return None, f"Could not import package '{package}': {e}"

    benchmark_cls = getattr(mod, "Benchmark", None)
    if benchmark_cls is None:
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name, None)
            if attr is not None and inspect.isclass(attr) and attr.__name__ == "Benchmark":
                benchmark_cls = attr
                break

    if benchmark_cls is None:
        return None, (
            f"Package '{package}' has no 'cube.benchmarks' entry point and does not export "
            f"a class named 'Benchmark'. Register an entry point in pyproject.toml:\n"
            f"  [project.entry-points.'cube.benchmarks']\n"
            f'  {package} = "your_module:YourBenchmark"'
        )

    return benchmark_cls, ""
