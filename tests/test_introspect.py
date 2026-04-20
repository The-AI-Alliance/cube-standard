"""Unit tests for cube.introspect.find_benchmark_class()."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from cube.introspect import find_benchmark_class

# ── helpers ───────────────────────────────────────────────────────────────────


def _fake_ep(name: str, cls):
    """Return a mock entry point that loads *cls*."""
    ep = MagicMock()
    ep.name = name
    ep.load.return_value = cls
    return ep


class _FakeBenchmark:
    def get_task_configs(self):
        return []


# ── entry-point resolution ────────────────────────────────────────────────────


def test_entry_point_found_and_loaded():
    ep = _fake_ep("my-pkg", _FakeBenchmark)
    with patch("cube.introspect.importlib.metadata.entry_points", return_value=[ep]):
        cls, err = find_benchmark_class("my-pkg")
    assert cls is _FakeBenchmark
    assert err == ""


def test_entry_point_load_failure_returns_error():
    ep = MagicMock()
    ep.name = "my-pkg"
    ep.value = "my_pkg:Benchmark"
    ep.load.side_effect = ImportError("bad module")
    with patch("cube.introspect.importlib.metadata.entry_points", return_value=[ep]):
        cls, err = find_benchmark_class("my-pkg")
    assert cls is None
    assert err == "Entry point 'my_pkg:Benchmark' failed to load: bad module"


def test_no_matching_entry_point_returns_error():
    ep = _fake_ep("other-pkg", _FakeBenchmark)
    with patch("cube.introspect.importlib.metadata.entry_points", return_value=[ep]):
        cls, err = find_benchmark_class("my-pkg")
    assert cls is None
    assert err == (
        "Package 'my-pkg' has no registered 'cube.benchmarks' entry point. "
        "Add one to pyproject.toml:\n"
        "  [project.entry-points.'cube.benchmarks']\n"
        '  my-pkg = "your_module:YourBenchmark"'
    )


def test_no_entry_points_at_all_returns_error():
    with patch("cube.introspect.importlib.metadata.entry_points", return_value=[]):
        cls, err = find_benchmark_class("my-pkg")
    assert cls is None
    assert err == (
        "Package 'my-pkg' has no registered 'cube.benchmarks' entry point. "
        "Add one to pyproject.toml:\n"
        "  [project.entry-points.'cube.benchmarks']\n"
        '  my-pkg = "your_module:YourBenchmark"'
    )
