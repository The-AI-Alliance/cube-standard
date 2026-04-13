"""Unit tests for cube.introspect.find_benchmark_class()."""

from __future__ import annotations

from types import ModuleType
from unittest.mock import MagicMock, patch

from cube.introspect import find_benchmark_class

# ── helpers ───────────────────────────────────────────────────────────────────


def _fake_ep(name: str, cls):
    """Return a mock entry point that loads *cls*."""
    ep = MagicMock()
    ep.name = name
    ep.load.return_value = cls
    return ep


def _fake_module(**attrs) -> ModuleType:
    mod = ModuleType("fake_pkg")
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


class _FakeBenchmark:
    def get_task_configs(self):
        return []


class _NotABenchmark:
    pass


# ── entry-point resolution path ───────────────────────────────────────────────


def test_entry_point_found_and_loaded():
    ep = _fake_ep("my-pkg", _FakeBenchmark)
    with patch("cube.introspect.importlib.metadata.entry_points", return_value=[ep]):
        cls, err = find_benchmark_class("my-pkg")
    assert cls is _FakeBenchmark
    assert err == ""


def test_entry_point_load_failure_returns_error():
    ep = MagicMock()
    ep.name = "my-pkg"
    ep.load.side_effect = ImportError("bad module")
    with patch("cube.introspect.importlib.metadata.entry_points", return_value=[ep]):
        cls, err = find_benchmark_class("my-pkg")
    assert cls is None
    assert "bad module" in err


def test_entry_point_name_mismatch_falls_through_to_import():
    """When no entry point matches, we fall through to the import-based path."""
    ep = _fake_ep("other-pkg", _FakeBenchmark)
    mod = _fake_module(Benchmark=_FakeBenchmark)
    with patch("cube.introspect.importlib.metadata.entry_points", return_value=[ep]):
        with patch("cube.introspect.importlib.import_module", return_value=mod):
            cls, err = find_benchmark_class("my-pkg")
    assert cls is _FakeBenchmark
    assert err == ""


# ── import-based resolution path ─────────────────────────────────────────────


def test_import_path_finds_benchmark_attribute():
    mod = _fake_module(Benchmark=_FakeBenchmark)
    with patch("cube.introspect.importlib.metadata.entry_points", return_value=[]):
        with patch("cube.introspect.importlib.import_module", return_value=mod):
            cls, err = find_benchmark_class("my-pkg")
    assert cls is _FakeBenchmark
    assert err == ""


def test_import_path_finds_benchmark_via_dir_scan():
    """Finds Benchmark via dir() scan when it's not exported as top-level 'Benchmark' name."""

    class Benchmark:  # class name must be "Benchmark" for the dir scan to find it
        pass

    mod = _fake_module(AliasedName=Benchmark)
    with patch("cube.introspect.importlib.metadata.entry_points", return_value=[]):
        with patch("cube.introspect.importlib.import_module", return_value=mod):
            cls, err = find_benchmark_class("my-pkg")
    assert cls is Benchmark
    assert err == ""


def test_import_error_returns_error_message():
    with patch("cube.introspect.importlib.metadata.entry_points", return_value=[]):
        with patch("cube.introspect.importlib.import_module", side_effect=ImportError("no module")):
            cls, err = find_benchmark_class("nonexistent-pkg")
    assert cls is None
    assert "nonexistent-pkg" in err


def test_no_benchmark_class_in_module_returns_error():
    mod = _fake_module(SomeClass=_NotABenchmark)
    with patch("cube.introspect.importlib.metadata.entry_points", return_value=[]):
        with patch("cube.introspect.importlib.import_module", return_value=mod):
            cls, err = find_benchmark_class("my-pkg")
    assert cls is None
    assert "my-pkg" in err
    assert "entry point" in err.lower() or "benchmark" in err.lower()


def test_hyphen_in_package_name_converts_to_underscore_for_import():
    """find_benchmark_class("my-pkg") must import "my_pkg", not "my-pkg"."""
    mod = _fake_module(Benchmark=_FakeBenchmark)
    with patch("cube.introspect.importlib.metadata.entry_points", return_value=[]):
        with patch("cube.introspect.importlib.import_module", return_value=mod) as mock_import:
            find_benchmark_class("my-pkg")
    mock_import.assert_called_once_with("my_pkg")
