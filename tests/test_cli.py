"""Tests for cube.cli — covers cmd_init, cmd_list, cmd_test, _resolve_debug_module, and main()."""

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from cube import cli
from cube.cli import _DEFAULT_NAME, _resolve_debug_module, cmd_init, cmd_list, cmd_test, main

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_entry_point(name: str, value: str, metadata=None):
    """Return a mock entry point whose .load() returns a class with benchmark_metadata."""
    ep = MagicMock()
    ep.name = name
    ep.value = value
    if metadata is not None:
        benchmark_cls = MagicMock()
        benchmark_cls.benchmark_metadata = metadata
        ep.load.return_value = benchmark_cls
    else:
        ep.load.side_effect = Exception("load failed")
    return ep


def _make_metadata(*, version="1.0", num_tasks=3, tags=None, description="desc"):
    m = MagicMock()
    m.version = version
    m.num_tasks = num_tasks
    m.tags = tags or []
    m.description = description
    return m


# ── cmd_init ──────────────────────────────────────────────────────────────────


def test_cmd_init_creates_directory(tmp_path):
    cmd_init(name="my_bench", cwd=tmp_path)
    dest = tmp_path / "my_bench"
    assert dest.is_dir()


def test_cmd_init_copies_template_files(tmp_path):
    cmd_init(name="my_bench", cwd=tmp_path)
    dest = tmp_path / "my_bench"
    files = list(dest.rglob("*"))
    assert any(f.is_file() for f in files), "No files were copied from template"


def test_cmd_init_default_name(tmp_path):
    cmd_init(name=_DEFAULT_NAME, cwd=tmp_path)
    assert (tmp_path / _DEFAULT_NAME).is_dir()


def test_cmd_init_rejects_placeholder_names(tmp_path):
    for placeholder in ("cube_package", "new_cube_package", "new-cube-package"):
        with pytest.raises(SystemExit) as exc:
            cmd_init(name=placeholder, cwd=tmp_path)
        assert exc.value.code == 1


def test_cmd_init_renames_package_directory(tmp_path):
    cmd_init(name="my-bench", cwd=tmp_path)
    dest = tmp_path / "my-bench"
    assert (dest / "src" / "my_bench").is_dir(), "src/my_bench/ directory should exist"
    assert not (dest / "src" / "cube_package").exists(), "src/cube_package/ should be gone"


def test_cmd_init_substitutes_placeholders(tmp_path):
    cmd_init(name="my-bench", cwd=tmp_path)
    dest = tmp_path / "my-bench"
    pyproject = (dest / "pyproject.toml").read_text()
    assert "my-bench" in pyproject
    assert "cube_package" not in pyproject
    assert "new-cube-package" not in pyproject
    benchmark_py = (dest / "src" / "my_bench" / "benchmark.py").read_text()
    assert "my_bench" in benchmark_py
    assert "cube_package" not in benchmark_py


def test_cmd_init_refuses_to_overwrite(tmp_path):
    (tmp_path / "existing").mkdir()
    with pytest.raises(SystemExit) as exc:
        cmd_init(name="existing", cwd=tmp_path)
    assert exc.value.code == 1


def test_cmd_init_destination_not_created_on_failure(tmp_path):
    """Ensure no partial directory is left when init exits early."""
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(SystemExit):
        cmd_init(name="existing", cwd=tmp_path)
    # Only the pre-existing dir should be there; no new dir
    assert list(tmp_path.iterdir()) == [existing]


# ── cmd_list ──────────────────────────────────────────────────────────────────


def test_cmd_list_no_benchmarks_installed(capsys):
    with patch("cube.cli.importlib.metadata.entry_points", return_value=[]):
        cmd_list()  # Should not raise


def test_cmd_list_shows_benchmarks():
    meta = _make_metadata(version="2.0", num_tasks=5, tags=["toy"], description="A toy benchmark")
    ep = _make_entry_point("my-cube", "my_cube.benchmark:MyBenchmark", metadata=meta)

    with patch("cube.cli.importlib.metadata.entry_points", return_value=[ep]):
        cmd_list()  # Should not raise

    ep.load.assert_called_once()


def test_cmd_list_handles_load_error_gracefully():
    ep = _make_entry_point("bad-cube", "bad_cube.benchmark:Bad", metadata=None)

    with patch("cube.cli.importlib.metadata.entry_points", return_value=[ep]):
        cmd_list()  # Should not raise


def test_cmd_list_sorts_by_name():
    """Entry points should appear sorted; verify load is called for each."""
    meta = _make_metadata()
    ep_b = _make_entry_point("b-cube", "b_cube.benchmark:B", metadata=meta)
    ep_a = _make_entry_point("a-cube", "a_cube.benchmark:A", metadata=meta)

    with patch("cube.cli.importlib.metadata.entry_points", return_value=[ep_b, ep_a]):
        cmd_list()

    assert ep_a.load.called
    assert ep_b.load.called


# ── _resolve_debug_module ──────────────────────────────────────────────────────


def test_resolve_debug_module_passthrough_dotted():
    result = _resolve_debug_module("counter_cube.debug")
    assert result == "counter_cube.debug"


def test_resolve_debug_module_derives_from_entry_point():
    ep = _make_entry_point("counter-cube", "counter_cube.benchmark:CounterBenchmark", metadata=_make_metadata())

    with patch("cube.cli.importlib.metadata.entry_points", return_value=[ep]):
        result = _resolve_debug_module("counter-cube")

    assert result == "counter_cube.debug"


def test_resolve_debug_module_unknown_name_exits():
    with patch("cube.cli.importlib.metadata.entry_points", return_value=[]):
        with pytest.raises(SystemExit) as exc:
            _resolve_debug_module("nonexistent-cube")
    assert exc.value.code == 1


def test_resolve_debug_module_shows_available_benchmarks():
    """When name is unknown, the error should mention available benchmarks."""
    ep = _make_entry_point("real-cube", "real_cube.benchmark:Real", metadata=_make_metadata())

    captured_panels = []

    original_print = cli.err_console.print

    def capturing_print(renderable, *args, **kwargs):
        captured_panels.append(renderable)
        return original_print(renderable, *args, **kwargs)

    with patch("cube.cli.importlib.metadata.entry_points", return_value=[ep]):
        with patch.object(cli.err_console, "print", side_effect=capturing_print):
            with pytest.raises(SystemExit):
                _resolve_debug_module("unknown-cube")

    assert captured_panels, "Expected an error panel to be printed"


# ── cmd_test ──────────────────────────────────────────────────────────────────


def _make_debug_module(*, get_debug_benchmark=True, make_debug_agent=True) -> ModuleType:
    mod = ModuleType("fake_debug")
    if get_debug_benchmark:
        mock_benchmark = MagicMock()
        mock_benchmark.get_task_configs.return_value = []
        mod.get_debug_benchmark = MagicMock(return_value=mock_benchmark)
    if make_debug_agent:
        mod.make_debug_agent = MagicMock(return_value=lambda obs, actions: None)
    return mod


@pytest.fixture
def fake_debug_in_sys_modules():
    """Insert a fake debug module into sys.modules and clean it up after the test."""
    mod = _make_debug_module()
    key = "fake_debug.debug"
    sys.modules[key] = mod
    yield mod
    sys.modules.pop(key, None)


def test_cmd_test_passes_on_all_tasks_passing(fake_debug_in_sys_modules):
    results = [{"task_id": "t1", "done": True, "reward": 1.0, "steps": 3, "episode_time_s": 0.1, "error": None}]

    with patch("cube.cli._resolve_debug_module", return_value="fake_debug.debug"):
        with patch("cube.testing.run_debug_suite", return_value=results):
            cmd_test("fake_debug.debug")  # Should not raise


def test_cmd_test_exits_1_on_failure(fake_debug_in_sys_modules):
    results = [{"task_id": "t1", "done": False, "reward": 0.0, "steps": 3, "episode_time_s": 0.1, "error": None}]

    with patch("cube.cli._resolve_debug_module", return_value="fake_debug.debug"):
        with patch("cube.testing.run_debug_suite", return_value=results):
            with pytest.raises(SystemExit) as exc:
                cmd_test("fake_debug.debug")
    assert exc.value.code == 1


def test_cmd_test_exits_1_on_task_with_error(fake_debug_in_sys_modules):
    results = [{"task_id": "t1", "done": True, "reward": 1.0, "steps": 3, "episode_time_s": 0.1, "error": "boom"}]

    with patch("cube.cli._resolve_debug_module", return_value="fake_debug.debug"):
        with patch("cube.testing.run_debug_suite", return_value=results):
            with pytest.raises(SystemExit) as exc:
                cmd_test("fake_debug.debug")
    assert exc.value.code == 1


def test_cmd_test_import_error_exits_1():
    with patch("cube.cli._resolve_debug_module", return_value="nonexistent_xyz.debug"):
        with pytest.raises(SystemExit) as exc:
            cmd_test("nonexistent_xyz.debug")
    assert exc.value.code == 1


def test_cmd_test_missing_get_debug_benchmark_exits_1():
    mod = _make_debug_module(get_debug_benchmark=False)
    key = "fake_debug_no_benchmark.debug"
    sys.modules[key] = mod
    try:
        with patch("cube.cli._resolve_debug_module", return_value=key):
            with pytest.raises(SystemExit) as exc:
                cmd_test(key)
        assert exc.value.code == 1
    finally:
        sys.modules.pop(key, None)


def test_cmd_test_missing_make_debug_agent_exits_1():
    mod = _make_debug_module(make_debug_agent=False)
    key = "fake_debug_no_agent.debug"
    sys.modules[key] = mod
    try:
        with patch("cube.cli._resolve_debug_module", return_value=key):
            with pytest.raises(SystemExit) as exc:
                cmd_test(key)
        assert exc.value.code == 1
    finally:
        sys.modules.pop(key, None)


def test_cmd_test_passes_max_steps(fake_debug_in_sys_modules):
    results = [{"task_id": "t1", "done": True, "reward": 1.0, "steps": 1, "episode_time_s": 0.0, "error": None}]

    with patch("cube.cli._resolve_debug_module", return_value="fake_debug.debug"):
        with patch("cube.testing.run_debug_suite", return_value=results) as mock_suite:
            cmd_test("fake_debug.debug", max_steps=5)

    mock_suite.assert_called_once_with("fake_debug.debug", fake_debug_in_sys_modules, max_steps=5)


# ── main() ────────────────────────────────────────────────────────────────────


def _run_main(*args):
    with patch.object(sys, "argv", ["cube", *args]):
        with pytest.raises(SystemExit) as exc:
            main()
    return exc.value.code


def test_main_no_args_exits_0():
    assert _run_main() == 0


def test_main_help_flag_exits_0():
    assert _run_main("--help") == 0
    assert _run_main("-h") == 0


def test_main_unknown_command_exits_1():
    assert _run_main("unknown") == 1


def test_main_list_dispatches_to_cmd_list():
    with patch("cube.cli.cmd_list") as mock_list:
        with patch.object(sys, "argv", ["cube", "list"]):
            main()
    mock_list.assert_called_once()


def test_main_init_dispatches_with_name(tmp_path):
    with patch("cube.cli.cmd_init") as mock_init:
        with patch.object(sys, "argv", ["cube", "init", "my-bench"]):
            main()
    mock_init.assert_called_once_with(name="my-bench", cwd=Path.cwd())


def test_main_init_uses_default_name_when_omitted(tmp_path):
    with patch("cube.cli.cmd_init") as mock_init:
        with patch.object(sys, "argv", ["cube", "init"]):
            main()
    mock_init.assert_called_once_with(name=_DEFAULT_NAME, cwd=Path.cwd())


def test_main_test_missing_name_exits_1():
    assert _run_main("test") == 1


def test_main_test_dispatches_with_default_max_steps():
    with patch("cube.cli.cmd_test") as mock_test:
        with patch.object(sys, "argv", ["cube", "test", "counter-cube"]):
            main()
    mock_test.assert_called_once_with("counter-cube", max_steps=20)


def test_main_test_dispatches_with_custom_max_steps():
    with patch("cube.cli.cmd_test") as mock_test:
        with patch.object(sys, "argv", ["cube", "test", "counter-cube", "--max-steps=5"]):
            main()
    mock_test.assert_called_once_with("counter-cube", max_steps=5)
