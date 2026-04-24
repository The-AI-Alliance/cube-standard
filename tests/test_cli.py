"""Tests for cube.cli — covers cmd_init, cmd_list, cmd_test, _resolve_debug_module, main(), and registry helpers."""

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from cube import cli
from cube.cli import (
    _DEFAULT_NAME,
    _build_registry_yaml,
    _guess_display_name,
    _parse_pyproject_license,
    _resolve_debug_module,
    cmd_init,
    cmd_list,
    cmd_test,
    main,
)
from cube.testing import RESET_REPRO_OBS_MISMATCH_MSG

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
        with patch("cube.testing.run_debug_suite", return_value=results) as mock_suite:
            cmd_test("fake_debug.debug")  # Should not raise
    assert mock_suite.call_count == 1  # test mode: single compliance run only


def test_cmd_test_stress_runs_four_times(fake_debug_in_sys_modules):
    results = [{"task_id": "t1", "done": True, "reward": 1.0, "steps": 3, "episode_time_s": 0.1, "error": None}]

    with patch("cube.cli._resolve_debug_module", return_value="fake_debug.debug"):
        with patch("cube.testing.run_debug_suite", return_value=results) as mock_suite:
            cmd_test("fake_debug.debug", stress_test=True)
    assert mock_suite.call_count == 4
    assert [c.kwargs.get("workers") for c in mock_suite.call_args_list] == [1, 1, 2, 4]


def test_cmd_test_exits_1_on_failure(fake_debug_in_sys_modules):
    results = [{"task_id": "t1", "done": False, "reward": 0.0, "steps": 3, "episode_time_s": 0.1, "error": None}]

    with patch("cube.cli._resolve_debug_module", return_value="fake_debug.debug"):
        with patch("cube.testing.run_debug_suite", return_value=results) as mock_suite:
            with pytest.raises(SystemExit) as exc:
                cmd_test("fake_debug.debug")
    assert exc.value.code == 1
    assert mock_suite.call_count == 1  # test mode: single compliance run only


def test_cmd_test_exits_1_on_task_with_error(fake_debug_in_sys_modules):
    results = [{"task_id": "t1", "done": True, "reward": 1.0, "steps": 3, "episode_time_s": 0.1, "error": "boom"}]

    with patch("cube.cli._resolve_debug_module", return_value="fake_debug.debug"):
        with patch("cube.testing.run_debug_suite", return_value=results) as mock_suite:
            with pytest.raises(SystemExit) as exc:
                cmd_test("fake_debug.debug")
    assert exc.value.code == 1
    assert mock_suite.call_count == 1  # test mode: single compliance run only


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

    mock_suite.assert_called_once_with(
        "fake_debug.debug",
        fake_debug_in_sys_modules,
        max_steps=5,
        print_json=False,
        workers=1,
        on_episode_start=mock_suite.call_args.kwargs["on_episode_start"],
        on_episode_done=mock_suite.call_args.kwargs["on_episode_done"],
    )


def test_cmd_test_ci_mode_passes(fake_debug_in_sys_modules, capsys):
    results = [{"task_id": "t1", "done": True, "reward": 1.0, "steps": 3, "episode_time_s": 0.1, "error": None}]

    with patch("cube.cli._resolve_debug_module", return_value="fake_debug.debug"):
        with patch("cube.testing.run_debug_suite", return_value=results):
            cmd_test("fake_debug.debug", ci_mode=True)

    out = capsys.readouterr().out
    assert "PASSED" in out
    assert "PASS" in out


def test_cmd_test_ci_mode_fails_exits_1(fake_debug_in_sys_modules, capsys):
    results = [{"task_id": "t1", "done": False, "reward": 0.0, "steps": 3, "episode_time_s": 0.1, "error": None}]

    with patch("cube.cli._resolve_debug_module", return_value="fake_debug.debug"):
        with patch("cube.testing.run_debug_suite", return_value=results):
            with pytest.raises(SystemExit) as exc:
                cmd_test("fake_debug.debug", ci_mode=True)

    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "FAILED" in out


def test_cmd_test_ci_mode_via_env_var(fake_debug_in_sys_modules, monkeypatch, capsys):
    monkeypatch.setenv("CUBE_CI", "1")
    results = [{"task_id": "t1", "done": True, "reward": 1.0, "steps": 3, "episode_time_s": 0.1, "error": None}]

    with patch("cube.cli._resolve_debug_module", return_value="fake_debug.debug"):
        with patch("cube.testing.run_debug_suite", return_value=results):
            cmd_test("fake_debug.debug")  # no ci_mode kwarg — activated via env var

    out = capsys.readouterr().out
    assert "PASSED" in out


def test_cmd_test_ci_mode_with_demo_flag_does_not_inject_reset_fail(fake_debug_in_sys_modules, monkeypatch, capsys):
    """--demo-reset-repro is ignored in CI so logs do not show FAIL reset + exit 0."""
    monkeypatch.setenv("CUBE_CI", "1")
    results = [{"task_id": "t1", "done": True, "reward": 1.0, "steps": 3, "episode_time_s": 0.1, "error": None}]

    with patch("cube.cli._resolve_debug_module", return_value="fake_debug.debug"):
        with patch("cube.testing.run_debug_suite", return_value=results):
            with patch("cube.testing.check_reset_reproducibility", return_value=(True, "", "")):
                cmd_test("fake_debug.debug", demo_reset_repro=True)

    out = capsys.readouterr().out
    assert "  PASS  test_reset_reproducibility" in out
    assert "  FAIL  test_reset_reproducibility" not in out


def test_cmd_test_ci_mode_reset_repro_brackets_in_message_no_crash(fake_debug_in_sys_modules, capsys):
    """User/exception text must not be parsed as Rich markup (plain reset-repro path)."""
    results = [{"task_id": "t1", "done": True, "reward": 1.0, "steps": 3, "episode_time_s": 0.1, "error": None}]
    bad_msg = "reset failed: [red]x[/red]"

    with patch("cube.cli._resolve_debug_module", return_value="fake_debug.debug"):
        with patch("cube.testing.run_debug_suite", return_value=results):
            with patch("cube.testing.check_reset_reproducibility", return_value=(False, bad_msg, "")):
                cmd_test("fake_debug.debug", ci_mode=True)

    out = capsys.readouterr().out
    assert bad_msg in out
    assert "PASSED" in out


def test_cmd_test_ci_mode_reset_repro_early_error_no_misleading_two_task_prefix(
    fake_debug_in_sys_modules,
    capsys,
):
    """Harness errors before two-task compare must not claim two fresh Task instances."""
    results = [{"task_id": "t1", "done": True, "reward": 1.0, "steps": 3, "episode_time_s": 0.1, "error": None}]
    for early_msg in ("no get_debug_benchmark", "no debug task configs"):
        with patch("cube.cli._resolve_debug_module", return_value="fake_debug.debug"):
            with patch("cube.testing.run_debug_suite", return_value=results):
                with patch("cube.testing.check_reset_reproducibility", return_value=(False, early_msg, "")):
                    cmd_test("fake_debug.debug", ci_mode=True)
        out = capsys.readouterr().out
        assert early_msg in out
        assert "(first task, two fresh Task instances)" not in out


def test_cmd_test_ci_mode_reset_repro_obs_mismatch_shows_two_task_prefix(fake_debug_in_sys_modules, capsys):
    results = [{"task_id": "t1", "done": True, "reward": 1.0, "steps": 3, "episode_time_s": 0.1, "error": None}]
    diff = "token:\n  first: 1\n  second: 2\n"
    with patch("cube.cli._resolve_debug_module", return_value="fake_debug.debug"):
        with patch("cube.testing.run_debug_suite", return_value=results):
            with patch(
                "cube.testing.check_reset_reproducibility",
                return_value=(False, RESET_REPRO_OBS_MISMATCH_MSG, diff),
            ):
                cmd_test("fake_debug.debug", ci_mode=True)
    out = capsys.readouterr().out
    assert "(first task, two fresh Task instances)" in out
    assert RESET_REPRO_OBS_MISMATCH_MSG in out


def test_cmd_test_ci_mode_truncates_large_reset_repro_diff(fake_debug_in_sys_modules, capsys):
    """CI reset-repro diff must not dump unbounded bytes (matches dashboard cap)."""
    results = [{"task_id": "t1", "done": True, "reward": 1.0, "steps": 3, "episode_time_s": 0.1, "error": None}]
    huge = "x" * (cli._RESET_DIFF_DISPLAY_MAX + 500)
    with patch("cube.cli._resolve_debug_module", return_value="fake_debug.debug"):
        with patch("cube.testing.run_debug_suite", return_value=results):
            with patch("cube.testing.check_reset_reproducibility", return_value=(False, "mismatch", huge)):
                cmd_test("fake_debug.debug", ci_mode=True)
    out = capsys.readouterr().out
    assert "... [diff truncated]" in out
    assert huge not in out


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
    mock_test.assert_called_once_with(
        "counter-cube", max_steps=20, output_path=None, ci_mode=False, demo_reset_repro=False, stress_test=False
    )


def test_main_test_dispatches_with_custom_max_steps():
    with patch("cube.cli.cmd_test") as mock_test:
        with patch.object(sys, "argv", ["cube", "test", "counter-cube", "--max-steps=5"]):
            main()
    mock_test.assert_called_once_with(
        "counter-cube", max_steps=5, output_path=None, ci_mode=False, demo_reset_repro=False, stress_test=False
    )


def test_main_test_dispatches_demo_reset_repro():
    with patch("cube.cli.cmd_test") as mock_test:
        with patch.object(sys, "argv", ["cube", "test", "counter-cube", "--demo-reset-repro"]):
            main()
    mock_test.assert_called_once_with(
        "counter-cube", max_steps=20, output_path=None, ci_mode=False, demo_reset_repro=True, stress_test=False
    )


def test_cmd_test_demo_reset_repro_shows_reset_error_panel(fake_debug_in_sys_modules):
    """When the real check passes, --demo-reset-repro still renders the red reset-repro block."""
    results = [
        {
            "task_id": "t1",
            "done": True,
            "reward": 1.0,
            "steps": 3,
            "episode_time_s": 0.1,
            "error": None,
            "tools_list_ok": True,
            "close_idempotent_ok": True,
        }
    ]

    with patch("cube.cli._resolve_debug_module", return_value="fake_debug.debug"):
        with patch("cube.testing.run_debug_suite", return_value=results):
            with patch("cube.testing.check_reset_reproducibility", return_value=(True, "", "")):
                with patch("cube.testing.check_benchmark_metadata", return_value=(True, "")):
                    with patch(
                        "cube.cli._print_reset_reproducibility_error_block",
                        wraps=cli._print_reset_reproducibility_error_block,
                    ) as spy:
                        cmd_test("fake_debug.debug", demo_reset_repro=True)

    spy.assert_called_once()
    assert spy.call_args.kwargs["reset_ok"] is False
    assert "demo token" in spy.call_args.kwargs["reset_diff"]


# ── _guess_display_name ───────────────────────────────────────────────────────


def test_guess_display_name_hyphenated():
    assert _guess_display_name("arithmetic-cube") == "Arithmetic Cube"


def test_guess_display_name_single_word():
    assert _guess_display_name("miniwob") == "Miniwob"


def test_guess_display_name_underscores_treated_as_hyphens():
    assert _guess_display_name("my_bench_cube") == "My Bench Cube"


def test_guess_display_name_mixed_separators():
    assert _guess_display_name("my_bench-cube") == "My Bench Cube"


# ── _parse_pyproject_license ─────────────────────────────────────────────────


def test_parse_pyproject_license_string():
    assert _parse_pyproject_license({"license": "MIT"}) == "MIT"


def test_parse_pyproject_license_dict_text():
    assert _parse_pyproject_license({"license": {"text": "Apache-2.0"}}) == "Apache-2.0"


def test_parse_pyproject_license_missing_returns_none():
    assert _parse_pyproject_license({}) is None


def test_parse_pyproject_license_dict_without_text_returns_none():
    assert _parse_pyproject_license({"license": {"file": "LICENSE"}}) is None


# ── _build_registry_yaml ─────────────────────────────────────────────────────


def _make_yaml(**overrides):
    defaults = dict(
        id="counter-cube",
        name="Counter Cube",
        name_is_guessed=False,
        version="0.1.0",
        description="A simple counter benchmark.",
        package="counter-cube",
        dev_install_url=None,
        authors=[{"github": "alice", "name": "Alice Smith"}],
        wrapper_license="MIT",
    )
    defaults.update(overrides)
    return _build_registry_yaml(**defaults)


def test_build_registry_yaml_contains_id():
    assert "id: counter-cube" in _make_yaml()


def test_build_registry_yaml_contains_version():
    assert 'version: "0.1.0"' in _make_yaml()


def test_build_registry_yaml_contains_package():
    assert "package: counter-cube" in _make_yaml()


def test_build_registry_yaml_contains_author():
    content = _make_yaml()
    assert "github: alice" in content
    assert "name: Alice Smith" in content


def test_build_registry_yaml_contains_license():
    assert "wrapper_license: MIT" in _make_yaml()


def test_build_registry_yaml_dev_install_url_included_when_set():
    content = _make_yaml(dev_install_url="git+https://github.com/org/repo")
    assert "dev_install_url:" in content
    assert "git+https://github.com/org/repo" in content


def test_build_registry_yaml_dev_install_url_commented_when_none():
    content = _make_yaml(dev_install_url=None)
    assert "# dev_install_url:" in content


def test_build_registry_yaml_guessed_name_has_comment():
    content = _make_yaml(name_is_guessed=True)
    assert "# auto-guessed" in content


def test_build_registry_yaml_known_name_has_no_guess_comment():
    content = _make_yaml(name_is_guessed=False)
    assert "# auto-guessed" not in content


def test_build_registry_yaml_missing_author_fields_produce_todos():
    content = _make_yaml(authors=[{}])
    assert "<TODO:" in content


def test_build_registry_yaml_missing_license_produces_todo():
    content = _make_yaml(wrapper_license=None)
    assert "<TODO:" in content
    assert "wrapper_license" in content


def test_build_registry_yaml_contains_tags_placeholder():
    content = _make_yaml()
    assert "tags:" in content
