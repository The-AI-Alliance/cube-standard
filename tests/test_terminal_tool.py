"""Tests for cube.tools.terminal — TerminalTool (Protocol), ContainerTerminalTool,
TerminalToolConfig, _format_exec_result, _truncate_output, and _parse_line_range."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from cube.container import ExecResult
from cube.tools.terminal import (
    ContainerTerminalTool,
    TerminalTool,
    TerminalToolConfig,
    _format_exec_result,
    _parse_line_range,
    _truncate_output,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool(
    *,
    stdout: str = "",
    stderr: str = "",
    exit_code: int = 0,
    **cfg_kwargs: Any,
) -> ContainerTerminalTool:
    """Build a ContainerTerminalTool backed by a mock container that always returns the given result."""
    container = MagicMock()
    container.exec.return_value = ExecResult(stdout=stdout, stderr=stderr, exit_code=exit_code)
    cfg = TerminalToolConfig(**cfg_kwargs)
    return ContainerTerminalTool(config=cfg, container=container)


# ---------------------------------------------------------------------------
# _truncate_output
# ---------------------------------------------------------------------------


class TestTruncateOutput:
    def test_short_text_unchanged(self) -> None:
        assert _truncate_output("hello", 100) == "hello"

    def test_exact_limit_unchanged(self) -> None:
        text = "a" * 100
        assert _truncate_output(text, 100) == text

    def test_truncated_contains_elided_marker(self) -> None:
        text = "a" * 200
        result = _truncate_output(text, 100)
        assert "bytes elided" in result

    def test_truncated_respects_max_bytes(self) -> None:
        text = "x" * 1000
        result = _truncate_output(text, 100)
        # Result may exceed max_bytes slightly due to the marker string, but
        # head + tail together must not exceed max_bytes.
        head_tail_bytes = len(result.split("bytes elided")[0].encode()) + len(result.split("\n")[-1].encode())
        assert head_tail_bytes <= 100 + 10  # small slack for partial decoding

    def test_head_ratio_zero_is_tail_only(self) -> None:
        text = "HEAD" + ("x" * 200) + "TAIL"
        result = _truncate_output(text, 20, head_ratio=0.0)
        assert "HEAD" not in result
        assert "TAIL" in result

    def test_head_ratio_one_is_head_only(self) -> None:
        text = "HEAD" + ("x" * 200) + "TAIL"
        result = _truncate_output(text, 20, head_ratio=1.0)
        assert "HEAD" in result
        assert "TAIL" not in result

    def test_default_ratio_biases_tail(self) -> None:
        # 1000 bytes, limit 100, head_ratio=0.2 → 20 head / 80 tail
        text = "H" * 500 + "T" * 500
        result = _truncate_output(text, 100)
        head_part = result.split("bytes elided")[0]
        tail_part = result.split("\n")[-1]
        assert len(tail_part) > len(head_part)

    def test_multibyte_utf8_no_decode_error(self) -> None:
        text = "αβγδ" * 100  # 4 bytes per char
        result = _truncate_output(text, 50)
        # Must not raise, and marker must be present
        assert "bytes elided" in result


# ---------------------------------------------------------------------------
# _format_exec_result
# ---------------------------------------------------------------------------


class TestFormatExecResult:
    def test_stdout_only(self) -> None:
        r = ExecResult(stdout="hello", stderr="", exit_code=0)
        assert _format_exec_result(r, 120) == "hello"

    def test_stderr_appended(self) -> None:
        r = ExecResult(stdout="out", stderr="err", exit_code=0)
        result = _format_exec_result(r, 120)
        assert "out" in result
        assert "err" in result

    def test_nonzero_exit_annotated(self) -> None:
        r = ExecResult(stdout="output", stderr="", exit_code=1)
        result = _format_exec_result(r, 120)
        assert "[exit_code: 1]" in result

    def test_zero_exit_not_annotated(self) -> None:
        r = ExecResult(stdout="output", stderr="", exit_code=0)
        assert "[exit_code:" not in _format_exec_result(r, 120)

    def test_timeout_annotated(self) -> None:
        r = ExecResult(stdout="partial", stderr="", exit_code=124)
        result = _format_exec_result(r, 30)
        assert "[timed out after 30s]" in result
        assert "[exit_code:" not in result

    def test_no_output(self) -> None:
        r = ExecResult(stdout="", stderr="", exit_code=0)
        assert _format_exec_result(r, 120) == "(no output)"


# ---------------------------------------------------------------------------
# TerminalToolConfig
# ---------------------------------------------------------------------------


class TestTerminalToolConfig:
    def test_requires_container(self) -> None:
        cfg = TerminalToolConfig()
        with pytest.raises(ValueError, match="requires a container"):
            cfg.make(container=None)

    def test_make_returns_bash_tool(self) -> None:
        container = MagicMock()
        cfg = TerminalToolConfig()
        tool = cfg.make(container=container)
        assert isinstance(tool, TerminalTool)

    def test_non_interactive_env_merged(self) -> None:
        container = MagicMock()
        container.exec.return_value = ExecResult(stdout="ok", stderr="", exit_code=0)
        cfg = TerminalToolConfig(extra_env={"MY_VAR": "1"})
        tool = cfg.make(container=container)
        tool.bash("echo hi")
        _, kwargs = container.exec.call_args
        env = kwargs["env"]
        assert env["PAGER"] == "cat"
        assert env["MY_VAR"] == "1"

    def test_extra_env_overrides_default(self) -> None:
        container = MagicMock()
        container.exec.return_value = ExecResult(stdout="ok", stderr="", exit_code=0)
        cfg = TerminalToolConfig(extra_env={"PAGER": "less"})
        tool = cfg.make(container=container)
        tool.bash("echo hi")
        _, kwargs = container.exec.call_args
        assert kwargs["env"]["PAGER"] == "less"


# ---------------------------------------------------------------------------
# TerminalTool.action_set
# ---------------------------------------------------------------------------


class TestTerminalToolActionSet:
    def test_bash_only_by_default(self) -> None:
        tool = _make_tool()
        names = {a.name for a in tool.action_set}
        assert names == {"bash"}

    def test_file_actions_enabled(self) -> None:
        tool = _make_tool(enable_file_actions=True)
        names = {a.name for a in tool.action_set}
        assert names == {"bash", "read_file", "write_file"}


# ---------------------------------------------------------------------------
# TerminalTool.bash — plain format
# ---------------------------------------------------------------------------


class TestTerminalToolPlain:
    def test_returns_stdout(self) -> None:
        tool = _make_tool(stdout="hello\n")
        assert tool.bash("echo hello") == "hello\n"

    def test_nonzero_exit_appended(self) -> None:
        tool = _make_tool(stdout="", stderr="", exit_code=1)
        result = tool.bash("false")
        assert "[exit_code: 1]" in result

    def test_truncation_applied(self) -> None:
        tool = _make_tool(stdout="x" * 200, max_output_bytes=50)
        result = tool.bash("cat bigfile")
        assert "bytes elided" in result

    def test_timeout_forwarded(self) -> None:
        container = MagicMock()
        container.exec.return_value = ExecResult(stdout="ok", stderr="", exit_code=0)
        tool = ContainerTerminalTool(config=TerminalToolConfig(), container=container)
        tool.bash("sleep 5", timeout=300)
        _, kwargs = container.exec.call_args
        assert kwargs["timeout"] == 300


# ---------------------------------------------------------------------------
# TerminalTool.bash — returncode_envelope format
# ---------------------------------------------------------------------------


class TestTerminalToolEnvelope:
    def test_envelope_wraps_output(self) -> None:
        tool = _make_tool(stdout="hello", output_format="returncode_envelope")
        result = tool.bash("echo hello")
        assert result.startswith("<returncode>0</returncode>")
        assert "<output>" in result
        assert "hello" in result

    def test_nonzero_in_tag_not_appended(self) -> None:
        tool = _make_tool(stdout="err", exit_code=2, output_format="returncode_envelope")
        result = tool.bash("badcmd")
        assert "<returncode>2</returncode>" in result
        assert "[exit_code:" not in result

    def test_envelope_truncates_body_not_header(self) -> None:
        tool = _make_tool(stdout="x" * 200, output_format="returncode_envelope", max_output_bytes=50)
        result = tool.bash("cat")
        # Header must be intact
        assert result.startswith("<returncode>0</returncode>")
        assert "bytes elided" in result


# ---------------------------------------------------------------------------
# TerminalTool.read_file
# ---------------------------------------------------------------------------


class TestTerminalToolReadFile:
    def test_returns_content(self) -> None:
        tool = _make_tool(stdout="line1\nline2\n", enable_file_actions=True)
        assert tool.read_file("/tmp/f.txt") == "line1\nline2\n"

    def test_error_on_nonzero_exit(self) -> None:
        tool = _make_tool(stderr="no such file", exit_code=1, enable_file_actions=True)
        result = tool.read_file("/tmp/missing.txt")
        assert "Error reading" in result

    def test_too_large_returns_error(self) -> None:
        big = "x" * 200
        tool = _make_tool(stdout=big, enable_file_actions=True, max_output_bytes=50)
        result = tool.read_file("/tmp/big.txt")
        assert "too large" in result
        assert "line_start" in result

    def test_line_range_uses_sed(self) -> None:
        container = MagicMock()
        container.exec.return_value = ExecResult(stdout="lines", stderr="", exit_code=0)
        tool = ContainerTerminalTool(config=TerminalToolConfig(enable_file_actions=True), container=container)
        tool.read_file("/tmp/f.txt", line_start=10, line_end=20)
        cmd = container.exec.call_args[0][0]
        assert "sed" in cmd
        assert "10,20" in cmd

    def test_line_range_no_size_check(self) -> None:
        # A range read of a "big" file should return content, not a size error.
        container = MagicMock()
        container.exec.return_value = ExecResult(stdout="x" * 200, stderr="", exit_code=0)
        tool = ContainerTerminalTool(
            config=TerminalToolConfig(enable_file_actions=True, max_output_bytes=50),
            container=container,
        )
        result = tool.read_file("/tmp/f.txt", line_start=1, line_end=5)
        assert "too large" not in result


# ---------------------------------------------------------------------------
# TerminalTool.write_file
# ---------------------------------------------------------------------------


class TestTerminalToolWriteFile:
    def test_success_message(self) -> None:
        container = MagicMock()
        container.exec.return_value = ExecResult(stdout="", stderr="", exit_code=0)
        tool = ContainerTerminalTool(config=TerminalToolConfig(enable_file_actions=True), container=container)
        result = tool.write_file("/tmp/out.txt", "hello")
        assert "Wrote" in result
        assert "5" in result  # len("hello")

    def test_write_error_reported(self) -> None:
        call_count = 0

        def fake_exec(cmd: str, **_: Any) -> ExecResult:
            nonlocal call_count
            call_count += 1
            # mkdir succeeds; printf fails
            if call_count == 1:
                return ExecResult(stdout="", stderr="", exit_code=0)
            return ExecResult(stdout="", stderr="permission denied", exit_code=1)

        container = MagicMock()
        container.exec.side_effect = fake_exec
        tool = ContainerTerminalTool(config=TerminalToolConfig(enable_file_actions=True), container=container)
        result = tool.write_file("/tmp/out.txt", "hello")
        assert "Error writing" in result


class TestTerminalToolResolvedPath:
    """A relative path resolves against the static working_dir, never a prior
    `cd` (shell state does not persist). The reported path must be absolute so
    that mismatch is visible instead of silently corrupting the wrong file."""

    def test_relative_write_reports_absolute_resolved_path(self) -> None:
        container = MagicMock()
        container.exec.return_value = ExecResult(stdout="", stderr="", exit_code=0)
        tool = ContainerTerminalTool(
            config=TerminalToolConfig(enable_file_actions=True, working_dir="/repo"),
            container=container,
        )
        result = tool.write_file("src/app.py", "x")
        assert "/repo/src/app.py" in result

    def test_absolute_write_path_passes_through(self) -> None:
        container = MagicMock()
        container.exec.return_value = ExecResult(stdout="", stderr="", exit_code=0)
        tool = ContainerTerminalTool(
            config=TerminalToolConfig(enable_file_actions=True, working_dir="/repo"),
            container=container,
        )
        result = tool.write_file("/etc/hosts", "x")
        assert "/etc/hosts" in result and "/repo/etc" not in result

    def test_relative_read_error_reports_absolute_resolved_path(self) -> None:
        container = MagicMock()
        container.exec.return_value = ExecResult(stdout="", stderr="No such file", exit_code=1)
        tool = ContainerTerminalTool(
            config=TerminalToolConfig(enable_file_actions=True, working_dir="/repo"),
            container=container,
        )
        result = tool.read_file("missing.txt")
        assert "Error reading /repo/missing.txt" in result


# ---------------------------------------------------------------------------
# _parse_line_range
# ---------------------------------------------------------------------------


class TestParseLineRange:
    def test_plain_ints(self) -> None:
        assert _parse_line_range(10, 20) == (10, 20)

    def test_none_passthrough(self) -> None:
        assert _parse_line_range(None, None) == (None, None)

    def test_string_int(self) -> None:
        assert _parse_line_range("10", "20") == (10, 20)

    def test_range_string_extracts_both(self) -> None:
        # LLM passes whole range as line_start with no line_end
        assert _parse_line_range("[200, 210]", None) == (200, 210)

    def test_dash_range_string(self) -> None:
        assert _parse_line_range("10-20", None) == (10, 20)

    def test_unparseable_returns_none(self) -> None:
        assert _parse_line_range("abc", None) == (None, None)

    def test_read_file_range_string(self) -> None:
        container = MagicMock()
        container.exec.return_value = ExecResult(stdout="lines", stderr="", exit_code=0)
        tool = ContainerTerminalTool(config=TerminalToolConfig(enable_file_actions=True), container=container)
        tool.read_file("/tmp/f.txt", line_start="[200, 210]")
        cmd = container.exec.call_args[0][0]
        assert "200,210" in cmd


# ---------------------------------------------------------------------------
# TerminalTool.bash_unlimited
# ---------------------------------------------------------------------------


class TestBashUnlimited:
    def test_not_in_action_set(self) -> None:
        tool = _make_tool()
        names = {a.name for a in tool.action_set}
        assert "bash_unlimited" not in names

    def test_returns_full_output(self) -> None:
        big = "x" * 200
        tool = _make_tool(stdout=big, max_output_bytes=50)
        # bash() would truncate; bash_unlimited() must not
        assert tool.bash_unlimited("cmd") == big

    def test_exit_code_annotated(self) -> None:
        tool = _make_tool(exit_code=1)
        assert "[exit_code: 1]" in tool.bash_unlimited("false")


# ---------------------------------------------------------------------------
# TerminalToolConfig.max_timeout
# ---------------------------------------------------------------------------


class TestMaxTimeout:
    def test_timeout_capped(self) -> None:
        container = MagicMock()
        container.exec.return_value = ExecResult(stdout="ok", stderr="", exit_code=0)
        tool = ContainerTerminalTool(config=TerminalToolConfig(max_timeout=600), container=container)
        tool.bash("sleep 9999", timeout=9999)
        _, kwargs = container.exec.call_args
        assert kwargs["timeout"] == 600

    def test_timeout_below_cap_unchanged(self) -> None:
        container = MagicMock()
        container.exec.return_value = ExecResult(stdout="ok", stderr="", exit_code=0)
        tool = ContainerTerminalTool(config=TerminalToolConfig(max_timeout=600), container=container)
        tool.bash("sleep 1", timeout=30)
        _, kwargs = container.exec.call_args
        assert kwargs["timeout"] == 30

    def test_no_cap_by_default(self) -> None:
        container = MagicMock()
        container.exec.return_value = ExecResult(stdout="ok", stderr="", exit_code=0)
        tool = ContainerTerminalTool(config=TerminalToolConfig(), container=container)
        tool.bash("sleep 1", timeout=1800)
        _, kwargs = container.exec.call_args
        assert kwargs["timeout"] == 1800
