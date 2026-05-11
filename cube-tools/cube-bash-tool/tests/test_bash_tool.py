"""Tests for cube_bash_tool.bash — BashTool, BashToolConfig, _format_exec_result,
and Tool._truncate_output."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from cube.container import ExecResult
from cube.tool import Tool

from cube_bash_tool.bash import BashTool, BashToolConfig, _format_exec_result

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool(
    *,
    stdout: str = "",
    stderr: str = "",
    exit_code: int = 0,
    **cfg_kwargs: Any,
) -> BashTool:
    """Build a BashTool backed by a mock container that always returns the given result."""
    container = MagicMock()
    container.exec.return_value = ExecResult(stdout=stdout, stderr=stderr, exit_code=exit_code)
    cfg = BashToolConfig(**cfg_kwargs)
    return BashTool(config=cfg, container=container)


# ---------------------------------------------------------------------------
# Tool._truncate_output
# ---------------------------------------------------------------------------


class TestTruncateOutput:
    def test_short_text_unchanged(self) -> None:
        assert Tool._truncate_output("hello", 100) == "hello"

    def test_exact_limit_unchanged(self) -> None:
        text = "a" * 100
        assert Tool._truncate_output(text, 100) == text

    def test_truncated_contains_elided_marker(self) -> None:
        text = "a" * 200
        result = Tool._truncate_output(text, 100)
        assert "bytes elided" in result

    def test_truncated_respects_max_bytes(self) -> None:
        text = "x" * 1000
        result = Tool._truncate_output(text, 100)
        # Result may exceed max_bytes slightly due to the marker string, but
        # head + tail together must not exceed max_bytes.
        head_tail_bytes = len(result.split("bytes elided")[0].encode()) + len(result.split("\n")[-1].encode())
        assert head_tail_bytes <= 100 + 10  # small slack for partial decoding

    def test_head_ratio_zero_is_tail_only(self) -> None:
        text = "HEAD" + ("x" * 200) + "TAIL"
        result = Tool._truncate_output(text, 20, head_ratio=0.0)
        assert "HEAD" not in result
        assert "TAIL" in result

    def test_head_ratio_one_is_head_only(self) -> None:
        text = "HEAD" + ("x" * 200) + "TAIL"
        result = Tool._truncate_output(text, 20, head_ratio=1.0)
        assert "HEAD" in result
        assert "TAIL" not in result

    def test_default_ratio_biases_tail(self) -> None:
        # 1000 bytes, limit 100, head_ratio=0.2 → 20 head / 80 tail
        text = "H" * 500 + "T" * 500
        result = Tool._truncate_output(text, 100)
        head_part = result.split("bytes elided")[0]
        tail_part = result.split("\n")[-1]
        assert len(tail_part) > len(head_part)

    def test_multibyte_utf8_no_decode_error(self) -> None:
        text = "αβγδ" * 100  # 4 bytes per char
        result = Tool._truncate_output(text, 50)
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
# BashToolConfig
# ---------------------------------------------------------------------------


class TestBashToolConfig:
    def test_requires_container(self) -> None:
        cfg = BashToolConfig()
        with pytest.raises(ValueError, match="requires a container"):
            cfg.make(container=None)

    def test_make_returns_bash_tool(self) -> None:
        container = MagicMock()
        cfg = BashToolConfig()
        tool = cfg.make(container=container)
        assert isinstance(tool, BashTool)

    def test_non_interactive_env_merged(self) -> None:
        container = MagicMock()
        container.exec.return_value = ExecResult(stdout="ok", stderr="", exit_code=0)
        cfg = BashToolConfig(extra_env={"MY_VAR": "1"})
        tool = cfg.make(container=container)
        tool.bash("echo hi")
        _, kwargs = container.exec.call_args
        env = kwargs["env"]
        assert env["PAGER"] == "cat"
        assert env["MY_VAR"] == "1"

    def test_extra_env_overrides_default(self) -> None:
        container = MagicMock()
        container.exec.return_value = ExecResult(stdout="ok", stderr="", exit_code=0)
        cfg = BashToolConfig(extra_env={"PAGER": "less"})
        tool = cfg.make(container=container)
        tool.bash("echo hi")
        _, kwargs = container.exec.call_args
        assert kwargs["env"]["PAGER"] == "less"


# ---------------------------------------------------------------------------
# BashTool.action_set
# ---------------------------------------------------------------------------


class TestBashToolActionSet:
    def test_bash_only_by_default(self) -> None:
        tool = _make_tool()
        names = {a.name for a in tool.action_set}
        assert names == {"bash"}

    def test_file_actions_enabled(self) -> None:
        tool = _make_tool(enable_file_actions=True)
        names = {a.name for a in tool.action_set}
        assert names == {"bash", "read_file", "write_file"}


# ---------------------------------------------------------------------------
# BashTool.bash — plain format
# ---------------------------------------------------------------------------


class TestBashToolPlain:
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
        tool = BashTool(config=BashToolConfig(), container=container)
        tool.bash("sleep 5", timeout=300)
        _, kwargs = container.exec.call_args
        assert kwargs["timeout"] == 300


# ---------------------------------------------------------------------------
# BashTool.bash — returncode_envelope format
# ---------------------------------------------------------------------------


class TestBashToolEnvelope:
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
# BashTool.read_file
# ---------------------------------------------------------------------------


class TestBashToolReadFile:
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
        tool = BashTool(config=BashToolConfig(enable_file_actions=True), container=container)
        tool.read_file("/tmp/f.txt", line_start=10, line_end=20)
        cmd = container.exec.call_args[0][0]
        assert "sed" in cmd
        assert "10,20" in cmd

    def test_line_range_no_size_check(self) -> None:
        # A range read of a "big" file should return content, not a size error.
        container = MagicMock()
        container.exec.return_value = ExecResult(stdout="x" * 200, stderr="", exit_code=0)
        tool = BashTool(
            config=BashToolConfig(enable_file_actions=True, max_output_bytes=50),
            container=container,
        )
        result = tool.read_file("/tmp/f.txt", line_start=1, line_end=5)
        assert "too large" not in result


# ---------------------------------------------------------------------------
# BashTool.write_file
# ---------------------------------------------------------------------------


class TestBashToolWriteFile:
    def test_success_message(self) -> None:
        container = MagicMock()
        container.exec.return_value = ExecResult(stdout="", stderr="", exit_code=0)
        tool = BashTool(config=BashToolConfig(enable_file_actions=True), container=container)
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
        tool = BashTool(config=BashToolConfig(enable_file_actions=True), container=container)
        result = tool.write_file("/tmp/out.txt", "hello")
        assert "Error writing" in result
