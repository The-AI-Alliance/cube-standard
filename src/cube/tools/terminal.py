"""Terminal tool — abstract task-side contract + container-backed reference impl.

``TerminalTool`` is the abstract base for terminal-style tools. It extends
``cube.tool.Tool`` and declares only what *tasks* need from any terminal
implementation — a single task-side primitive, ``bash_unlimited()``, used by
``Task.reset()`` / ``Task.evaluate()`` / ``Task._make_tool()`` for setup and
validation. The abstract intentionally does NOT enumerate ``@tool_action``
methods: the agent-facing action space is the concrete implementation's
choice.

``ContainerTerminalTool`` is the reference implementation — talks to a
``cube.container.Container`` via ``container.exec()``, and exposes
``bash``, ``read_file``, ``write_file`` as ``@tool_action``s (the latter two
gated by ``TerminalToolConfig.enable_file_actions``).

Typical usage::

    from cube.tools.terminal import TerminalToolConfig

    # Bash-only (SWE-agent style):
    config = TerminalToolConfig(working_dir="/testbed")

    # With file helpers:
    config = TerminalToolConfig(
        working_dir="/testbed",
        enable_file_actions=True,
    )

    # Genny2 / mini-SWE-agent returncode envelope:
    config = TerminalToolConfig(
        working_dir="/testbed",
        output_format="returncode_envelope",
    )

The tool is created via ``config.make(container)`` and registered with a task
through ``task.tool_config = config``. ``make()`` returns a ``TerminalTool``
backed by a ``ContainerTerminalTool``.
"""

from __future__ import annotations

import logging
import re
import shlex
from abc import abstractmethod
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from cube.container import Container, ExecResult
from cube.core import ActionSchema
from cube.tool import Tool, ToolConfig, tool_action

logger = logging.getLogger(__name__)

NON_INTERACTIVE_ENV: dict[str, str] = {
    "PAGER": "cat",
    "MANPAGER": "cat",
    "LESS": "-R",
    "PIP_PROGRESS_BAR": "off",
    "TQDM_DISABLE": "1",
}


def _truncate_output(text: str, max_bytes: int, head_ratio: float = 0.2) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    head_bytes = max(0, int(max_bytes * head_ratio))
    tail_bytes = max_bytes - head_bytes
    head = encoded[:head_bytes].decode("utf-8", errors="ignore")
    tail = encoded[-tail_bytes:].decode("utf-8", errors="ignore") if tail_bytes > 0 else ""
    elided = len(encoded) - max_bytes
    return f"{head}\n[... {elided} bytes elided ...]\n{tail}"


def _format_exec_result(result: ExecResult, timeout: int) -> str:
    """Merge stdout+stderr and append plain-text exit-code annotations.

    Produces the observation string for ``output_format="plain"`` tools.  For
    ``"returncode_envelope"`` the caller merges output itself and wraps it in
    the ``<returncode>`` tag, so this helper is not used.
    """
    parts: list[str] = []
    if result.stdout:
        parts.append(result.stdout)
    if result.stderr:
        parts.append(result.stderr)
    if result.exit_code == 124:
        parts.append(f"[timed out after {timeout}s]")
    elif result.exit_code != 0:
        parts.append(f"[exit_code: {result.exit_code}]")
    return "\n".join(parts) if parts else "(no output)"


def _parse_line_arg(val: Any) -> int | None:
    """Parse a line-number argument that may arrive as int, str, or range notation.

    Handles LLM edge cases such as ``"[200, 210]"`` or ``"10-20"`` by returning
    the first number found. Returns ``None`` for ``None`` or unparseable input.
    """
    if val is None:
        return None
    if isinstance(val, int):
        return val
    s = str(val).strip().strip("[](){}")
    parts = re.split(r"[,\s\-:]+", s)
    try:
        return int(parts[0])
    except (ValueError, IndexError):
        return None


def _parse_line_range(line_start: Any, line_end: Any) -> tuple[int | None, int | None]:
    """Parse line_start / line_end, recovering range-style inputs.

    When a model passes ``line_start="[200, 210]"`` with no ``line_end``, both
    endpoints are extracted from that single value so the intent is preserved.
    """
    if isinstance(line_start, str) and line_end is None:
        s = line_start.strip().strip("[](){}")
        parts = re.split(r"[,\s\-:]+", s)
        if len(parts) >= 2:
            try:
                return int(parts[0]), int(parts[-1])
            except (ValueError, IndexError):
                pass
    return _parse_line_arg(line_start), _parse_line_arg(line_end)


class TerminalTool(Tool):
    """Abstract base for terminal-style tools.

    Captures only the task-side contract — methods that ``Task.reset()``,
    ``Task.evaluate()``, and ``Task._make_tool()`` need from any terminal
    implementation regardless of which concrete impl they got. Concrete
    implementations choose their own ``@tool_action`` surface; the abstract
    intentionally does NOT enumerate agent-facing actions.
    """

    @abstractmethod
    def bash_unlimited(self, command: str, timeout: int = 120) -> str:
        """Execute a shell command without output truncation.

        Not exposed as an agent action — used by tasks where the caller
        needs the full output (test runs, patch application, validation
        checks). Concrete implementations route this through their
        underlying transport.
        """
        ...


class TerminalToolConfig(ToolConfig):
    """Configuration for ``ContainerTerminalTool``.

    Args:
        working_dir: Default working directory inside the container.
        max_output_bytes: Maximum byte length of any returned observation.
            Output exceeding this limit is head+tail truncated.
        extra_env: Additional environment variables merged on top of
            ``NON_INTERACTIVE_ENV`` (pager / progress-bar suppression).
        output_format: ``"plain"`` appends ``[exit_code: N]`` to failed
            commands; ``"returncode_envelope"`` wraps output in
            ``<returncode>N</returncode>\\n<output>…</output>`` for agents
            (e.g. Genny2) that parse the tag.
        enable_file_actions: When ``True``, ``read_file()`` and
            ``write_file()`` are included in ``action_set`` in addition to
            ``bash()``.
        max_timeout: Optional ceiling (seconds) applied to every ``bash()``
            call. Prevents agents from requesting arbitrarily long timeouts.
            ``None`` means no cap.
    """

    working_dir: str = "/app"
    max_output_bytes: int = 100_000
    extra_env: dict[str, str] = {}
    output_format: Literal["plain", "returncode_envelope"] = "plain"
    enable_file_actions: bool = False
    max_timeout: int | None = None

    def make(self, container: Container | None = None) -> TerminalTool:
        if container is None:
            raise ValueError("ContainerTerminalTool requires a container")
        return ContainerTerminalTool(config=self, container=container)


class ContainerTerminalTool(TerminalTool):
    """Container-backed reference implementation of ``TerminalTool``.

    Always exposes ``bash()``.  Opt in to ``read_file()`` / ``write_file()``
    via ``TerminalToolConfig(enable_file_actions=True)``.

    The tool never splits commands on ``&&`` — multi-command strings are
    passed verbatim to the container's exec, which returns the exit code of
    the last executed command (standard shell semantics).  Avoid piping
    through ``tail`` or similar when the caller needs the exit code; truncate
    in Python via ``_truncate_output`` instead.
    """

    def __init__(self, config: TerminalToolConfig, container: Container) -> None:
        self._config = config
        self._container = container
        self._env = {**NON_INTERACTIVE_ENV, **config.extra_env}

    def reset(self) -> None:
        pass

    @property
    def action_set(self) -> list[ActionSchema]:
        actions = [ActionSchema.from_function(self.bash)]
        if self._config.enable_file_actions:
            actions += [
                ActionSchema.from_function(self.read_file),
                ActionSchema.from_function(self.write_file),
            ]
        return actions

    # auto-fix(177)↓
    def _resolved_path(self, path: str) -> str:
        """Absolute in-container path a relative *path* actually lands at.

        Every action ``exec``s with ``workdir=self._config.working_dir`` and
        shell state does not persist between calls, so a relative path always
        resolves against the static working dir — never the directory a prior
        ``bash`` ``cd``'d into. Reporting the resolved absolute path makes that
        (otherwise silent) mismatch immediately visible to the caller.
        """
        return str(PurePosixPath(self._config.working_dir) / path)

    # /auto-fix(177)

    def bash_unlimited(self, command: str, timeout: int = 120) -> str:
        """Like ``bash()`` but without output truncation — for internal harness use only.

        Not exposed as an agent action. Use this in ``evaluate()`` / ``apply_patch()``
        where the caller needs the full output and truncation would cause silent data loss.
        """
        result = self._container.exec(command, timeout=timeout, workdir=self._config.working_dir, env=self._env)
        return _format_exec_result(result, timeout)

    @tool_action
    def bash(self, command: str, timeout: int = 120) -> str:
        """Execute a bash command in the sandbox and return its output.

        Args:
            command: Shell command to run. The working directory is set by the
                task configuration. Environment changes do not persist between
                calls.
            timeout: Wall-clock seconds (not milliseconds). Default 120s. Use
                larger values (600–1800) for test suites or builds.
        """
        if self._config.max_timeout is not None:
            timeout = min(timeout, self._config.max_timeout)
        result = self._container.exec(
            command,
            timeout=timeout,
            workdir=self._config.working_dir,
            env=self._env,
        )
        if self._config.output_format == "returncode_envelope":
            merged = "\n".join(filter(None, [result.stdout, result.stderr])) or "(no output)"
            if result.exit_code == 124:
                merged = f"{merged}\n[timed out after {timeout}s]"
            body = _truncate_output(merged, self._config.max_output_bytes)
            return f"<returncode>{result.exit_code}</returncode>\n<output>\n{body}\n</output>"
        output = _format_exec_result(result, timeout)
        return _truncate_output(output, self._config.max_output_bytes)

    @tool_action
    def read_file(self, path: str, line_start: Any = None, line_end: Any = None) -> str:
        """Read the contents of a file in the sandbox.

        Args:
            path: Path to the file. Relative paths resolve against the
                working directory.
            line_start: First line to return (1-indexed, inclusive). Omit to
                read from the beginning. Also accepts range notation like
                ``[200, 210]`` when ``line_end`` is omitted.
            line_end: Last line to return (1-indexed, inclusive). Omit to
                read to the end.
        """
        parsed_start, parsed_end = _parse_line_range(line_start, line_end)
        if parsed_start is not None or parsed_end is not None:
            start = max(1, parsed_start if parsed_start is not None else 1)
            end = str(parsed_end) if parsed_end is not None else "$"
            cmd = f"sed -n '{start},{end}p' {shlex.quote(path)}"
            result = self._container.exec(cmd, workdir=self._config.working_dir)
            if result.exit_code != 0:
                return f"Error reading {self._resolved_path(path)}: {result.stderr or result.stdout}"
            return result.stdout
        result = self._container.exec(f"cat {shlex.quote(path)}", workdir=self._config.working_dir)
        if result.exit_code != 0:
            return f"Error reading {self._resolved_path(path)}: {result.stderr or result.stdout}"
        encoded = result.stdout.encode("utf-8")
        if len(encoded) > self._config.max_output_bytes:
            size_kb = len(encoded) // 1024
            return f"File too large ({size_kb} KB). Use line_start/line_end to read a specific range."
        return result.stdout

    @tool_action
    def write_file(self, path: str, content: str) -> str:
        """Write content to a file in the sandbox (overwrites any existing file).

        Args:
            path: Destination path. Parent directories are created as needed.
                Relative paths resolve against the working directory (NOT any
                directory a prior ``bash`` ``cd``'d into — shell state does not
                persist). The result echoes the absolute path written.
            content: Full file contents to write. This is not a patch tool —
                pass the entire new file body.
        """
        self._container.exec(
            f"mkdir -p {shlex.quote(str(Path(path).parent))}",
            workdir=self._config.working_dir,
        )
        escaped = content.replace("'", "'\\''")
        result = self._container.exec(
            f"printf '%s' '{escaped}' > {shlex.quote(path)}",
            workdir=self._config.working_dir,
        )
        resolved = self._resolved_path(path)
        if result.exit_code != 0:
            return f"Error writing {resolved}: {result.stderr or result.stdout}"
        return f"Wrote {len(content)} bytes to {resolved}"


# === auto-fix notes ===  (spec: openspec/specs/auto-fix/spec.md)
# auto-fix-note(177) {class=L1 issue=177 hash=PENDING ctx=any-infra/tbench2:fix-git/cube-standard@0e91ae1}
#   symptoms:  tbench2 fix-git; Genny did a relative write_file after a bash
#              `cd`. Every action execs at the static working_dir (shell
#              state never persists), so the file landed elsewhere and a
#              commit was corrupted -- silently. Tool-level; infra-agnostic.
#   invariant: a relative path reported by read_file/write_file is the
#              absolute path it actually resolved to (the static working_dir).
#   why:       lean anti-footgun (visible resolved path). The real fix -- a
#              stateful session working dir -- is a deferred contract change
#              (needs a cube-standard openspec proposal).
#   tested:    tests/ terminal resolved-path cases.
#   hash=PENDING: stamped by scripts/auto_fix_lint.py (Tier-1) on first run.
