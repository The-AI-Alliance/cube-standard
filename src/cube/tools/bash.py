"""Concrete bash tool for container-backed sandbox execution.

``BashTool`` exposes a ``bash()`` action and, optionally, ``read_file()`` and
``write_file()`` when ``enable_file_actions=True``.  All execution is delegated
to a ``cube.container.Container``.

Typical usage::

    from cube.tools.bash import BashToolConfig

    # Bash-only (SWE-agent style):
    config = BashToolConfig(working_dir="/testbed")

    # With file helpers:
    config = BashToolConfig(
        working_dir="/testbed",
        enable_file_actions=True,
    )

    # Genny2 / mini-SWE-agent returncode envelope:
    config = BashToolConfig(
        working_dir="/testbed",
        output_format="returncode_envelope",
    )

The tool is created via ``config.make(container)`` and registered with a task
through ``task.tool_config = config``.
"""

from __future__ import annotations

import logging
import shlex
from pathlib import Path
from typing import Literal

from cube.container import Container, ExecResult
from cube.core import ActionSchema
from cube.tool import Tool, ToolConfig, tool_action

logger = logging.getLogger(__name__)


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


class BashToolConfig(ToolConfig):
    """Configuration for ``BashTool``.

    Args:
        working_dir: Default working directory inside the container.
        max_output_bytes: Maximum byte length of any returned observation.
            Output exceeding this limit is head+tail truncated.
        extra_env: Additional environment variables merged on top of
            ``Tool.NON_INTERACTIVE_ENV`` (pager / progress-bar suppression).
        output_format: ``"plain"`` appends ``[exit_code: N]`` to failed
            commands; ``"returncode_envelope"`` wraps output in
            ``<returncode>N</returncode>\\n<output>…</output>`` for agents
            (e.g. Genny2) that parse the tag.
        enable_file_actions: When ``True``, ``read_file()`` and
            ``write_file()`` are included in ``action_set`` in addition to
            ``bash()``.
    """

    working_dir: str = "/app"
    max_output_bytes: int = 100_000
    extra_env: dict[str, str] = {}
    output_format: Literal["plain", "returncode_envelope"] = "plain"
    enable_file_actions: bool = False

    def make(self, container: Container | None = None) -> "BashTool":
        if container is None:
            raise ValueError("BashTool requires a container")
        return BashTool(config=self, container=container)


class BashTool(Tool):
    """Container-backed bash tool.

    Always exposes ``bash()``.  Opt in to ``read_file()`` / ``write_file()``
    via ``BashToolConfig(enable_file_actions=True)``.

    The tool never splits commands on ``&&`` — multi-command strings are
    passed verbatim to the container's exec, which returns the exit code of
    the last executed command (standard shell semantics).  Avoid piping
    through ``tail`` or similar when the caller needs the exit code; truncate
    in Python via ``_truncate_output`` instead.
    """

    def __init__(self, config: BashToolConfig, container: Container) -> None:
        self._config = config
        self._container = container
        self._env = {**Tool.NON_INTERACTIVE_ENV, **config.extra_env}

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
            body = Tool._truncate_output(merged, self._config.max_output_bytes)
            return f"<returncode>{result.exit_code}</returncode>\n<output>\n{body}\n</output>"
        output = _format_exec_result(result, timeout)
        return Tool._truncate_output(output, self._config.max_output_bytes)

    @tool_action
    def read_file(self, path: str, line_start: int | None = None, line_end: int | None = None) -> str:
        """Read the contents of a file in the sandbox.

        Args:
            path: Path to the file. Relative paths resolve against the
                working directory.
            line_start: First line to return (1-indexed, inclusive). Omit to
                read from the beginning.
            line_end: Last line to return (1-indexed, inclusive). Omit to
                read to the end.
        """
        if line_start is not None or line_end is not None:
            start = max(1, line_start if line_start is not None else 1)
            end = str(line_end) if line_end is not None else "$"
            cmd = f"sed -n '{start},{end}p' {shlex.quote(path)}"
            result = self._container.exec(cmd, workdir=self._config.working_dir)
            if result.exit_code != 0:
                return f"Error reading {path}: {result.stderr or result.stdout}"
            return result.stdout
        result = self._container.exec(f"cat {shlex.quote(path)}", workdir=self._config.working_dir)
        if result.exit_code != 0:
            return f"Error reading {path}: {result.stderr or result.stdout}"
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
                Relative paths resolve against the working directory.
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
        if result.exit_code != 0:
            return f"Error writing {path}: {result.stderr or result.stdout}"
        return f"Wrote {len(content)} bytes to {path}"
