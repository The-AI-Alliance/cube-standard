"""DaytonaContainer — ``cube.container.Container`` implementation backed by a Daytona sandbox.

Canonical home for the Daytona driver.  ``DaytonaInfraConfig`` (this package)
produces these via ``launch()``.
"""

from __future__ import annotations

import logging
import shlex
import time
from typing import Any, Dict
from uuid import uuid4

from daytona import (
    Daytona,
    DaytonaError,
    SessionExecuteRequest,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from cube.container import (
    Container,
    ContainerError,
    ContainerExecError,
    ContainerStatus,
    ExecResult,
    HealthCheckError,
    port_from_url,
)

logger = logging.getLogger(__name__)

retry_sandbox = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
    retry=retry_if_not_exception_type(HealthCheckError),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)

_retry_io = retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)

# Async-exec poll tuning. The submit returns a cmd_id immediately, so its HTTP read
# is short; the long wait is decomposed into these short, idempotent polls.
_SUBMIT_TIMEOUT_S = 60
_POLL_INTERVAL_S = 2.0
_POLL_GRACE_S = 15.0


class DaytonaContainer(Container):
    """Runtime handle backed by a Daytona sandbox."""

    def __init__(
        self,
        sandbox: Any,
        client: Daytona,
        allowed_ports: set[int] | None = None,
    ) -> None:
        super().__init__()  # populates ResourceHandle fields with defaults
        self._sandbox = sandbox
        self._client = client
        self._url_cache: dict[int, str] = {}
        self._allowed_ports = allowed_ports
        self._session_id = str(uuid4())
        self._sandbox.process.create_session(self._session_id)

    @property
    def id(self) -> str:
        return self._sandbox.id

    def __repr__(self) -> str:
        return f"DaytonaContainer(id={self.id!r}, run_id={self.run_id!r})"

    def exec(
        self,
        command: str,
        timeout: int | None = None,
        workdir: str | None = None,
        env: Dict[str, str] | None = None,
    ) -> ExecResult:
        effective_timeout = timeout if timeout is not None else 120

        parts = []
        if env:
            for k, v in env.items():
                parts.append(f"export {k}={shlex.quote(v)}")
        if workdir:
            parts.append(f"cd {shlex.quote(workdir)}")
        parts.append(command)
        full_command = " && ".join(parts)

        wrapped = f"timeout {effective_timeout}s bash -lc {shlex.quote(full_command)}"

        start = time.monotonic()
        try:
            # auto-fix(207)↓
            # Submit asynchronously and poll, instead of one long synchronous read.
            # `run_async=False` held a single HTTP read open for the whole command, so a
            # transient `proxy.app.daytona.io` ReadTimeout (~135s) raised ContainerExecError
            # and crashed the whole (often many-step) episode — unretryable, because by then
            # the command had already started. With `run_async=True` the long wait becomes
            # short, *idempotent* polls (`_get_command`/`_command_logs`, each @_retry_io), so a
            # transient blip retries safely; the command's real bound stays the in-sandbox
            # `timeout Ns` wrapper. Logs come back already split into stdout/stderr.
            submit = self._sandbox.process.execute_session_command(
                self._session_id,
                SessionExecuteRequest(command=wrapped, run_async=True),
                timeout=_SUBMIT_TIMEOUT_S,
            )
            cmd_id = submit.cmd_id
            if cmd_id is None:
                raise ContainerExecError("Daytona async submit returned no cmd_id")

            exit_code = submit.exit_code
            deadline = start + effective_timeout + _POLL_GRACE_S
            while exit_code is None and time.monotonic() < deadline:
                time.sleep(_POLL_INTERVAL_S)
                exit_code = self._get_command(cmd_id).exit_code

            logs = self._command_logs(cmd_id)
            duration = time.monotonic() - start
            # exit_code still None ⇒ our poll deadline beat the in-sandbox `timeout`
            # wrapper; report it as a timeout (124) rather than a spurious success.
            return ExecResult(
                stdout=(logs.stdout or "").strip(),
                stderr=(logs.stderr or "").strip(),
                exit_code=exit_code if exit_code is not None else 124,
                duration_seconds=round(duration, 3),
            )
            # /auto-fix(207)
        except ContainerExecError:
            raise
        except Exception as exc:
            raise ContainerExecError(f"Daytona exec failed: {exc}") from exc

    @_retry_io
    def _get_command(self, cmd_id: str) -> Any:
        """Poll a session command's status. Idempotent → safe to retry on transient transport errors."""
        return self._sandbox.process.get_session_command(self._session_id, cmd_id)

    @_retry_io
    def _command_logs(self, cmd_id: str) -> Any:
        """Fetch a session command's stdout/stderr. Idempotent → safe to retry."""
        return self._sandbox.process.get_session_command_logs(self._session_id, cmd_id)

    def forward_port(self, container_port: int) -> int:
        return port_from_url(self.get_url(container_port))

    def get_url(self, container_port: int) -> str:
        self._assert_allowed_port(container_port)
        if container_port in self._url_cache:
            return self._url_cache[container_port]

        try:
            preview = self._sandbox.create_signed_preview_url(container_port, expires_in_seconds=86400)
            url = preview.url
            self._url_cache[container_port] = url
            return url
        except Exception as exc:
            raise ContainerError(f"Failed to get URL for port {container_port}: {exc}") from exc

    @_retry_io
    def stop(self, timeout: int = 10) -> None:
        try:
            self._sandbox.process.delete_session(self._session_id)
        except Exception:
            pass
        try:
            self._client.delete(self._sandbox)
        except DaytonaError:
            pass
        except Exception as exc:
            logger.warning("Error deleting sandbox %s: %s", self.id, exc)

    def get_status(self) -> ContainerStatus:
        try:
            sandbox = self._client.get(self._sandbox.id)
            state = str(sandbox.state) if sandbox.state else "unknown"
            running = "started" in state.lower() or "running" in state.lower()
            return ContainerStatus(
                running=running,
                healthy=running,
                backend_info={"daytona_state": state, "id": self.id},
            )
        except Exception:
            return ContainerStatus(
                running=False,
                healthy=False,
                backend_info={"daytona_state": "unknown", "id": self.id},
            )

    def _assert_allowed_port(self, container_port: int) -> None:
        if self._allowed_ports is None:
            return
        if container_port in self._allowed_ports:
            return
        raise ContainerError(f"Port {container_port} is not in declared spec ports: {sorted(self._allowed_ports)}")


# === auto-fix notes ===  (spec: openspec/specs/auto-fix/spec.md)
# auto-fix-note(207) {class=L1 anchor=PR#207 hash=PENDING ctx=daytona/exec-sync-read-timeout-crashed-episode/async-submit+idempotent-poll-retry/tbench2:kv-store-grpc+mailman+rstan-to-pystan/cube-standard@a9d98a7}
