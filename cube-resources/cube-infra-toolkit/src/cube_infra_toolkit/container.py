"""ToolkitContainer — ``cube.container.Container`` implementation backed by an EAI Toolkit job.

Canonical home for the Toolkit driver.  ``cube.backends.toolkit`` re-exports
this class for back-compat with the deprecated ``ToolkitContainerBackend``
factory; new code should use ``ToolkitInfraConfig`` from this package.

Upstream EAI Toolkit documentation: https://docs.console.elementai.com/
"""

from __future__ import annotations

import base64
import json
import logging
import os
import signal
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict

from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)

from cube.container import (
    Container,
    ContainerExecError,
    ContainerLaunchError,
    ContainerStatus,
    ExecResult,
)

_EXEC_RELAY_SERVER_PATH = Path(__file__).parent / "_exec_relay_server.py"
_EXEC_RELAY_PORT = 8787
_EXEC_RELAY_KICK_TIMEOUT = 15  # eai exec calls during bootstrap; short so CLOSE_WAIT fails fast
_EXEC_RELAY_HEALTH_TIMEOUT = 15
# Path inside the container where the sidecar binary lives once
# ``ToolkitInfraConfig.cube_data`` is mounted at ``/opt/cube:ro``.
_SIDECAR_MOUNT_PATH = "/opt/cube/cube-sidecar"


class ExecRelayUnavailable(RuntimeError):
    """Raised when the exec relay cannot be reached.

    Distinguishes "the relay never started" (image lacked both the cube-sidecar
    mount and python3) from a transient network failure — both are unrecoverable
    here since direct ``eai job exec`` is no longer a fallback.  Surface this
    early with a clear message instead of hanging on health probes.
    """


logger = logging.getLogger(__name__)

_retry_io = retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)


def _run_eai(
    args: list[str],
    *,
    eai_path: str = "eai",
    profile: str | None = None,
    account: str | None = None,
    timeout: int | None = 60,
    retries: int = 2,
    input: bytes | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run an ``eai`` CLI command with process-group cleanup + retry-on-hang.

    Empirically, ``eai job exec`` has a small but non-zero per-invocation hang
    rate — a fresh process retry almost always succeeds.  Strategy:

    1. Put ``eai`` in its own process group (``start_new_session=True``) so
       ``killpg`` on timeout tears down any forked children.  Without this,
       ``(eai)`` zombies accumulate when the CLI's internal loop gets stuck.
    2. On ``TimeoutExpired``, kill the process group and retry up to
       ``retries`` times.  Each retry starts a fresh ``eai`` subprocess, which
       virtually eliminates the stuck-state carryover.

    Only ``TimeoutExpired`` triggers a retry.  Non-zero exit codes pass through
    unchanged — those indicate real failures (bad args, 404, auth) where
    retrying is a waste.
    """
    cmd = [eai_path]
    if profile:
        cmd += ["--profile", profile]
    if account:
        cmd += ["--account", account]
    cmd += args

    last_err: subprocess.TimeoutExpired | None = None
    for attempt in range(retries + 1):
        logger.debug("Running (attempt %d/%d): %s", attempt + 1, retries + 1, " ".join(cmd))
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise ContainerLaunchError(
                f"The 'eai' CLI tool was not found at {cmd[0]!r}. Install it or set ToolkitInfraConfig(eai_path=...)."
            ) from exc

        try:
            raw_out, raw_err = proc.communicate(input=input, timeout=timeout)
            stdout = raw_out.decode("utf-8", errors="replace")
            stderr = raw_err.decode("utf-8", errors="replace")
        except subprocess.TimeoutExpired as exc:
            last_err = exc
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                proc.communicate(timeout=5)
            except Exception:
                pass
            if attempt < retries:
                backoff = 5 * (attempt + 1)
                logger.warning(
                    "eai command timed out after %ds (attempt %d/%d); backing off %ds before retry: %s",
                    timeout,
                    attempt + 1,
                    retries + 1,
                    backoff,
                    " ".join(cmd),
                )
                time.sleep(backoff)
                continue
            raise ContainerExecError(
                f"eai command timed out after {timeout}s (tried {retries + 1} times): {' '.join(cmd)}"
            ) from exc

        return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)

    assert last_err is not None
    raise ContainerExecError("eai: unreachable branch") from last_err


def _find_free_port() -> int:
    """Find a free local port by binding to port 0."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def relay_startup_args(token: str) -> list[str]:
    """Return the ``['/bin/sh', '-c', <script>]`` args for a job launch command.

    Prefers the Go sidecar binary if mounted at ``_SIDECAR_MOUNT_PATH`` (works on
    any image regardless of python3).  Falls back to the Python relay if python3 is
    available.  Either way the relay is up before the first port-forward — zero
    bootstrap ``eai job exec`` calls.

    Security note: the base64-encoded token appears in the one-shot startup shell's
    argv.  The long-lived relay server reads the token from a file, never its argv.
    """
    script_b64 = base64.b64encode(_EXEC_RELAY_SERVER_PATH.read_bytes()).decode()
    token_b64 = base64.b64encode(token.encode()).decode()
    startup = "\n".join(
        [
            f"printf '%s' '{token_b64}' | base64 -d > /tmp/.cube_relay_token",
            "chmod 600 /tmp/.cube_relay_token",
            f"if [ -f {_SIDECAR_MOUNT_PATH} ]; then",
            f"  cp {_SIDECAR_MOUNT_PATH} /tmp/cube-sidecar && chmod +x /tmp/cube-sidecar",
            f"  CUBE_EXEC_RELAY_PORT={_EXEC_RELAY_PORT} CUBE_EXEC_RELAY_TOKEN_FILE=/tmp/.cube_relay_token"
            " nohup /tmp/cube-sidecar </dev/null >/tmp/_cube_sidecar.log 2>&1 &",
            "elif command -v python3 >/dev/null 2>&1; then",
            f"  printf '%s' '{script_b64}' | base64 -d > /tmp/_cube_exec_relay.py",
            "  chmod 600 /tmp/_cube_exec_relay.py",
            f"  CUBE_EXEC_RELAY_PORT={_EXEC_RELAY_PORT} CUBE_EXEC_RELAY_TOKEN_FILE=/tmp/.cube_relay_token"
            " nohup python3 /tmp/_cube_exec_relay.py </dev/null >/tmp/_cube_exec_relay.log 2>&1 &",
            "fi",
            "exec sleep infinity",
        ]
    )
    return ["/bin/sh", "-c", startup]


class ToolkitContainer(Container):
    """Runtime handle backed by an EAI Toolkit job.

    Exec routing: a small HTTP server (the exec relay) runs inside the container,
    started as part of the job's launch command and tunneled via
    ``eai job port-forward``.  All ``.exec()`` calls go through the relay,
    bypassing the ``eai job exec`` CLOSE_WAIT hang bug entirely.

    The relay starts in one of two ways at job launch (see ``relay_startup_args``):
      1. Pre-built Go binary mounted at ``/opt/cube/cube-sidecar`` via
         ``ToolkitInfraConfig.cube_data``.  Works on any image (no python3).
      2. Python relay script embedded in the startup command, if the image has
         ``python3`` on PATH.

    If neither path fires the relay never comes up; ``.exec()`` raises
    ``ExecRelayUnavailable`` at first call.  There is no ``eai job exec``
    fallback — that path was deleted along with its CLOSE_WAIT retry machinery.
    """

    def __init__(
        self,
        job_id: str,
        profile: str | None = None,
        account: str | None = None,
        eai_path: str = "eai",
        relay_prestarted_token: str | None = None,
    ) -> None:
        super().__init__()  # populates ResourceHandle fields with defaults
        self._job_id = job_id
        self._profile = profile
        self._account = account
        self._eai_path = eai_path
        self._port_forwards: dict[int, subprocess.Popen] = {}
        self._port_map: dict[int, int] = {}
        self._port_forward_logs: dict[int, str] = {}
        # Token is set by the launcher when it embeds the relay startup script.
        self._relay_token: str | None = relay_prestarted_token
        self._relay_local_port: int | None = None
        self._relay_ready = False

    @property
    def id(self) -> str:
        return self._job_id

    def __repr__(self) -> str:
        return f"ToolkitContainer(job_id={self._job_id!r}, profile={self._profile!r}, run_id={self.run_id!r})"

    # ------------------------- exec relay bootstrap -------------------------

    def _bootstrap_exec_relay(self) -> None:
        """Port-forward + health-check the exec relay started at job launch.

        Single path: the relay was started by the job's startup command (either
        the cube-sidecar Go binary mounted at ``/opt/cube/cube-sidecar``
        or, on images with python3, the embedded relay script — see
        ``relay_startup_args``).  Open the tunnel, health-check, done.

        If the health probe fails the relay never came up — the image lacked
        both the sidecar mount and python3.  Raise so the caller sees a clean
        configuration error instead of subsequent exec calls hanging.
        """
        if self._relay_ready:
            return
        if self._relay_token is None:
            raise ExecRelayUnavailable(
                f"Container for job {self._job_id[:8]} was launched without an exec-relay "
                "prestart token. This indicates a programming error in the launcher — "
                "see ToolkitInfraConfig.launch()."
            )

        local_port = self.forward_port(_EXEC_RELAY_PORT)
        if self._probe_health(local_port, timeout=_EXEC_RELAY_HEALTH_TIMEOUT):
            self._relay_local_port = local_port
            self._relay_ready = True
            logger.info("Exec relay ready for job %s", self._job_id[:8])
            return

        diag = self._fetch_relay_diagnostics()
        raise ExecRelayUnavailable(
            f"Exec relay never came up in job {self._job_id[:8]}. The image likely "
            "lacks both /opt/cube/cube-sidecar (set ToolkitInfraConfig."
            "cube_data, or rely on the 'auto' default) and python3 on PATH. "
            "Diagnostics:\n" + diag
        )

    def _probe_health(self, local_port: int, *, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{local_port}/health", timeout=1.0) as r:
                    if r.status == 200:
                        return True
            except (urllib.error.URLError, ConnectionError, OSError):
                time.sleep(0.2)
        return False

    def _fetch_relay_diagnostics(self) -> str:
        parts = []
        try:
            r = _run_eai(
                [
                    "job",
                    "exec",
                    self._job_id,
                    "--",
                    "bash",
                    "-c",
                    "tail -20 /tmp/_cube_exec_relay.log 2>/dev/null; "
                    "echo ---ps---; ps -ef | grep _cube_exec_relay | grep -vw grep; "
                    "echo ---curl---; curl -sS --max-time 2 http://127.0.0.1:8787/health 2>&1 | head -5",
                ],
                eai_path=self._eai_path,
                profile=self._profile,
                account=self._account,
                timeout=30,
                retries=0,
            )
            parts.append(r.stdout.strip() or "<empty>")
        except Exception as exc:
            parts.append(f"<diagnostics fetch failed: {exc}>")

        log_path = self._port_forward_logs.get(_EXEC_RELAY_PORT)
        if log_path:
            try:
                with open(log_path) as f:
                    pf_err = f.read().strip()
            except Exception as exc:
                pf_err = f"<read failed: {exc}>"
            proc = self._port_forwards.get(_EXEC_RELAY_PORT)
            pf_alive = proc is not None and proc.poll() is None
            parts.append(f"---port-forward alive={pf_alive}---\n{pf_err or '<empty>'}")

        return "\n".join(parts)

    # ------------------------- exec -------------------------

    def exec(
        self,
        command: str,
        timeout: int | None = None,
        workdir: str | None = None,
        env: Dict[str, str] | None = None,
    ) -> ExecResult:
        effective_timeout = timeout if timeout is not None else 120
        if not self._relay_ready:
            self._bootstrap_exec_relay()
        return self._exec_via_relay(command, effective_timeout, workdir, env)

    def _exec_via_relay(
        self,
        command: str,
        timeout: int,
        workdir: str | None,
        env: Dict[str, str] | None,
    ) -> ExecResult:
        assert self._relay_ready and self._relay_token and self._relay_local_port
        # Clamp to relay server's 24-hour hard limit (86400s) so we never get a
        # bad_timeout 400. The real cap is enforced by the caller (e.g. bash tool).
        payload: dict = {"command": command, "timeout": min(timeout, 86399)}
        if workdir:
            payload["workdir"] = workdir
        if env:
            payload["env"] = env
        data = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            f"http://127.0.0.1:{self._relay_local_port}/exec",
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._relay_token}",
            },
        )
        logger.info("exec [%s] (relay): %s", self._job_id[:8], command)
        start = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=timeout + 30) as r:
                body = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = ""
            try:
                error_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            logger.warning(
                "exec [%s] (relay): HTTP %d — body: %s",
                self._job_id[:8],
                exc.code,
                error_body or "<empty>",
            )
            raise ExecRelayUnavailable(f"HTTP {exc.code}: {exc.reason} — {error_body}") from exc
        except (urllib.error.URLError, ConnectionError, OSError, json.JSONDecodeError) as exc:
            raise ExecRelayUnavailable(str(exc)) from exc

        duration = time.monotonic() - start
        logger.info(
            "exec [%s] (relay): done in %.1fs, exit_code=%s",
            self._job_id[:8],
            duration,
            body.get("exit_code"),
        )
        return ExecResult(
            stdout=body.get("stdout", "").strip(),
            stderr=body.get("stderr", "").strip(),
            exit_code=int(body.get("exit_code", 1)),
            duration_seconds=round(float(body.get("duration_seconds", duration)), 3),
        )

    def forward_port(self, container_port: int) -> int:
        if container_port in self._port_map:
            return self._port_map[container_port]

        local_port = _find_free_port()

        cmd = [self._eai_path]
        if self._profile:
            cmd += ["--profile", self._profile]
        if self._account:
            cmd += ["--account", self._account]
        cmd += [
            "job",
            "port-forward",
            self._job_id,
            f"{local_port}:{container_port}",
        ]

        logger.info(
            "Starting port-forward %d->%d for job %s",
            local_port,
            container_port,
            self._job_id,
        )
        stderr_file = tempfile.NamedTemporaryFile(
            prefix=f"eai-pf-{container_port}-", suffix=".log", delete=False, mode="w"
        )
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=stderr_file,
        )

        time.sleep(2)
        if proc.poll() is not None:
            stderr_file.close()
            try:
                with open(stderr_file.name) as f:
                    err_text = f.read().strip()
            except Exception:
                err_text = "<unavailable>"
            raise ContainerExecError(
                f"Port-forward process exited immediately (rc={proc.returncode}) "
                f"for port {container_port}. stderr: {err_text}"
            )
        logger.debug("Port-forward stderr log at %s", stderr_file.name)
        self._port_forward_logs[container_port] = stderr_file.name

        self._port_forwards[container_port] = proc
        self._port_map[container_port] = local_port
        return local_port

    def get_url(self, container_port: int) -> str:
        host_port = self.forward_port(container_port)
        return f"http://localhost:{host_port}"

    @_retry_io
    def stop(self, timeout: int = 10) -> None:
        for port, proc in self._port_forwards.items():
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
                logger.debug("Force-killed port-forward for port %d", port)
        self._port_forwards.clear()
        self._port_map.clear()

        logger.info("stop [%s]: killing job", self._job_id[:8])
        try:
            _run_eai(
                ["job", "kill", self._job_id],
                eai_path=self._eai_path,
                profile=self._profile,
                account=self._account,
                timeout=30,
            )
        except Exception as exc:
            logger.warning("Error killing job %s: %s", self._job_id, exc)

    def get_status(self) -> ContainerStatus:
        try:
            result = _run_eai(
                ["job", "get", self._job_id, "--format", "json", "--no-header"],
                eai_path=self._eai_path,
                profile=self._profile,
                account=self._account,
                timeout=30,
            )
            info = json.loads(result.stdout)
            state = info.get("state", "unknown").lower()
            running = state == "running"
            return ContainerStatus(
                running=running,
                healthy=running,
                backend_info={"eai_state": state, "id": self._job_id},
            )
        except Exception:
            return ContainerStatus(
                running=False,
                healthy=False,
                backend_info={"eai_state": "unknown", "id": self._job_id},
            )
