"""ToolkitContainer — ``cube.container.Container`` implementation backed by an EAI Toolkit job.

Canonical home for the Toolkit driver.  ``cube.backends.toolkit`` re-exports
this class for back-compat with the deprecated ``ToolkitContainerBackend``
factory; new code should use ``ToolkitInfraConfig`` from this package.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import secrets
import shlex
import signal
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Literal

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


class _ExecRelayUnavailable(Exception):
    """Raised when the exec relay is unreachable for a single call; triggers fallback to direct exec."""


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

    Embeds the exec relay script + token directly in the job's startup command so
    the relay is running before the first port-forward — eliminating all bootstrap
    ``eai job exec`` calls for images that already have python3.
    """
    script_b64 = base64.b64encode(_EXEC_RELAY_SERVER_PATH.read_bytes()).decode()
    token_b64 = base64.b64encode(token.encode()).decode()
    startup = (
        f"printf '%s' '{token_b64}' | base64 -d > /tmp/.cube_exec_relay_token && "
        f"printf '%s' '{script_b64}' | base64 -d > /tmp/_cube_exec_relay.py && "
        "chmod 600 /tmp/.cube_exec_relay_token /tmp/_cube_exec_relay.py && "
        f"CUBE_EXEC_RELAY_PORT={_EXEC_RELAY_PORT} "
        "CUBE_EXEC_RELAY_TOKEN_FILE=/tmp/.cube_exec_relay_token "
        "nohup python3 /tmp/_cube_exec_relay.py "
        "</dev/null >/tmp/_cube_exec_relay.log 2>&1 & "
        "exec sleep infinity"
    )
    return ["/bin/sh", "-c", startup]


class ToolkitContainer(Container):
    """Runtime handle backed by an EAI Toolkit job.

    Exec routing: a small HTTP server (exec relay) runs inside the container,
    started as part of the job's launch command and tunneled via
    ``eai job port-forward``.  All ``.exec()`` calls go through the relay,
    bypassing the ``eai job exec`` CLOSE_WAIT hang bug entirely.

    Fallback chain when the relay is unavailable:
      1. Fast path: relay pre-started at job launch (relay_prestarted_token set).
         Only a health check is needed — no eai exec for bootstrap.
      2. Slow path: relay not pre-started or python3 was absent at launch.
         Bootstrap via eai exec (install python3 if needed, kick the relay).
      3. Direct exec: relay bootstrap failed entirely; fall back to eai job exec
         with the CLOSE_WAIT sentinel + retry logic.
    """

    def __init__(
        self,
        job_id: str,
        profile: str | None = None,
        account: str | None = None,
        exec_mode: Literal["exec_relay", "direct"] = "exec_relay",
        eai_path: str = "eai",
        relay_prestarted_token: str | None = None,
    ) -> None:
        super().__init__()  # populates ResourceHandle fields with defaults
        self._job_id = job_id
        self._profile = profile
        self._account = account
        self._exec_mode = exec_mode
        self._eai_path = eai_path
        self._port_forwards: dict[int, subprocess.Popen] = {}
        self._port_map: dict[int, int] = {}
        self._port_forward_logs: dict[int, str] = {}
        # Token is set either at construction (pre-started) or after bootstrap.
        self._relay_token: str | None = relay_prestarted_token
        self._relay_local_port: int | None = None
        self._relay_ready = False

    @property
    def id(self) -> str:
        return self._job_id

    # ------------------------- exec relay bootstrap -------------------------

    def _ensure_python3(self) -> None:
        """Install python3 via apt if the image doesn't have it (slow path only)."""
        check = _run_eai(
            [
                "job",
                "exec",
                self._job_id,
                "--",
                "bash",
                "-c",
                "python3 --version 2>/dev/null && echo HAS_PYTHON || echo NO_PYTHON",
            ],
            eai_path=self._eai_path,
            profile=self._profile,
            account=self._account,
            timeout=15,
            retries=1,
        )
        if "NO_PYTHON" not in check.stdout:
            return
        logger.info("python3 missing in job %s — installing via apt-get", self._job_id[:8])
        _run_eai(
            [
                "job",
                "exec",
                self._job_id,
                "--",
                "bash",
                "-c",
                "apt-get update -qq && apt-get install -y --no-install-recommends python3 python3-pip 2>&1",
            ],
            eai_path=self._eai_path,
            profile=self._profile,
            account=self._account,
            timeout=120,
            retries=1,
        )

    def _kick_relay_python(self, token: str) -> None:
        """Upload the Python relay script + token, kill any prior instance, start detached."""
        server_src = _EXEC_RELAY_SERVER_PATH.read_text()
        server_b64 = base64.b64encode(server_src.encode()).decode()
        token_b64 = base64.b64encode(token.encode()).decode()
        script = (
            "umask 077\n"
            "pgrep -f _cube_exec_relay.py 2>/dev/null "
            '| grep -vw "$$" '
            "| xargs -r kill 2>/dev/null || true\n"
            "sleep 0.3\n"
            f"printf '%s' '{server_b64}' | base64 -d > /tmp/_cube_exec_relay.py\n"
            f"printf '%s' '{token_b64}' | base64 -d > /tmp/.cube_exec_relay_token\n"
            "chmod 600 /tmp/.cube_exec_relay_token /tmp/_cube_exec_relay.py\n"
            f"export CUBE_EXEC_RELAY_PORT={_EXEC_RELAY_PORT}\n"
            "export CUBE_EXEC_RELAY_TOKEN_FILE=/tmp/.cube_exec_relay_token\n"
            "PYTHON3=$(command -v python3 || command -v python)\n"
            'echo "PYTHON3=$PYTHON3"\n'
            "[ -z \"$PYTHON3\" ] && echo 'ERROR: no python3 found' && exit 1\n"
            'nohup "$PYTHON3" /tmp/_cube_exec_relay.py '
            "</dev/null >/tmp/_cube_exec_relay.log 2>&1 &\n"
            "echo KICKED\n"
        )
        logger.info("Bootstrapping exec relay in job %s …", self._job_id[:8])
        # Retry the kick: CLOSE_WAIT hangs ~6% of exec calls; short timeout + retries
        # reduces P(all attempts hang) to near-zero without adding much latency.
        for kick_attempt in range(4):
            try:
                result = _run_eai(
                    ["job", "exec", self._job_id, "--", "bash", "-c", script],
                    eai_path=self._eai_path,
                    profile=self._profile,
                    account=self._account,
                    timeout=_EXEC_RELAY_KICK_TIMEOUT,
                    retries=0,
                )
                logger.info("Exec relay kick output for job %s: %s", self._job_id[:8], result.stdout.strip())
                return  # kick sent; health probe decides if it worked
            except ContainerExecError:
                if kick_attempt < 3:
                    logger.warning(
                        "Exec relay kick timed out for job %s (CLOSE_WAIT, attempt %d/4); retrying",
                        self._job_id[:8],
                        kick_attempt + 1,
                    )
                else:
                    logger.warning("Exec relay kick failed after 4 attempts for job %s", self._job_id[:8])

    def _bootstrap_exec_relay(self) -> None:
        """Port-forward + health-check the exec relay; kick via eai exec if needed.

        Fast path (relay_prestarted_token set): relay was started as part of the
        job launch command — just open the tunnel and health-check.  Zero eai execs.

        Slow path (no token, or health check fails): install python3 if needed,
        kick the relay via eai exec, health-check again.
        """
        if self._relay_ready or self._exec_mode == "direct":
            return

        local_port = self.forward_port(_EXEC_RELAY_PORT)

        # Fast path: relay was pre-started at job launch.
        if self._relay_token is not None:
            if self._probe_health(local_port, timeout=_EXEC_RELAY_HEALTH_TIMEOUT):
                self._relay_local_port = local_port
                self._relay_ready = True
                logger.info("Exec relay ready (pre-started) for job %s", self._job_id[:8])
                return
            logger.info(
                "Pre-started relay not healthy for job %s — python3 likely absent; bootstrapping via eai exec",
                self._job_id[:8],
            )

        # Slow path: need to kick the relay manually via eai exec.
        token = self._relay_token or secrets.token_urlsafe(32)
        self._ensure_python3()
        last_diag = ""
        for attempt in range(2):
            self._kick_relay_python(token)
            if self._probe_health(local_port, timeout=_EXEC_RELAY_HEALTH_TIMEOUT):
                self._relay_token = token
                self._relay_local_port = local_port
                self._relay_ready = True
                logger.info("Exec relay ready (bootstrapped) for job %s (attempt %d)", self._job_id[:8], attempt + 1)
                return
            last_diag = self._fetch_relay_diagnostics()
            logger.warning(
                "Exec relay health failed on attempt %d/2 for job %s. Diag:\n%s",
                attempt + 1,
                self._job_id[:8],
                last_diag,
            )

        logger.warning(
            "Exec relay bootstrap failed for job %s — falling back to direct exec. Last diag:\n%s",
            self._job_id[:8],
            last_diag,
        )
        self._exec_mode = "direct"

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

        if self._exec_mode == "exec_relay":
            try:
                if not self._relay_ready:
                    self._bootstrap_exec_relay()
                # _bootstrap_exec_relay may have silently fallen back to direct
                # mode (e.g. image has no python3); re-check before dispatching.
                if self._relay_ready:
                    return self._exec_via_relay(command, effective_timeout, workdir, env)
            except _ExecRelayUnavailable as exc:
                logger.warning(
                    "Exec relay failed for job %s, falling back to direct eai exec: %s",
                    self._job_id[:8],
                    exc,
                )
                self._relay_ready = False
                self._exec_mode = "direct"

        return self._exec_direct(command, effective_timeout, workdir, env)

    def _exec_via_relay(
        self,
        command: str,
        timeout: int,
        workdir: str | None,
        env: Dict[str, str] | None,
    ) -> ExecResult:
        assert self._relay_ready and self._relay_token and self._relay_local_port
        payload: dict = {"command": command, "timeout": timeout}
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
        except (urllib.error.URLError, ConnectionError, OSError, json.JSONDecodeError) as exc:
            raise _ExecRelayUnavailable(str(exc)) from exc

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

    def _exec_direct(
        self,
        command: str,
        effective_timeout: int,
        workdir: str | None,
        env: Dict[str, str] | None,
    ) -> ExecResult:
        parts: list[str] = []
        if env:
            for k, v in env.items():
                parts.append(f"export {k}={shlex.quote(v)}")
        if workdir:
            parts.append(f"cd {shlex.quote(workdir)}")
        parts.append(command)
        full_command = " && ".join(parts)

        _SENTINEL = "__CUBE_EXEC_OK__"
        wrapped = (
            f"timeout {effective_timeout}s bash -lc {shlex.quote(full_command)}; echo EXIT_CODE:$?; echo {_SENTINEL}"
        )

        logger.info("exec [%s] (direct): %s", self._job_id[:8], command)
        result = None
        duration = 0.0
        for attempt in range(3):
            start = time.monotonic()
            result = _run_eai(
                ["job", "exec", self._job_id, "--", "bash", "-c", wrapped],
                eai_path=self._eai_path,
                profile=self._profile,
                account=self._account,
                timeout=effective_timeout + 30,
            )
            duration = time.monotonic() - start
            logger.info(
                "exec [%s] (direct): done in %.1fs, exit_code=%s",
                self._job_id[:8],
                duration,
                result.returncode,
            )
            # Phantom execution: sentinel missing + fast return + outer rc=0.
            # Happens after CLOSE_WAIT recovery — the eai CLI returns immediately
            # with empty stdout while the job-side bash never actually ran.
            if _SENTINEL not in result.stdout and result.returncode == 0 and duration < 2.0:
                logger.warning(
                    "exec [%s] (direct): phantom execution detected (attempt %d/3), retrying in 5s",
                    self._job_id[:8],
                    attempt + 1,
                )
                time.sleep(5)
                continue
            break

        stdout = result.stdout  # type: ignore[union-attr]
        stderr = result.stderr  # type: ignore[union-attr]

        exit_code = 1 if result.returncode != 0 else 0  # type: ignore[union-attr]
        lines = stdout.rstrip().split("\n")
        sentinel_removed = False
        for i in range(len(lines) - 1, -1, -1):
            if not sentinel_removed and lines[i] == _SENTINEL:
                lines.pop(i)
                sentinel_removed = True
            elif lines[i].startswith("EXIT_CODE:"):
                try:
                    exit_code = int(lines[i].split(":", 1)[1])
                except ValueError:
                    pass
                lines.pop(i)
                break
        stdout = "\n".join(lines)

        return ExecResult(
            stdout=stdout.strip(),
            stderr=stderr.strip(),
            exit_code=exit_code,
            duration_seconds=round(duration, 3),
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
