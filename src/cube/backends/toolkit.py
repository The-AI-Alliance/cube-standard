"""
EAI Toolkit / HPC backend — runs containers as EAI jobs via the ``eai`` CLI.
Public documentation: https://docs.console.elementai.com/
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import shlex
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Literal

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from cube.container import (
    Container,
    ContainerBackend,
    ContainerConfig,
    ContainerExecError,
    ContainerLaunchError,
    ContainerStatus,
    ExecResult,
    HealthCheckError,
)

_SIDECAR_SERVER_PATH = Path(__file__).parent / "_toolkit_sidecar_server.py"
_SIDECAR_CONTAINER_PORT = 8787
_SIDECAR_BOOTSTRAP_TIMEOUT = 30
_SIDECAR_HEALTH_TIMEOUT = 15


class _SidecarUnavailable(Exception):
    """Raised when the sidecar is unreachable for a single call; triggers fallback to direct exec."""

logger = logging.getLogger(__name__)

_retry_launch = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
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


def _run_eai(
    args: list[str],
    *,
    profile: str | None = None,
    account: str | None = None,
    timeout: int | None = 60,
    retries: int = 2,
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
    import os
    import signal

    cmd = ["eai"]
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
                stdin=subprocess.DEVNULL,
                text=True,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise ContainerLaunchError("The 'eai' CLI tool is not installed or not on PATH.") from exc

        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            last_err = exc
            # Kill the entire process group so no defunct children survive.
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                proc.communicate(timeout=5)  # drain pipes so the PID reaps cleanly
            except Exception:
                pass
            if attempt < retries:
                # Hangs empirically arrive in clusters — back off a few seconds
                # so a retry isn't likely to hit the same transient window.
                backoff = 5 * (attempt + 1)
                logger.warning(
                    "eai command timed out after %ds (attempt %d/%d); backing off %ds before retry: %s",
                    timeout, attempt + 1, retries + 1, backoff, " ".join(cmd),
                )
                time.sleep(backoff)
                continue
            raise ContainerExecError(
                f"eai command timed out after {timeout}s (tried {retries + 1} times): {' '.join(cmd)}"
            ) from exc

        return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)

    # Unreachable — loop either returns or raises.
    assert last_err is not None
    raise ContainerExecError(f"eai: unreachable branch") from last_err


def _find_free_port() -> int:
    """Find a free local port by binding to port 0."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class ToolkitContainer(Container):
    """Runtime handle backed by an EAI Toolkit job.

    Exec routing: `eai job exec` has a known TCP half-close bug that hangs
    ~6% of calls. By default (exec_mode="sidecar") we bootstrap a small HTTP
    server inside the container at launch and route all `.exec()` calls via
    an `eai job port-forward` tunnel to it. The direct `eai job exec` path
    is still used for bootstrap + as a fallback. See _toolkit_sidecar_server.py
    for the server-side security posture.
    """

    def __init__(
        self,
        job_id: str,
        profile: str | None = None,
        account: str | None = None,
        exec_mode: Literal["sidecar", "direct"] = "sidecar",
    ) -> None:
        self._job_id = job_id
        self._profile = profile
        self._account = account
        self._exec_mode = exec_mode
        self._port_forwards: dict[int, subprocess.Popen] = {}
        self._port_map: dict[int, int] = {}
        self._port_forward_logs: dict[int, str] = {}
        self._sidecar_token: str | None = None
        self._sidecar_local_port: int | None = None
        self._sidecar_ready = False

    @property
    def id(self) -> str:
        return self._job_id

    # ------------------------- sidecar bootstrap -------------------------

    def _bootstrap_sidecar(self) -> None:
        """Upload server, launch, port-forward, health-check.

        Because the eai-exec RPC that uploads + starts the server is itself
        subject to the CLOSE_WAIT hang we're trying to fix, we can't trust
        the bootstrap's exit code — SIGTERMing the remote bash does NOT kill
        the setsid-detached python child that's already running.  Instead:
        retry the whole bootstrap up to 3 times, and treat the health probe
        as the only ground truth for "sidecar is usable".
        """
        if self._sidecar_ready or self._exec_mode == "direct":
            return

        # 256-bit token, reused across bootstrap retries.  Server reads from
        # /tmp/.cube_sidecar_token (0600) at startup, so writing the same
        # content on each retry is idempotent.
        token = secrets.token_urlsafe(32)
        last_diag = ""

        # Open the tunnel BEFORE the first `eai job exec`: when exec hangs
        # (the CLOSE_WAIT bug) and we kill its process group, the eai tunnel
        # gateway gets into a bad state for this job and new port-forwards
        # consistently fail with "websocket: bad handshake".  Forwarding
        # first, while the gateway is fresh, avoids the corruption.
        local_port = self.forward_port(_SIDECAR_CONTAINER_PORT)

        for attempt in range(3):
            self._kick_sidecar(token)
            if self._probe_health(local_port, timeout=_SIDECAR_HEALTH_TIMEOUT):
                self._sidecar_token = token
                self._sidecar_local_port = local_port
                self._sidecar_ready = True
                logger.info(
                    "Sidecar ready for job %s on local port %d (attempt %d)",
                    self._job_id[:8], local_port, attempt + 1,
                )
                return
            last_diag = self._fetch_sidecar_diagnostics()
            logger.warning(
                "Sidecar health failed on attempt %d/3 for job %s. Diag:\n%s",
                attempt + 1, self._job_id[:8], last_diag,
            )

        # All 3 health probes failed — sidecar unbootstrappable (no python3, or
        # persistent network issue).  Silently fall back to direct eai exec so
        # the task can still run; the caller does not need to handle this case.
        logger.warning(
            "Sidecar bootstrap failed after 3 attempts for job %s — "
            "falling back to direct exec mode.  Last diag:\n%s",
            self._job_id[:8], last_diag,
        )
        self._exec_mode = "direct"

    def _kick_sidecar(self, token: str) -> None:
        """Upload the server + token, kill any prior instance, start detached.

        Idempotent: safe to call repeatedly.  Ignores bootstrap exit code —
        the health probe is what decides success.  The eai CLI often SIGTERMs
        the remote bash when the response-delivery channel hangs (CLOSE_WAIT);
        setsid --fork detaches python from that signal, so the server survives
        even when the bash appears to have failed with rc=143.
        """
        server_src = _SIDECAR_SERVER_PATH.read_text()
        script = (
            "umask 077\n"
            # pgrep matches the current bash's argv (it contains this script,
            # which references '_cube_sidecar.py' in the heredoc below).  $$
            # excludes us from the kill set.
            "pgrep -f _cube_sidecar.py 2>/dev/null "
            "| grep -vw \"$$\" "
            "| xargs -r kill 2>/dev/null || true\n"
            "sleep 0.3\n"
            "cat > /tmp/_cube_sidecar.py <<'CUBE_SIDECAR_EOF'\n"
            f"{server_src}\n"
            "CUBE_SIDECAR_EOF\n"
            "cat > /tmp/.cube_sidecar_token <<'CUBE_TOKEN_EOF'\n"
            f"{token}\n"
            "CUBE_TOKEN_EOF\n"
            "chmod 600 /tmp/.cube_sidecar_token /tmp/_cube_sidecar.py\n"
            f"export CUBE_SIDECAR_PORT={_SIDECAR_CONTAINER_PORT}\n"
            "export CUBE_SIDECAR_TOKEN_FILE=/tmp/.cube_sidecar_token\n"
            # setsid --fork puts python in a new session AND makes it a child
            # of pid 1, so when the eai exec closes (taking bash with it), the
            # python survives.  stdin from /dev/null prevents SIGHUP on close.
            "setsid --fork python3 /tmp/_cube_sidecar.py "
            "</dev/null >/tmp/_cube_sidecar.log 2>&1\n"
            "echo KICKED\n"
        )
        logger.info("Bootstrapping sidecar in job %s …", self._job_id[:8])
        try:
            _run_eai(
                ["job", "exec", self._job_id, "--", "bash", "-c", script],
                profile=self._profile,
                account=self._account,
                timeout=_SIDECAR_BOOTSTRAP_TIMEOUT,
                retries=0,  # we retry at the bootstrap level via _bootstrap_sidecar
            )
        except ContainerExecError as exc:
            # CLI-level timeout — bash was killed, but setsid-forked python
            # probably survived.  Proceed to health probe to find out.
            logger.info("Bootstrap RPC timed out for job %s (%s); health probe will decide",
                        self._job_id[:8], exc)

    def _reset_sidecar_port_forward(self) -> None:
        """Kill any existing port-forward for the sidecar port so forward_port reopens it."""
        port = _SIDECAR_CONTAINER_PORT
        proc = self._port_forwards.pop(port, None)
        self._port_map.pop(port, None)
        log_path = self._port_forward_logs.pop(port, None)
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        if log_path:
            try:
                os.unlink(log_path)
            except Exception:
                pass

    def _probe_health(self, local_port: int, *, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{local_port}/health", timeout=1.0
                ) as r:
                    if r.status == 200:
                        return True
            except (urllib.error.URLError, ConnectionError, OSError):
                time.sleep(0.2)
        return False

    def _fetch_sidecar_diagnostics(self) -> str:
        parts = []
        try:
            r = _run_eai(
                ["job", "exec", self._job_id, "--", "bash", "-c",
                 "tail -20 /tmp/_cube_sidecar.log 2>/dev/null; "
                 "echo ---ps---; ps -ef | grep _cube_sidecar | grep -v grep; "
                 "echo ---curl---; curl -sS --max-time 2 http://127.0.0.1:8787/health 2>&1 | head -5"],
                profile=self._profile,
                account=self._account,
                timeout=30,
                retries=0,
            )
            parts.append(r.stdout.strip() or "<empty>")
        except Exception as exc:
            parts.append(f"<diagnostics fetch failed: {exc}>")

        # Also capture the eai port-forward process's own stderr — silent tunnel
        # failures are often visible only here.
        log_path = self._port_forward_logs.get(_SIDECAR_CONTAINER_PORT)
        if log_path:
            try:
                with open(log_path) as f:
                    pf_err = f.read().strip()
            except Exception as exc:
                pf_err = f"<read failed: {exc}>"
            proc = self._port_forwards.get(_SIDECAR_CONTAINER_PORT)
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

        if self._exec_mode == "sidecar":
            try:
                if not self._sidecar_ready:
                    self._bootstrap_sidecar()
                # _bootstrap_sidecar may have silently fallen back to direct
                # mode (e.g. image has no python3); re-check before dispatching.
                if self._sidecar_ready:
                    return self._exec_via_sidecar(command, effective_timeout, workdir, env)
            except _SidecarUnavailable as exc:
                logger.warning(
                    "Sidecar exec failed for job %s, falling back to direct eai exec: %s",
                    self._job_id[:8],
                    exc,
                )
                # Don't keep retrying bootstrap for this container.
                self._sidecar_ready = False
                self._exec_mode = "direct"

        return self._exec_direct(command, effective_timeout, workdir, env)

    def _exec_via_sidecar(
        self,
        command: str,
        timeout: int,
        workdir: str | None,
        env: Dict[str, str] | None,
    ) -> ExecResult:
        assert self._sidecar_ready and self._sidecar_token and self._sidecar_local_port
        payload: dict = {"command": command, "timeout": timeout}
        if workdir:
            payload["workdir"] = workdir
        if env:
            payload["env"] = env
        data = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            f"http://127.0.0.1:{self._sidecar_local_port}/exec",
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._sidecar_token}",
            },
        )
        logger.info("exec [%s] (sidecar): %s", self._job_id[:8], command)
        start = time.monotonic()
        try:
            # Client-side read timeout = command timeout + generous overhead
            # for JSON serialization on large stdout/stderr payloads.
            with urllib.request.urlopen(req, timeout=timeout + 30) as r:
                body = json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, ConnectionError, OSError, json.JSONDecodeError) as exc:
            raise _SidecarUnavailable(str(exc)) from exc

        duration = time.monotonic() - start
        logger.info(
            "exec [%s] (sidecar): done in %.1fs, exit_code=%s",
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

        # Wrap with exit-code capture since eai job exec doesn't relay exit codes
        wrapped = f"timeout {effective_timeout}s bash -lc {shlex.quote(full_command)}; echo EXIT_CODE:$?"

        logger.info("exec [%s] (direct): %s", self._job_id[:8], command)
        start = time.monotonic()
        result = _run_eai(
            ["job", "exec", self._job_id, "--", "bash", "-c", wrapped],
            profile=self._profile,
            account=self._account,
            # Tight CLI buffer: the inner `timeout ${N}s` already bounds the
            # command, so this buffer only needs to cover ``eai``'s wire-up
            # and response-delivery overhead (normally <5s).  Keeping it tight
            # makes _run_eai's retry-on-hang fire quickly instead of waiting a
            # full extra minute on every hung call.
            timeout=effective_timeout + 30,
        )
        duration = time.monotonic() - start
        logger.info(
            "exec [%s] (direct): done in %.1fs, exit_code=%s",
            self._job_id[:8],
            duration,
            result.returncode,
        )

        stdout = result.stdout
        stderr = result.stderr

        # Parse exit code from our EXIT_CODE marker
        exit_code = 1 if result.returncode != 0 else 0
        lines = stdout.rstrip().split("\n")
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith("EXIT_CODE:"):
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

    def exec_long_running(
        self,
        command: str,
        *,
        timeout: int,
        poll_interval: int = 30,
        workdir: str | None = None,
        env: Dict[str, str] | None = None,
    ) -> ExecResult:
        """Run a long-running command via background+poll.

        Root cause: ``eai job exec`` occasionally hangs for minutes on the
        response-delivery side of long-running commands (the CLI blocks in
        a ``read()`` on an HTTPS socket where the server has already closed
        its half — see `docs/toolkit-hang-bugreport.md`).  A retry on the
        outer call re-runs the entire command, wasting the original work.

        Fix: decouple the long command from the RPC channel.

          1. One short ``exec`` kicks off ``(cmd > out 2>&1; echo $? > rc) & disown``
             and returns in ~1 s.
          2. Short polls (``test -f <rc>``) every ``poll_interval`` s check
             whether the command has completed.  Each poll is short and
             retry-safe.
          3. A final short ``exec`` reads the captured stdout/stderr and
             returns the observed exit code.

        Works because every individual RPC call is short — the 6 % per-call
        eai hang rate is amortised across many short polls instead of bet
        on one long ``exec``.
        """
        import uuid as _uuid

        marker_base = f"/tmp/cube_lr_{_uuid.uuid4().hex[:8]}"
        out_path = f"{marker_base}.out"
        rc_path = f"{marker_base}.rc"

        # Build an idempotent kick-off command.
        # `mkdir <lock>` is POSIX-atomic: on first kick it succeeds and we
        # fire the background command; on any retry (because the first kick
        # RPC hung) it fails and we no-op.  This guarantees the command runs
        # exactly once in the container — critical for non-idempotent
        # commands like `git apply` where a second execution fails because
        # the patch is already applied.
        lock_path = f"{marker_base}.lock"
        parts = []
        if env:
            for k, v in env.items():
                parts.append(f"export {k}={shlex.quote(v)}")
        if workdir:
            parts.append(f"cd {shlex.quote(workdir)}")
        parts.append(
            f"if mkdir {lock_path} 2>/dev/null; then "
            f"(timeout {timeout}s bash -c {shlex.quote(command)} > {out_path} 2>&1; "
            f"echo $? > {rc_path}) & disown; "
            f"echo KICKED; "
            f"else "
            f"echo ALREADY_STARTED; "
            f"fi"
        )
        kick_cmd = " && ".join(parts)

        logger.info("exec_long_running [%s]: kicking off, marker=%s", self._job_id[:8], marker_base)
        # Even if the kick RPC hangs (we've observed this — see bugreport H4),
        # the container may still have received and started the command.  So
        # we swallow CLI-level hangs / non-zero exit codes on the kick and
        # proceed to polling.  If the command was never actually submitted,
        # the poll deadline will catch that.
        try:
            kick = self.exec(kick_cmd, timeout=30)
            if kick.exit_code != 0:
                logger.warning(
                    "exec_long_running [%s]: kick exited non-zero (rc=%d stderr=%r) — "
                    "proceeding to poll anyway (bg command may still be running)",
                    self._job_id[:8], kick.exit_code, kick.stderr[:200],
                )
        except ContainerExecError as exc:
            logger.warning(
                "exec_long_running [%s]: kick RPC failed (%s) — proceeding to poll anyway "
                "(container may have received the command despite the CLI hang)",
                self._job_id[:8], exc,
            )

        # Poll for completion.  Each poll is a short exec that's retry-safe
        # via _run_eai's built-in retry; on the rare chance all retries still
        # hang, we swallow the error and poll again on the next tick.
        deadline = time.monotonic() + timeout + 60  # small grace period for filesystem flush
        last_log = 0.0
        while time.monotonic() < deadline:
            try:
                # retries=0 on polls: a hung poll costs 30s (its own timeout)
                # instead of 30+5+30+10+30=105s with full retry chain.
                # The outer while-loop naturally retries on the next tick —
                # no need to double up retries here.
                poll_result = _run_eai(
                    ["job", "exec", self._job_id, "--", "bash", "-c",
                     f"if [ -f {rc_path} ]; then cat {rc_path}; else echo PENDING; fi"],
                    profile=self._profile,
                    account=self._account,
                    timeout=30,
                    retries=0,
                )
                body = poll_result.stdout.strip()
            except ContainerExecError as exc:
                logger.warning("exec_long_running [%s]: poll hung (%s); retrying on next tick", self._job_id[:8], exc)
                body = "PENDING"
            if body and body != "PENDING":
                try:
                    rc = int(body.split()[0])
                except ValueError:
                    rc = 1
                try:
                    out_result = self.exec(f"cat {out_path} 2>/dev/null", timeout=60)
                    stdout = out_result.stdout
                except ContainerExecError as exc:
                    logger.warning(
                        "exec_long_running [%s]: couldn't fetch output (%s); "
                        "returning rc=%d with empty stdout", self._job_id[:8], exc, rc,
                    )
                    stdout = ""
                # Best-effort cleanup.  Failure to clean up is not fatal.
                try:
                    self.exec(f"rm -rf {out_path} {rc_path} {lock_path}", timeout=30)
                except Exception:
                    pass
                return ExecResult(
                    stdout=stdout,
                    stderr="",
                    exit_code=rc,
                    duration_seconds=timeout + 60 - (deadline - time.monotonic()),
                )
            # Log infrequently to avoid spam on multi-minute commands.
            now = time.monotonic()
            if now - last_log > 60:
                logger.info("exec_long_running [%s]: still running …", self._job_id[:8])
                last_log = now
            time.sleep(poll_interval)

        raise ContainerExecError(
            f"exec_long_running timed out after {timeout}s "
            f"(marker {marker_base} in container {self._job_id})"
        )

    def forward_port(self, container_port: int) -> int:
        if container_port in self._port_map:
            return self._port_map[container_port]

        local_port = _find_free_port()

        cmd = ["eai"]
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
        # Capture stderr to a tempfile so we can diagnose silent failures
        # ("exited 0 but the tunnel never actually works").
        import tempfile
        stderr_file = tempfile.NamedTemporaryFile(
            prefix=f"eai-pf-{container_port}-", suffix=".log", delete=False, mode="w"
        )
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=stderr_file,
        )

        # Give port-forward a moment to establish
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
        # Kill all port-forward background processes
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

        # Kill the EAI job
        logger.info("stop [%s]: killing job", self._job_id[:8])
        try:
            _run_eai(
                ["job", "kill", self._job_id],
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


class ToolkitContainerBackend(ContainerBackend):
    """Launch containers as EAI Toolkit jobs via the ``eai`` CLI."""

    profile: str | None = None
    account: str | None = None
    interactive: bool = False
    preemptable: bool = False
    exec_mode: Literal["sidecar", "direct"] = "sidecar"

    def launch(self, config: ContainerConfig) -> ToolkitContainer:
        return self._launch_with_retry(config)

    @_retry_launch
    def _launch_with_retry(self, config: ContainerConfig) -> ToolkitContainer:
        cmd: list[str] = ["job", "new"]

        if self.interactive:
            cmd.append("--interactive")
        elif self.preemptable:
            cmd.append("--preemptable")
        else:
            cmd.append("--non-preemptable")

        cmd += ["--tunnel"]
        cmd += ["--format", "json", "--no-header"]
        cmd += ["-i", config.image]
        cmd += ["--cpu", str(int(config.cpu_cores))]
        cmd += ["--mem", f"{int(config.ram_gb)}"]

        if config.gpu:
            cmd += ["--gpu", "1"]

        cmd += ["--", "sleep", "infinity"]

        logger.info("Creating EAI Toolkit job with image %s …", config.image)

        result = _run_eai(
            cmd,
            profile=self.profile,
            account=self.account,
            timeout=self.timeout_seconds,
        )

        if result.returncode != 0:
            raise ContainerLaunchError(f"Failed to create EAI job: {result.stderr.strip()}")

        # Parse job ID from output
        try:
            output = json.loads(result.stdout)
            job_id = output["id"]
        except (json.JSONDecodeError, KeyError) as exc:
            # Fallback: try to extract ID from first line of output
            first_line = result.stdout.strip().split("\n")[0].strip()
            if first_line:
                job_id = first_line
            else:
                raise ContainerLaunchError(f"Could not parse job ID from eai output: {result.stdout}") from exc

        logger.info("EAI job created: %s — waiting for RUNNING state …", job_id)

        # Poll until the job is running
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            status_result = _run_eai(
                ["job", "get", job_id, "--format", "json", "--no-header"],
                profile=self.profile,
                account=self.account,
                timeout=30,
            )
            try:
                info = json.loads(status_result.stdout)
                state = info.get("state", "").lower()
            except (json.JSONDecodeError, AttributeError):
                state = ""

            if state == "running":
                break
            if state in ("failed", "cancelled", "killed"):
                raise ContainerLaunchError(f"EAI job {job_id} entered terminal state: {state}")
            logger.info("EAI job [%s] state=%s, waiting …", job_id[:8], state)
            time.sleep(5)
        else:
            # Timed out — try to kill the job before raising
            try:
                _run_eai(
                    ["job", "kill", job_id],
                    profile=self.profile,
                    account=self.account,
                    timeout=30,
                )
            except Exception:
                pass
            raise ContainerLaunchError(
                f"EAI job {job_id} did not reach RUNNING state within {self.timeout_seconds}s (last state: {state})"
            )

        container = ToolkitContainer(
            job_id,
            profile=self.profile,
            account=self.account,
            exec_mode=self.exec_mode,
        )
        logger.info("EAI job running: %s", job_id)

        self._run_health_check(container)
        return container
