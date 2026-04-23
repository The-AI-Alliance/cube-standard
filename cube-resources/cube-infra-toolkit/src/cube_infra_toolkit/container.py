"""ToolkitContainer — ``cube.container.Container`` implementation backed by an EAI Toolkit job.

Canonical home for the Toolkit driver.  ``cube.backends.toolkit`` re-exports
this class for back-compat with the deprecated ``ToolkitContainerBackend``
factory; new code should use ``ToolkitInfraConfig`` from this package.
"""

from __future__ import annotations

import importlib.resources
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

_SIDECAR_SERVER_PATH = Path(__file__).parent / "_sidecar_server.py"
_SIDECAR_CONTAINER_PORT = 8787
_SIDECAR_BOOTSTRAP_TIMEOUT = 30
_SIDECAR_HEALTH_TIMEOUT = 15

# Architecture token → subdirectory name inside _bin/
_ARCH_MAP: dict[str, str] = {
    "x86_64": "linux-amd64",
    "aarch64": "linux-arm64",
    "arm64": "linux-arm64",
}


def _load_sidecar_binary(arch_dir: str) -> bytes:
    """Return bytes of the pre-built static cube-sidecar binary for ``arch_dir``.

    ``arch_dir`` is one of the keys in ``_ARCH_MAP`` values, e.g. 'linux-amd64'.
    Raises ``FileNotFoundError`` if the binary is absent (e.g. dev checkout
    without running ``make`` in sidecar-go/).
    """
    pkg = importlib.resources.files("cube_infra_toolkit")
    binary_path = pkg / "_bin" / arch_dir / "cube-sidecar"
    try:
        return binary_path.read_bytes()
    except (FileNotFoundError, TypeError) as exc:
        raise FileNotFoundError(f"Pre-built sidecar binary not found at _bin/{arch_dir}/cube-sidecar") from exc


class _SidecarUnavailable(Exception):
    """Raised when the sidecar is unreachable for a single call; triggers fallback to direct exec."""


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
            f"The 'eai' CLI tool was not found at {cmd[0]!r}. "
            "Install it or set ToolkitInfraConfig(eai_path=...)."
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
                    timeout, attempt + 1, retries + 1, backoff, " ".join(cmd),
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


class ToolkitContainer(Container):
    """Runtime handle backed by an EAI Toolkit job.

    Exec routing: `eai job exec` has a known TCP half-close bug that hangs
    ~6% of calls. By default (exec_mode="sidecar") we bootstrap a small HTTP
    server inside the container at launch and route all `.exec()` calls via
    an `eai job port-forward` tunnel to it. The direct `eai job exec` path
    is still used for bootstrap + as a fallback. See ``_sidecar_server.py``
    for the server-side security posture.
    """

    def __init__(
        self,
        job_id: str,
        profile: str | None = None,
        account: str | None = None,
        exec_mode: Literal["sidecar", "direct"] = "sidecar",
        eai_path: str = "eai",
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
        self._sidecar_token: str | None = None
        self._sidecar_local_port: int | None = None
        self._sidecar_ready = False

    @property
    def id(self) -> str:
        return self._job_id

    # ------------------------- sidecar bootstrap -------------------------

    def _detect_arch(self) -> str:
        """Return the arch dir name ('linux-amd64' or 'linux-arm64') for the container."""
        result = _run_eai(
            ["job", "exec", self._job_id, "--", "uname", "-m"],
            eai_path=self._eai_path,
            profile=self._profile,
            account=self._account,
            timeout=15,
            retries=1,
        )
        machine = result.stdout.strip()
        arch_dir = _ARCH_MAP.get(machine)
        if not arch_dir:
            raise ContainerLaunchError(f"Unsupported container architecture: {machine!r}")
        return arch_dir

    def _ensure_python3(self) -> None:
        """Install python3 via apt if the image doesn't have it.

        The sidecar server is a Python script — it can't start without python3.
        Minimal images (bare LaTeX, Alpine-derived, etc.) often omit it.
        Uses direct eai exec (bypasses sidecar which doesn't exist yet).
        """
        check = _run_eai(
            ["job", "exec", self._job_id, "--", "bash", "-c", "python3 --version 2>/dev/null && echo HAS_PYTHON || echo NO_PYTHON"],
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
            ["job", "exec", self._job_id, "--", "bash", "-c",
             "apt-get update -qq && apt-get install -y --no-install-recommends python3 python3-pip 2>&1"],
            eai_path=self._eai_path,
            profile=self._profile,
            account=self._account,
            timeout=120,
            retries=1,
        )

    def _bootstrap_sidecar(self) -> None:
        """Upload sidecar, launch, port-forward, health-check.

        Tries the pre-built static Go binary first (no python3 needed).
        Falls back to the Python script if the binary is unavailable.
        Retries the whole bootstrap up to 3 times; health probe is the only
        ground truth for "sidecar is usable".
        """
        if self._sidecar_ready or self._exec_mode == "direct":
            return

        # Prefer Go binary — no python3 dependency, works in any Linux image.
        binary: bytes | None = None
        try:
            arch_dir = self._detect_arch()
            binary = _load_sidecar_binary(arch_dir)
            logger.info("Using Go sidecar binary (%s, %d bytes) for job %s",
                        arch_dir, len(binary), self._job_id[:8])
        except (FileNotFoundError, ContainerLaunchError) as exc:
            logger.info("Go binary unavailable (%s) — falling back to Python sidecar + apt install", exc)
            self._ensure_python3()

        token = secrets.token_urlsafe(32)
        # Open tunnel BEFORE the first exec — avoids gateway corruption on hang.
        local_port = self.forward_port(_SIDECAR_CONTAINER_PORT)
        last_diag = ""

        for attempt in range(3):
            if binary is not None:
                self._kick_sidecar_binary(token, binary)
            else:
                self._kick_sidecar_python(token)
            if self._probe_health(local_port, timeout=_SIDECAR_HEALTH_TIMEOUT):
                self._sidecar_token = token
                self._sidecar_local_port = local_port
                self._sidecar_ready = True
                logger.info("Sidecar ready for job %s on local port %d (attempt %d, binary=%s)",
                            self._job_id[:8], local_port, attempt + 1, binary is not None)
                return
            last_diag = self._fetch_sidecar_diagnostics()
            logger.warning("Sidecar health failed on attempt %d/3 for job %s. Diag:\n%s",
                           attempt + 1, self._job_id[:8], last_diag)

        logger.warning("Sidecar bootstrap failed after 3 attempts for job %s — "
                       "falling back to direct exec mode.  Last diag:\n%s",
                       self._job_id[:8], last_diag)
        self._exec_mode = "direct"

    def _kick_sidecar_binary(self, token: str, binary: bytes) -> None:
        """Upload the static binary via stdin, write token file, start detached."""
        logger.info("Uploading Go sidecar binary to job %s (%d bytes) …",
                    self._job_id[:8], len(binary))
        upload_script = (
            "umask 077\n"
            "pgrep -f _cube_sidecar 2>/dev/null | grep -vw \"$$\" | xargs -r kill 2>/dev/null || true\n"
            "cat > /tmp/_cube_sidecar && chmod 700 /tmp/_cube_sidecar\n"
            "echo UPLOADED\n"
        )
        try:
            _run_eai(
                ["job", "exec", self._job_id, "--", "bash", "-c", upload_script],
                eai_path=self._eai_path,
                profile=self._profile,
                account=self._account,
                timeout=60,
                retries=1,
                input=binary,
            )
        except ContainerExecError as exc:
            logger.info("Binary upload timed out for job %s (%s); health probe will decide",
                        self._job_id[:8], exc)
            return

        start_script = (
            "cat > /tmp/.cube_sidecar_token <<'CUBE_TOKEN_EOF'\n"
            f"{token}\n"
            "CUBE_TOKEN_EOF\n"
            "chmod 600 /tmp/.cube_sidecar_token\n"
            f"export CUBE_SIDECAR_PORT={_SIDECAR_CONTAINER_PORT}\n"
            "export CUBE_SIDECAR_TOKEN_FILE=/tmp/.cube_sidecar_token\n"
            "setsid --fork /tmp/_cube_sidecar </dev/null >/tmp/_cube_sidecar.log 2>&1\n"
            "echo KICKED\n"
        )
        try:
            _run_eai(
                ["job", "exec", self._job_id, "--", "bash", "-c", start_script],
                eai_path=self._eai_path,
                profile=self._profile,
                account=self._account,
                timeout=_SIDECAR_BOOTSTRAP_TIMEOUT,
                retries=0,
            )
        except ContainerExecError as exc:
            logger.info("Start RPC timed out for job %s (%s); health probe will decide",
                        self._job_id[:8], exc)

    def _kick_sidecar_python(self, token: str) -> None:
        """Upload the Python sidecar script + token, kill any prior instance, start detached."""
        server_src = _SIDECAR_SERVER_PATH.read_text()
        script = (
            "umask 077\n"
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
            "setsid --fork python3 /tmp/_cube_sidecar.py "
            "</dev/null >/tmp/_cube_sidecar.log 2>&1\n"
            "echo KICKED\n"
        )
        logger.info("Bootstrapping Python sidecar in job %s …", self._job_id[:8])
        try:
            _run_eai(
                ["job", "exec", self._job_id, "--", "bash", "-c", script],
                eai_path=self._eai_path,
                profile=self._profile,
                account=self._account,
                timeout=_SIDECAR_BOOTSTRAP_TIMEOUT,
                retries=0,
            )
        except ContainerExecError as exc:
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
                 "echo ---ps---; ps -ef | grep _cube_sidecar | grep -vw grep; "
                 "echo ---curl---; curl -sS --max-time 2 http://127.0.0.1:8787/health 2>&1 | head -5"],
                eai_path=self._eai_path,
                profile=self._profile,
                account=self._account,
                timeout=30,
                retries=0,
            )
            parts.append(r.stdout.strip() or "<empty>")
        except Exception as exc:
            parts.append(f"<diagnostics fetch failed: {exc}>")

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

        wrapped = f"timeout {effective_timeout}s bash -lc {shlex.quote(full_command)}; echo EXIT_CODE:$?"

        logger.info("exec [%s] (direct): %s", self._job_id[:8], command)
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

        stdout = result.stdout
        stderr = result.stderr

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
