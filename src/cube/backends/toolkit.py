"""
EAI Toolkit / HPC backend — runs containers as EAI jobs via the ``eai`` CLI.
Public documentation: https://docs.console.elementai.com/
"""

from __future__ import annotations

import json
import logging
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
) -> subprocess.CompletedProcess[str]:
    """Run an ``eai`` CLI command and return the completed process."""
    cmd = ["eai"]
    if profile:
        cmd += ["--profile", profile]
    if account:
        cmd += ["--account", account]
    cmd += args

    logger.debug("Running: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError as exc:
        raise ContainerLaunchError("The 'eai' CLI tool is not installed or not on PATH.") from exc
    except subprocess.TimeoutExpired as exc:
        raise ContainerExecError(f"eai command timed out after {timeout}s: {' '.join(cmd)}") from exc
    return result


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
        self._sidecar_token: str | None = None
        self._sidecar_local_port: int | None = None
        self._sidecar_ready = False

    @property
    def id(self) -> str:
        return self._job_id

    # ------------------------- sidecar bootstrap -------------------------

    def _bootstrap_sidecar(self) -> None:
        """Upload server, launch, port-forward, health-check. Idempotent."""
        if self._sidecar_ready or self._exec_mode == "direct":
            return

        # 256-bit token; stays in memory on the client, and on disk in the
        # container only as /tmp/.cube_sidecar_token (chmod 600, uid 13011).
        token = secrets.token_urlsafe(32)
        server_src = _SIDECAR_SERVER_PATH.read_text()

        # Heredoc-upload the server, write the token to a 0600 file, launch
        # nohup in background. The token never appears on the process argv —
        # it flows via stdin of one eai exec and lands in a 0600 file.
        bootstrap_script = (
            "set -e\n"
            "umask 077\n"
            "cat > /tmp/_cube_sidecar.py <<'CUBE_SIDECAR_EOF'\n"
            f"{server_src}\n"
            "CUBE_SIDECAR_EOF\n"
            "cat > /tmp/.cube_sidecar_token <<'CUBE_TOKEN_EOF'\n"
            f"{token}\n"
            "CUBE_TOKEN_EOF\n"
            "chmod 600 /tmp/.cube_sidecar_token /tmp/_cube_sidecar.py\n"
            f"export CUBE_SIDECAR_PORT={_SIDECAR_CONTAINER_PORT}\n"
            "export CUBE_SIDECAR_TOKEN_FILE=/tmp/.cube_sidecar_token\n"
            "python3 --version >/dev/null 2>&1 || { echo NO_PYTHON3; exit 1; }\n"
            "nohup python3 /tmp/_cube_sidecar.py "
            ">/tmp/_cube_sidecar.log 2>&1 &\n"
            "disown\n"
            "sleep 0.5\n"
            "echo SIDECAR_KICKED\n"
        )

        logger.info("Bootstrapping sidecar in job %s …", self._job_id[:8])
        result = _run_eai(
            ["job", "exec", self._job_id, "--", "bash", "-c", bootstrap_script],
            profile=self._profile,
            account=self._account,
            timeout=_SIDECAR_BOOTSTRAP_TIMEOUT,
        )
        if "SIDECAR_KICKED" not in result.stdout:
            raise ContainerLaunchError(
                f"Sidecar bootstrap failed: stdout={result.stdout!r} stderr={result.stderr!r}"
            )

        local_port = self.forward_port(_SIDECAR_CONTAINER_PORT)

        # Poll /health. Fresh port-forwards take a moment; the server also
        # takes ~100-300ms to bind.
        deadline = time.monotonic() + _SIDECAR_HEALTH_TIMEOUT
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{local_port}/health", timeout=1.0
                ) as r:
                    if r.status == 200:
                        self._sidecar_token = token
                        self._sidecar_local_port = local_port
                        self._sidecar_ready = True
                        logger.info(
                            "Sidecar ready for job %s on local port %d",
                            self._job_id[:8],
                            local_port,
                        )
                        return
            except (urllib.error.URLError, ConnectionError, OSError) as exc:
                last_err = exc
                time.sleep(0.2)
        raise ContainerLaunchError(
            f"Sidecar /health never returned 200 within {_SIDECAR_HEALTH_TIMEOUT}s: {last_err}"
        )

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
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Give port-forward a moment to establish
        time.sleep(2)
        if proc.poll() is not None:
            raise ContainerExecError(
                f"Port-forward process exited immediately (rc={proc.returncode}) for port {container_port}"
            )

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
