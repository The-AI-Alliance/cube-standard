"""EAI Toolkit / HPC backend — runs containers as EAI jobs via the ``eai`` CLI."""

from __future__ import annotations

import json
import logging
import shlex
import socket
import subprocess
import time
from typing import Dict

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
    ContainerExecError,
    ContainerLaunchError,
    ContainerSpec,
    ContainerStatus,
    ExecResult,
    HealthCheckError,
)

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
        )
    except FileNotFoundError as exc:
        raise ContainerLaunchError(
            "The 'eai' CLI tool is not installed or not on PATH."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise ContainerExecError(
            f"eai command timed out after {timeout}s: {' '.join(cmd)}"
        ) from exc
    return result


def _find_free_port() -> int:
    """Find a free local port by binding to port 0."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class ToolkitContainer(Container):
    """Runtime handle backed by an EAI Toolkit job."""

    def __init__(
        self,
        job_id: str,
        profile: str | None = None,
        account: str | None = None,
    ) -> None:
        self._job_id = job_id
        self._profile = profile
        self._account = account
        self._port_forwards: dict[int, subprocess.Popen] = {}
        self._port_map: dict[int, int] = {}

    @property
    def id(self) -> str:
        return self._job_id

    def exec(
        self,
        command: str,
        timeout: int | None = None,
        workdir: str | None = None,
        env: Dict[str, str] | None = None,
    ) -> ExecResult:
        effective_timeout = timeout if timeout is not None else 120

        parts: list[str] = []
        if env:
            for k, v in env.items():
                parts.append(f"export {k}={shlex.quote(v)}")
        if workdir:
            parts.append(f"cd {shlex.quote(workdir)}")
        parts.append(command)
        full_command = " && ".join(parts)

        # Wrap with exit-code capture since eai job exec doesn't relay exit codes
        wrapped = (
            f"timeout {effective_timeout}s bash -lc {shlex.quote(full_command)}; "
            f"echo EXIT_CODE:$?"
        )

        start = time.monotonic()
        result = _run_eai(
            ["job", "exec", self._job_id, "--", "bash", "-c", wrapped],
            profile=self._profile,
            account=self._account,
            timeout=effective_timeout + 30,
        )
        duration = time.monotonic() - start

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
                # Remove the marker line from stdout
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
                f"Port-forward process exited immediately "
                f"(rc={proc.returncode}) for port {container_port}"
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
    interactive: bool = True
    preemptable: bool = False

    def launch(self, spec: ContainerSpec) -> ToolkitContainer:
        return self._launch_with_retry(spec)

    @_retry_launch
    def _launch_with_retry(self, spec: ContainerSpec) -> ToolkitContainer:
        cmd: list[str] = ["job", "new"]

        if self.interactive:
            cmd.append("--interactive")
        if self.preemptable:
            cmd.append("--preemptable")

        cmd += ["--tunnel"]
        cmd += ["-i", spec.image]
        cmd += ["--cpu", str(int(spec.cpu_cores))]
        cmd += ["--mem", f"{int(spec.ram_gb)}"]

        if spec.gpu:
            cmd += ["--gpu", "1"]

        cmd += ["--", "sleep", "infinity"]

        logger.info("Creating EAI Toolkit job with image %s …", spec.image)

        result = _run_eai(
            cmd,
            profile=self.profile,
            account=self.account,
            timeout=self.timeout_seconds,
        )

        if result.returncode != 0:
            raise ContainerLaunchError(
                f"Failed to create EAI job: {result.stderr.strip()}"
            )

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
                raise ContainerLaunchError(
                    f"Could not parse job ID from eai output: {result.stdout}"
                ) from exc

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
                raise ContainerLaunchError(
                    f"EAI job {job_id} entered terminal state: {state}"
                )
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
                f"EAI job {job_id} did not reach RUNNING state "
                f"within {self.timeout_seconds}s (last state: {state})"
            )

        container = ToolkitContainer(job_id, profile=self.profile, account=self.account)
        logger.info("EAI job running: %s", job_id)

        self._run_health_check(container)
        return container
