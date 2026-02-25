from __future__ import annotations

import logging
import shlex
import time
from typing import Any, Dict
from uuid import uuid4

from daytona import (
    CreateSandboxFromImageParams,
    Daytona,
    DaytonaConfig,
    DaytonaError,
    Image,
    Resources,
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
    ContainerBackend,
    ContainerConfig,
    ContainerError,
    ContainerExecError,
    ContainerLaunchError,
    ContainerStatus,
    ExecResult,
    HealthCheckError,
    port_from_url,
)

logger = logging.getLogger(__name__)

_retry_sandbox = retry(
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


def _parse_session_output(raw: str) -> tuple[str, str]:
    """Parse multiplexed session output into (stdout, stderr).

    The Daytona session protocol prefixes output chunks with
    ``\\x01`` for stdout and ``\\x02`` for stderr.
    """
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    current = stdout_parts
    i = 0
    while i < len(raw):
        if raw[i] == "\x01":
            current = stdout_parts
            i += 1
        elif raw[i] == "\x02":
            current = stderr_parts
            i += 1
        else:
            j = i
            while j < len(raw) and raw[j] not in ("\x01", "\x02"):
                j += 1
            current.append(raw[i:j])
            i = j
    return "".join(stdout_parts), "".join(stderr_parts)


class DaytonaContainer(Container):
    """Runtime handle backed by a Daytona sandbox."""

    def __init__(
        self,
        sandbox,
        client: Daytona,
        allowed_ports: set[int] | None = None,
    ) -> None:
        self._sandbox = sandbox
        self._client = client
        self._url_cache: dict[int, str] = {}
        self._allowed_ports = allowed_ports
        self._session_id = str(uuid4())
        self._sandbox.process.create_session(self._session_id)

    @property
    def id(self) -> str:
        return self._sandbox.id

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
            response = self._sandbox.process.execute_session_command(
                self._session_id,
                SessionExecuteRequest(command=wrapped, run_async=False),
                timeout=effective_timeout + 10,
            )
            duration = time.monotonic() - start

            stdout, stderr = _parse_session_output(response.output or "")

            return ExecResult(
                stdout=stdout.strip(),
                stderr=stderr.strip(),
                exit_code=response.exit_code if response.exit_code is not None else 1,
                duration_seconds=round(duration, 3),
            )
        except Exception as exc:
            raise ContainerExecError(f"Daytona exec failed: {exc}") from exc

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


class DaytonaContainerBackend(ContainerBackend):
    """Launch containers as Daytona sandboxes."""

    api_key: str | None = None
    api_url: str | None = None
    target: str | None = None
    ephemeral: bool = True
    auto_stop_minutes: int = 10
    auto_delete_minutes: int = 5

    def launch(self, config: ContainerConfig) -> DaytonaContainer:
        return self._launch_with_retry(config)

    @_retry_sandbox
    def _launch_with_retry(self, config: ContainerConfig) -> DaytonaContainer:
        config_kwargs: dict[str, Any] = {}
        if self.api_key:
            config_kwargs["api_key"] = self.api_key
        if self.api_url:
            config_kwargs["api_url"] = self.api_url
        if self.target:
            config_kwargs["target"] = self.target

        daytona_config = DaytonaConfig(**config_kwargs) if config_kwargs else DaytonaConfig()
        client = Daytona(daytona_config)

        logger.info("Creating Daytona sandbox with image %s …", config.image)

        cpu = int(config.cpu_cores)
        memory = int(config.ram_gb)
        disk = int(config.disk_gb)
        if cpu != config.cpu_cores or memory != config.ram_gb or disk != config.disk_gb:
            logger.warning(
                "Daytona requires integer resources — truncating cpu_cores=%.1f→%d, ram_gb=%.1f→%d, disk_gb=%.1f→%d",
                config.cpu_cores,
                cpu,
                config.ram_gb,
                memory,
                config.disk_gb,
                disk,
            )

        resources_kwargs: dict[str, Any] = {
            "cpu": cpu,
            "memory": memory,
            "disk": disk,
        }
        if config.gpu:
            resources_kwargs["gpu"] = 1

        try:
            create_kwargs: dict[str, Any] = {
                "image": Image.base(config.image),
                "resources": Resources(**resources_kwargs),
                "auto_stop_interval": self.auto_stop_minutes,
                "ephemeral": self.ephemeral,
            }
            if not self.ephemeral:
                create_kwargs["auto_delete_interval"] = self.auto_delete_minutes
            params = CreateSandboxFromImageParams(**create_kwargs)
            sandbox = client.create(params, timeout=self.timeout_seconds)
        except DaytonaError as exc:
            raise ContainerLaunchError(f"Failed to create Daytona sandbox from '{config.image}': {exc}") from exc

        allowed_ports = set(config.ports) if config.ports else None
        container = DaytonaContainer(sandbox, client, allowed_ports=allowed_ports)
        logger.info("Daytona sandbox created: %s", container.id)

        self._run_health_check(container)
        return container
