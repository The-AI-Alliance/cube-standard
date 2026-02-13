"""Daytona backend — run containers as Daytona sandboxes."""

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
    ContainerError,
    ContainerExecError,
    ContainerLaunchError,
    ContainerSpec,
    ContainerStatus,
    ExecResult,
    HealthCheckError,
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

_retry_poll = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)


# ---------------------------------------------------------------------------
# DaytonaContainer
# ---------------------------------------------------------------------------


class DaytonaContainer(Container):
    """Runtime handle backed by a Daytona sandbox."""

    def __init__(
        self,
        sandbox,
        client: Daytona,
    ) -> None:
        self._sandbox = sandbox
        self._client = client
        self._url_cache: dict[int, str] = {}

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
        session_id = str(uuid4())

        # Prepend env vars and cd to workdir if specified
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
            self._sandbox.process.create_session(session_id)

            response = self._sandbox.process.execute_session_command(
                session_id,
                SessionExecuteRequest(command=wrapped, run_async=True),
                timeout=effective_timeout,
            )

            if response.cmd_id is None:
                raise ContainerExecError("No command ID returned by Daytona")

            cmd, timed_out = self._poll_command(
                session_id, response.cmd_id, effective_timeout
            )

            if timed_out:
                duration = time.monotonic() - start
                return ExecResult(
                    stdout="",
                    stderr=f"Command timed out after {effective_timeout}s",
                    exit_code=124,
                    duration_seconds=round(duration, 3),
                )

            logs = self._get_session_logs(session_id, response.cmd_id)
            duration = time.monotonic() - start

            return ExecResult(
                stdout=(logs.stdout or "").strip(),
                stderr=(logs.stderr or "").strip(),
                exit_code=cmd.exit_code if cmd.exit_code is not None else 1,
                duration_seconds=round(duration, 3),
            )

        except ContainerExecError:
            raise
        except Exception as exc:
            raise ContainerExecError(f"Daytona exec failed: {exc}") from exc
        finally:
            try:
                self._sandbox.process.delete_session(session_id)
            except Exception:
                pass

    def forward_port(self, container_port: int) -> int:
        # Daytona uses signed preview URLs, not host port mapping.
        # We return the container port itself — get_url() provides the real URL.
        return container_port

    def get_url(self, container_port: int) -> str:
        if container_port in self._url_cache:
            return self._url_cache[container_port]

        try:
            preview = self._sandbox.create_signed_preview_url(
                container_port, expires_in_seconds=86400
            )
            url = preview.url
            self._url_cache[container_port] = url
            return url
        except Exception as exc:
            raise ContainerError(
                f"Failed to get URL for port {container_port}: {exc}"
            ) from exc

    @_retry_io
    def stop(self, timeout: int = 10) -> None:
        try:
            self._client.delete(self._sandbox)
        except DaytonaError:
            pass  # already gone — idempotent
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

    # -- internal helpers ---------------------------------------------------

    def _poll_command(self, session_id: str, command_id: str, timeout: int) -> tuple:
        deadline = time.monotonic() + timeout
        cmd = self._get_session_command(session_id, command_id)
        while cmd.exit_code is None:
            if time.monotonic() > deadline:
                return cmd, True
            time.sleep(0.5)
            cmd = self._get_session_command(session_id, command_id)
        return cmd, False

    @_retry_poll
    def _get_session_command(self, session_id: str, command_id: str):
        return self._sandbox.process.get_session_command(session_id, command_id)

    @_retry_poll
    def _get_session_logs(self, session_id: str, command_id: str):
        return self._sandbox.process.get_session_command_logs(session_id, command_id)


# ---------------------------------------------------------------------------
# DaytonaContainerBackend
# ---------------------------------------------------------------------------


class DaytonaContainerBackend(ContainerBackend):
    """Launch containers as Daytona sandboxes."""

    api_key: str | None = None
    api_url: str | None = None
    target: str | None = None
    ephemeral: bool = True
    auto_stop_minutes: int = 10
    auto_delete_minutes: int = 5

    @_retry_sandbox
    def launch(self, spec: ContainerSpec) -> DaytonaContainer:
        config_kwargs: dict[str, Any] = {}
        if self.api_key:
            config_kwargs["api_key"] = self.api_key
        if self.api_url:
            config_kwargs["api_url"] = self.api_url
        if self.target:
            config_kwargs["target"] = self.target

        daytona_config = (
            DaytonaConfig(**config_kwargs) if config_kwargs else DaytonaConfig()
        )
        client = Daytona(daytona_config)

        logger.info("Creating Daytona sandbox with image %s …", spec.image)
        try:
            create_kwargs: dict[str, Any] = {
                "image": Image.base(spec.image),
                "resources": Resources(
                    cpu=int(spec.cpu_cores),
                    memory=int(spec.ram_gb),
                    disk=int(spec.disk_gb),
                ),
                "auto_stop_interval": self.auto_stop_minutes,
                "ephemeral": self.ephemeral,
            }
            # ephemeral=True forces auto_delete_interval=0
            if not self.ephemeral:
                create_kwargs["auto_delete_interval"] = self.auto_delete_minutes
            params = CreateSandboxFromImageParams(**create_kwargs)
            sandbox = client.create(params, timeout=self.timeout_seconds)
        except DaytonaError as exc:
            raise ContainerLaunchError(
                f"Failed to create Daytona sandbox from '{spec.image}': {exc}"
            ) from exc

        container = DaytonaContainer(sandbox, client)
        logger.info("Daytona sandbox created: %s", container.id)

        self._run_health_check(container)
        return container
