"""Modal backend — run containers as Modal Sandboxes."""

from __future__ import annotations

import logging
import shlex
import time
from typing import Any, Dict

import modal
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


# ---------------------------------------------------------------------------
# ModalContainer
# ---------------------------------------------------------------------------


class ModalContainer(Container):
    """Runtime handle backed by a Modal Sandbox."""

    def __init__(self, sandbox: modal.Sandbox) -> None:
        self._sandbox = sandbox
        self._url_cache: dict[int, str] = {}

    @property
    def id(self) -> str:
        return self._sandbox.object_id

    def exec(
        self,
        command: str,
        timeout: int | None = None,
        workdir: str | None = None,
        env: Dict[str, str] | None = None,
    ) -> ExecResult:
        effective_timeout = timeout if timeout is not None else 120

        # Wrap with timeout for enforcement
        wrapped = f"timeout {effective_timeout}s bash -lc {shlex.quote(command)}"

        kwargs: dict[str, Any] = {}
        if workdir:
            kwargs["workdir"] = workdir
        if env:
            kwargs["env"] = env

        start = time.monotonic()
        try:
            process = self._sandbox.exec(
                "bash", "-c", wrapped, **kwargs
            )
            stdout = process.stdout.read()
            stderr = process.stderr.read()
            exit_code = process.wait()
        except Exception as exc:
            raise ContainerExecError(f"Modal exec failed: {exc}") from exc
        duration = time.monotonic() - start

        return ExecResult(
            stdout=stdout.strip(),
            stderr=stderr.strip(),
            exit_code=exit_code,
            duration_seconds=round(duration, 3),
        )

    def forward_port(self, container_port: int) -> int:
        # Modal uses tunnel URLs, not host port mapping.
        # Return the container port — get_url() provides the real URL.
        return container_port

    def get_url(self, container_port: int) -> str:
        if container_port in self._url_cache:
            return self._url_cache[container_port]

        try:
            tunnels = self._sandbox.tunnels()
            if container_port not in tunnels:
                raise ContainerError(
                    f"Port {container_port} has no tunnel. "
                    f"Available: {list(tunnels.keys())}"
                )
            url = tunnels[container_port].url
            self._url_cache[container_port] = url
            return url
        except ContainerError:
            raise
        except Exception as exc:
            raise ContainerError(
                f"Failed to get tunnel for port {container_port}: {exc}"
            ) from exc

    @_retry_io
    def stop(self, timeout: int = 10) -> None:
        try:
            self._sandbox.terminate()
        except Exception as exc:
            logger.warning("Error terminating Modal sandbox %s: %s", self.id, exc)

    def get_status(self) -> ContainerStatus:
        try:
            rc = self._sandbox.poll()
            running = rc is None
            return ContainerStatus(
                running=running,
                healthy=running,
                backend_info={
                    "returncode": rc,
                    "id": self.id,
                },
            )
        except Exception:
            return ContainerStatus(
                running=False,
                healthy=False,
                backend_info={"id": self.id},
            )


# ---------------------------------------------------------------------------
# ModalContainerBackend
# ---------------------------------------------------------------------------


class ModalContainerBackend(ContainerBackend):
    """Launch containers as Modal Sandboxes."""

    app_name: str = "cube-container"

    @_retry_sandbox
    def launch(self, spec: ContainerSpec) -> ModalContainer:
        try:
            app = modal.App.lookup(self.app_name, create_if_missing=True)
        except Exception as exc:
            raise ContainerLaunchError(
                f"Failed to look up Modal app '{self.app_name}': {exc}"
            ) from exc

        image = modal.Image.from_registry(spec.image)

        kwargs: dict[str, Any] = {
            "app": app,
            "image": image,
            "timeout": self.timeout_seconds,
        }

        if spec.cpu_cores:
            kwargs["cpu"] = spec.cpu_cores
        if spec.ram_gb:
            kwargs["memory"] = int(spec.ram_gb * 1024)  # Modal takes MB

        if spec.ports:
            kwargs["encrypted_ports"] = spec.ports

        logger.info("Creating Modal sandbox with image %s …", spec.image)
        try:
            sandbox = modal.Sandbox.create(**kwargs)
        except Exception as exc:
            raise ContainerLaunchError(
                f"Failed to create Modal sandbox from '{spec.image}': {exc}"
            ) from exc

        container = ModalContainer(sandbox)
        logger.info("Modal sandbox created: %s", container.id)

        self._run_health_check(container)
        return container
