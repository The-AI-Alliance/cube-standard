"""ModalContainer — ``cube.container.Container`` implementation backed by a Modal Sandbox.

Canonical home for the Modal driver.  ``ModalInfraConfig`` produces these via
``launch()``; the live handle IS a ``ResourceHandle`` and exposes the container
capability surface (exec / port-forward / status).
"""

from __future__ import annotations

import logging
import shlex
import time
from typing import Any, Dict

import modal
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)

from cube.container import (
    Container,
    ContainerError,
    ContainerExecError,
    ContainerStatus,
    ExecResult,
    port_from_url,
)

logger = logging.getLogger(__name__)

_retry_io = retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)


class ModalContainer(Container):
    """Runtime handle backed by a Modal Sandbox."""

    def __init__(self, sandbox: modal.Sandbox) -> None:
        super().__init__()  # populates ResourceHandle fields with defaults
        self._sandbox = sandbox
        self._url_cache: dict[int, str] = {}

    @property
    def id(self) -> str:
        return self._sandbox.object_id

    def __repr__(self) -> str:
        return f"ModalContainer(id={self.id!r}, run_id={self.run_id!r})"

    def exec(
        self,
        command: str,
        timeout: int | None = None,
        workdir: str | None = None,
        env: Dict[str, str] | None = None,
    ) -> ExecResult:
        effective_timeout = timeout if timeout is not None else 120

        wrapped = f"timeout {effective_timeout}s bash -lc {shlex.quote(command)}"

        kwargs: dict[str, Any] = {}
        if workdir:
            kwargs["workdir"] = workdir
        if env:
            kwargs["env"] = env

        start = time.monotonic()
        try:
            process = self._sandbox.exec("bash", "-c", wrapped, **kwargs)
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
        return port_from_url(self.get_url(container_port))

    def get_url(self, container_port: int) -> str:
        if container_port in self._url_cache:
            return self._url_cache[container_port]

        try:
            tunnels = self._sandbox.tunnels()
            if container_port not in tunnels:
                raise ContainerError(f"Port {container_port} has no tunnel. Available: {list(tunnels.keys())}")
            url = tunnels[container_port].url
            self._url_cache[container_port] = url
            return url
        except ContainerError:
            raise
        except Exception as exc:
            raise ContainerError(f"Failed to get tunnel for port {container_port}: {exc}") from exc

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
