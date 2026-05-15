"""LocalContainer — a live ``Container`` handle backed by a local Docker container.

This is a driver, not a provisioning factory.  It is constructed by
``LocalInfraConfig`` (see ``cube.infra_local``) once a container is running and
exposed to the tool layer via ``ResourceHandle.container``.
"""

from __future__ import annotations

import logging
import shlex
import time

import docker
import docker.errors
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
)

logger = logging.getLogger(__name__)

_retry_io = retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)


class LocalContainer(Container):
    """Runtime handle backed by a local Docker container."""

    def __init__(
        self,
        docker_container: docker.models.containers.Container,
        client: docker.DockerClient,
        remove_on_close: bool = True,
    ) -> None:
        super().__init__()  # populates ResourceHandle fields with defaults
        self._container = docker_container
        self._client = client
        self._remove_on_close = remove_on_close
        self._port_map: dict[int, int] = {}

    @property
    def id(self) -> str:
        return self._container.id

    def exec(
        self,
        command: str,
        timeout: int | None = None,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        if timeout is not None:
            wrapped = f"timeout {timeout}s sh -lc {shlex.quote(command)}"
        else:
            wrapped = f"sh -lc {shlex.quote(command)}"

        start = time.monotonic()
        try:
            exit_code, output = self._container.exec_run(
                wrapped,
                demux=True,
                workdir=workdir,
                environment=env,
            )
        except docker.errors.APIError as exc:
            raise ContainerExecError(f"Docker exec failed: {exc}") from exc
        duration = time.monotonic() - start

        stdout = (output[0] or b"").decode("utf-8", errors="replace")
        stderr = (output[1] or b"").decode("utf-8", errors="replace")

        return ExecResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            duration_seconds=round(duration, 3),
        )

    def forward_port(self, container_port: int) -> int:
        if container_port in self._port_map:
            return self._port_map[container_port]

        self._container.reload()
        ports = self._container.ports
        key = f"{container_port}/tcp"
        if key not in ports or not ports[key]:
            raise ContainerError(f"Port {container_port} is not exposed. Available ports: {list(ports.keys())}")
        host_port = int(ports[key][0]["HostPort"])
        self._port_map[container_port] = host_port
        return host_port

    def get_url(self, container_port: int) -> str:
        host_port = self.forward_port(container_port)
        return f"http://localhost:{host_port}"

    @_retry_io
    def stop(self, timeout: int = 10) -> None:
        try:
            self._container.stop(timeout=timeout)
        except docker.errors.NotFound:
            pass
        except docker.errors.APIError:
            pass

        if self._remove_on_close:
            try:
                self._container.remove(force=True)
            except docker.errors.NotFound:
                pass

    def get_status(self) -> ContainerStatus:
        try:
            self._container.reload()
            state = self._container.status
            running = state == "running"

            resource_usage: dict[str, float] = {}
            if running:
                try:
                    stats = self._container.stats(stream=False)
                    mem = stats.get("memory_stats", {})
                    resource_usage["memory_bytes"] = float(mem.get("usage", 0))
                    resource_usage["memory_limit_bytes"] = float(mem.get("limit", 0))
                except Exception:
                    pass

            return ContainerStatus(
                running=running,
                healthy=running,
                resource_usage=resource_usage,
                backend_info={"docker_status": state, "id": self.id},
            )
        except docker.errors.NotFound:
            return ContainerStatus(
                running=False,
                healthy=False,
                backend_info={"docker_status": "removed", "id": self.id},
            )
