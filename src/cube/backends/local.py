"""Local Docker backend — uses docker-py to run containers on the host."""

from __future__ import annotations

import logging
import shlex
import time
from typing import Any, Dict

import docker
import docker.errors
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

# ---------------------------------------------------------------------------
# Retry decorators
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# LocalContainer
# ---------------------------------------------------------------------------


class LocalContainer(Container):
    """Runtime handle backed by a local Docker container."""

    def __init__(
        self,
        docker_container: docker.models.containers.Container,
        client: docker.DockerClient,
        remove_on_close: bool = True,
    ) -> None:
        self._container = docker_container
        self._client = client
        self._remove_on_close = remove_on_close
        self._port_map: dict[int, int] = {}

    # -- Container interface ------------------------------------------------

    @property
    def id(self) -> str:
        return self._container.id

    def exec(
        self,
        command: str,
        timeout: int | None = None,
        workdir: str | None = None,
        env: Dict[str, str] | None = None,
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
            raise ContainerExecError(
                f"Docker exec failed: {exc}"
            ) from exc
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
            raise ContainerError(
                f"Port {container_port} is not exposed. "
                f"Available ports: {list(ports.keys())}"
            )
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
            pass  # already gone — idempotent
        except docker.errors.APIError:
            pass  # container may already be stopped

        if self._remove_on_close:
            try:
                self._container.remove(force=True)
            except docker.errors.NotFound:
                pass

    def get_status(self) -> ContainerStatus:
        try:
            self._container.reload()
            state = self._container.status  # "running", "exited", …
            running = state == "running"

            resource_usage: dict[str, float] = {}
            if running:
                try:
                    stats = self._container.stats(stream=False)
                    mem = stats.get("memory_stats", {})
                    resource_usage["memory_bytes"] = float(
                        mem.get("usage", 0)
                    )
                    resource_usage["memory_limit_bytes"] = float(
                        mem.get("limit", 0)
                    )
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


# ---------------------------------------------------------------------------
# LocalContainerBackend
# ---------------------------------------------------------------------------


class LocalContainerBackend(ContainerBackend):
    """Launch containers on the local Docker daemon."""

    pull_policy: str = "missing"
    network_mode: str = "bridge"
    remove_on_close: bool = True

    @_retry_launch
    def launch(self, spec: ContainerSpec) -> LocalContainer:
        client = docker.from_env()

        # -- pull image ---------------------------------------------------
        if self.pull_policy == "always" or (
            self.pull_policy == "missing"
            and not _image_exists(client, spec.image)
        ):
            logger.info("Pulling image %s …", spec.image)
            try:
                client.images.pull(spec.image)
            except docker.errors.APIError as exc:
                raise ContainerLaunchError(
                    f"Failed to pull image '{spec.image}': {exc}"
                ) from exc

        # -- resource limits ----------------------------------------------
        kwargs: dict[str, Any] = {}
        kwargs["mem_limit"] = f"{int(spec.ram_gb * 1024)}m"
        kwargs["nano_cpus"] = int(spec.cpu_cores * 1e9)

        if spec.gpu:
            kwargs["device_requests"] = [
                docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
            ]

        # -- port bindings ------------------------------------------------
        if spec.ports:
            kwargs["ports"] = {f"{p}/tcp": None for p in spec.ports}

        # -- create & start -----------------------------------------------
        try:
            docker_container = client.containers.run(
                spec.image,
                command="sleep infinity",
                detach=True,
                network_mode=self.network_mode,
                **kwargs,
            )
        except docker.errors.APIError as exc:
            raise ContainerLaunchError(
                f"Failed to create container from '{spec.image}': {exc}"
            ) from exc

        # -- wait until running -------------------------------------------
        container = LocalContainer(
            docker_container, client, remove_on_close=self.remove_on_close
        )
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            docker_container.reload()
            if docker_container.status == "running":
                break
            time.sleep(0.5)
        else:
            container.stop()
            raise ContainerLaunchError(
                f"Container did not reach 'running' state within "
                f"{self.timeout_seconds}s (status: {docker_container.status})"
            )

        self._run_health_check(container)
        return container


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _image_exists(client: docker.DockerClient, image: str) -> bool:
    try:
        client.images.get(image)
        return True
    except docker.errors.ImageNotFound:
        return False
