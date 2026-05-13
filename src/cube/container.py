"""Container — a ``ResourceHandle`` that adds exec / port-forwarding.

A ``Container`` IS a ``ResourceHandle``.  ``InfraConfig.launch()`` returns one
directly — no wrapper indirection.  Subclasses carry the handle bookkeeping
(run_id / resource / infra / created_at / expires_at) alongside their driver-
specific state.

``ContainerBackend`` and ``ContainerConfig`` below are the pre-``InfraConfig``
factory + config.  They are **deprecated** — use ``InfraConfig.launch()`` with
``DockerImageConfig`` or ``DockerServiceConfig`` (in ``cube.resource``) instead.
Kept here only so existing ``Task.container_backend`` / ``Benchmark.container_backend``
fields continue to type-check during the migration.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict

from pydantic import Field

from cube.core import TypedBaseModel
from cube.resource import ResourceHandle

logger = logging.getLogger(__name__)


@dataclass
class ExecResult:
    """Result of running a command inside a container."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    duration_seconds: float = 0.0


@dataclass
class ContainerStatus:
    """Snapshot of container health / resource usage."""

    running: bool = False
    healthy: bool = False
    resource_usage: dict[str, float] = field(default_factory=dict)
    backend_info: dict[str, Any] = field(default_factory=dict)


class ContainerError(Exception):
    """Base exception for all container-related errors."""


class ContainerLaunchError(ContainerError):
    """Raised when a container fails to start."""


class HealthCheckError(ContainerError):
    """Raised when a container's health check fails."""


class ContainerExecError(ContainerError):
    """Raised when command execution inside a container fails."""


class Container(ResourceHandle, ABC):
    """Live handle to a running container.

    Inherits ``ResourceHandle`` bookkeeping (run_id, resource, infra, created_at,
    expires_at, endpoint, endpoints) and adds the container-specific capability
    surface: exec / port-forward / status / id.

    ``ResourceHandle.close()`` is satisfied by delegating to ``stop()`` so callers
    can treat the handle uniformly via the ``with`` statement or ``close()``.
    """

    @abstractmethod
    def exec(
        self,
        command: str,
        timeout: int | None = None,
        workdir: str | None = None,
        env: Dict[str, str] | None = None,
    ) -> ExecResult:
        """Execute *command* inside the container."""

    @abstractmethod
    def forward_port(self, container_port: int) -> int:
        """Return the reachable port for *container_port* on this backend."""

    @abstractmethod
    def get_url(self, container_port: int) -> str:
        """Return a URL that reaches *container_port*."""

    @abstractmethod
    def stop(self, timeout: int = 10) -> None:
        """Stop (and optionally remove) the container. Idempotent."""

    @abstractmethod
    def get_status(self) -> ContainerStatus:
        """Return current container status."""

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique, backend-specific container identifier."""

    def close(self) -> None:
        """Delegates to ``stop()`` — satisfies ``ResourceHandle.close()``."""
        self.stop()

    @property
    def container(self) -> "Container":
        """Return self — Container IS ResourceHandle, no wrapper.

        Kept for uniformity with legacy handle types (e.g.
        ``LocalDockerServiceHandle.container``) that wrap multiple containers
        and need an indirection property.
        """
        return self


def relocate_if_readonly(
    container: Container,
    working_dir: str,
    new_wd: str,
    *,
    extra_setup: str | None = None,
) -> str:
    """Copy *working_dir* to *new_wd* if it isn't writable by the runtime user.

    Returns the effective working directory (either the original if writable, or
    *new_wd* after the copy).  Cubes that need additional setup after the copy
    (e.g. ``git config safe.directory``) can pass the commands as *extra_setup*
    — they are appended to the ``cp -a`` invocation with ``&&``.

    Typical usage in a cube's ``_build_tool()``::

        new_wd = relocate_if_readonly(
            self._container, self.tool_config.working_dir, "/tmp/testbed",
            extra_setup="git config --global --add safe.directory /tmp/testbed",
        )
        self._tool = self.tool_config.model_copy(update={"working_dir": new_wd}).make(
            container=self._container
        )
    """
    probe = container.exec(f"test -w {working_dir} && echo W || echo R", timeout=30)
    if "W" in probe.stdout:
        return working_dir
    logger.info("%s not writable by runtime user — copying to %s", working_dir, new_wd)
    cmd = f"cp -a {working_dir} {new_wd}"
    if extra_setup:
        cmd += f" && {extra_setup}"
    container.exec(cmd, timeout=300)
    return new_wd


def port_from_url(url: str) -> int:
    """Extract the effective port from a URL (443 for https, 80 for http)."""
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.port is not None:
        return parsed.port
    if parsed.scheme == "https":
        return 443
    if parsed.scheme == "http":
        return 80
    raise ContainerError(f"Could not determine port from URL: {url}")


# ── Deprecated pre-InfraConfig API ───────────────────────────────────────────


class ContainerConfig(TypedBaseModel):
    """DEPRECATED.  Use ``cube.resource.DockerImageConfig`` instead.

    Kept only so existing ``TaskMetadata.container_config`` fields continue to
    type-check during the infra migration.  No per-instance warning — would be
    noisy across CSV task-metadata loads.
    """

    image: str
    ram_gb: float = 4.0
    cpu_cores: float = 2.0
    gpu: bool = False
    disk_gb: float = 10.0
    ports: list[int] | None = None


class ContainerBackend(TypedBaseModel, ABC):
    """DEPRECATED.  Use ``cube.resource.InfraConfig`` + ``InfraConfig.launch(resource)`` instead.

    Kept only so existing ``Task.container_backend`` / ``Benchmark.container_backend``
    fields continue to type-check during the infra migration.  Subclasses emit a
    one-shot ``DeprecationWarning`` at *import* time (see ``cube.backends.*``),
    not per instantiation.
    """

    timeout_seconds: int = 1800
    backend_config: Dict[str, Any] = Field(default_factory=dict)

    @abstractmethod
    def launch(self, config: ContainerConfig) -> Container:
        """Launch a container described by *config*. Blocks until ready."""

    def health_check(self, container: Container) -> bool:
        """Override to perform custom health checks. Return True if healthy."""
        return True

    def _run_health_check(self, container: Container) -> None:
        """Run the health check, cleaning up on failure."""
        try:
            ok = self.health_check(container)
            if not ok:
                container.stop()
                raise HealthCheckError("Health check returned False")
        except HealthCheckError:
            raise
        except Exception as exc:
            container.stop()
            raise HealthCheckError(f"Health check raised an exception: {exc}") from exc
