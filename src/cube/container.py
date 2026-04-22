from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict

from pydantic import Field

from cube.core import TypedBaseModel


class ContainerConfig(TypedBaseModel):
    """Declarative description of *what* to run — owned by the benchmark."""

    image: str
    ram_gb: float = 4.0
    cpu_cores: float = 2.0
    gpu: bool = False
    disk_gb: float = 10.0
    ports: list[int] | None = None


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


class Container(ABC):
    """Runtime handle to a running container (not serializable)."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} id={self.id}>"

    @abstractmethod
    def exec(
        self,
        command: str,
        timeout: int | None = None,
        workdir: str | None = None,
        env: Dict[str, str] | None = None,
    ) -> ExecResult:
        """Execute *command* inside the container."""

    def exec_long_running(
        self,
        command: str,
        *,
        timeout: int,
        poll_interval: int = 30,
        workdir: str | None = None,
        env: Dict[str, str] | None = None,
    ) -> ExecResult:
        """Execute a long-running *command*, decoupled from any RPC channel.

        Default implementation is just ``exec(command, timeout=timeout)`` —
        fine for backends with reliable exec streaming (LocalContainer,
        DaytonaContainer, ModalContainer).  Backends whose exec primitive
        has reliability issues on long-running commands (notably
        ToolkitContainer — see `docs/toolkit-hang-bugreport.md`) override
        this to kick off the command in the background inside the
        container and poll a sentinel file for completion.  Each individual
        RPC call under this override is short (< 1 s), so a transient CLI
        hang can be retried cheaply instead of re-running a 30-minute
        pytest.
        """
        return self.exec(command, timeout=timeout, workdir=workdir, env=env)

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


class ContainerBackend(TypedBaseModel, ABC):
    """Serializable configuration for *how* to run containers."""

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
