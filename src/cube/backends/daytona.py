"""DEPRECATED — use ``cube_infra_daytona.DaytonaInfraConfig`` instead.

This module is kept only to serve the legacy ``ContainerBackend`` API and
existing callers (examples, tests, older recipes).  The canonical
``DaytonaContainer`` driver now lives in
``cube_infra_daytona.container``.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

from cube_infra_daytona.container import DaytonaContainer, retry_sandbox
from daytona import (
    CreateSandboxFromImageParams,
    Daytona,
    DaytonaConfig,
    DaytonaError,
    Image,
    Resources,
)

from cube.container import (
    ContainerBackend,
    ContainerConfig,
    ContainerLaunchError,
)

warnings.warn(
    "cube.backends.daytona is deprecated — use "
    "cube_infra_daytona.DaytonaInfraConfig for new code. "
    "The DaytonaContainer driver has moved to cube_infra_daytona.container.",
    DeprecationWarning,
    stacklevel=2,
)

logger = logging.getLogger(__name__)

__all__ = ["DaytonaContainer", "DaytonaContainerBackend"]


class DaytonaContainerBackend(ContainerBackend):
    """DEPRECATED.  Launches containers as Daytona sandboxes via the legacy API."""

    api_key: str | None = None
    api_url: str | None = None
    target: str | None = None
    ephemeral: bool = True
    auto_stop_minutes: int = 10
    auto_delete_minutes: int = 5

    def launch(self, config: ContainerConfig) -> DaytonaContainer:
        return self._launch_with_retry(config)

    @retry_sandbox
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
