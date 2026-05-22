"""DaytonaInfraConfig — InfraConfig serving ``DockerServiceConfig(scope="task")``.

Each call to ``launch()`` creates a fresh Daytona sandbox from the resource's image
and returns a live ``DaytonaContainer`` — which IS a ``ResourceHandle`` (closes the
sandbox via ``close()``) and also provides ``exec()`` / ``forward_port()`` for the
cube's tool layer to drive.

Authentication: reads ``DAYTONA_API_KEY``, ``DAYTONA_API_URL``, ``DAYTONA_TARGET`` from
env if not provided as fields.  Credentials are NEVER stored on the config itself
(would be serialised across process boundaries).
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from datetime import datetime, timedelta
from typing import Any

from daytona import (
    CreateSandboxFromImageParams,
    Daytona,
    DaytonaConfig,
    Image,
    Resources,
)

from cube.container import ContainerLaunchError
from cube.resource import (
    DockerServiceConfig,
    InfraConfig,
    ResourceConfig,
    UnsupportedResourceType,
)
from cube_infra_daytona.container import DaytonaContainer

logger = logging.getLogger(__name__)


class DaytonaInfraConfig(InfraConfig):
    """Launches per-task Docker containers as Daytona sandboxes.

    Serves ``DockerServiceConfig(scope="task")`` by creating a fresh sandbox with the
    resource's single image.  Multi-image stacks and volumes are not supported — those
    belong on cloud infras (AWS/Azure) via DockerServiceConfig(scope="benchmark").

    Fields:
        api_key / api_url / target:    Optional — override env-var credentials.
        ephemeral:                     If True (default), sandboxes are deleted after
                                       ``auto_delete_minutes``.  Set False only for
                                       long-lived debug sandboxes.
        auto_stop_minutes:             Idle minutes before Daytona auto-stops.
        auto_delete_minutes:           Minutes post-stop before Daytona auto-deletes
                                       (only if ``ephemeral=False``).
        launch_timeout_seconds:        How long to wait for sandbox creation.
    """

    ephemeral: bool = True
    auto_stop_minutes: int = 10
    auto_delete_minutes: int = 5
    launch_timeout_seconds: int = 180

    # ── InfraConfig interface ─────────────────────────────────────────────────

    def fingerprint(self) -> str:
        target = os.environ.get("DAYTONA_TARGET", "us")
        return f"daytona:{target}"

    def capabilities(self) -> set[str]:
        return {"docker", "network:egress", "container:root"}

    def provision(self, resource: ResourceConfig) -> None:
        """Validate resource shape and record a ProvisionStore entry.

        Daytona pulls images on-demand at sandbox creation, so there is no upfront
        pull step for us to run.  We still write to the ProvisionStore so ``launch()``
        doesn't raise ``ResourceNotReadyError``.
        """
        if not isinstance(resource, DockerServiceConfig):
            raise UnsupportedResourceType(resource, self)
        if len(resource.docker_images) != 1:
            raise ValueError(
                f"DaytonaInfraConfig only supports single-image resources, "
                f"got {len(resource.docker_images)} in {resource.name!r}."
            )

        from cube.provision_store import ProvisionStore

        ProvisionStore().put(resource, self, {"provisioned": True})
        logger.info("Registered %r with DaytonaInfraConfig (no upfront image pull)", resource.name)

    def launch(self, resource: ResourceConfig) -> DaytonaContainer:
        if not isinstance(resource, DockerServiceConfig):
            raise UnsupportedResourceType(resource, self)

        from cube.provision_store import ProvisionStore

        # Idempotently provision so first-launch doesn't trip ResourceNotReadyError.
        if ProvisionStore().get(resource, self) is None:
            self.provision(resource)

        client = _make_client()

        image = resource.docker_images[0]
        logger.info("Creating Daytona sandbox for %r (image=%s)…", resource.name, image)

        # Resources come from DockerServiceConfig defaults — 2 CPU / 4 GiB / 10 GiB.
        # Daytona wants ints.  Callers can subclass DockerServiceConfig to override.
        resources_kwargs: dict[str, Any] = {"cpu": 2, "memory": 4, "disk": 10}

        create_kwargs: dict[str, Any] = {
            "image": Image.base(image),
            "resources": Resources(**resources_kwargs),
            "auto_stop_interval": self.auto_stop_minutes,
            "ephemeral": self.ephemeral,
            # Leave outbound network open — cubes like terminal-bench run
            # task-level ``test.sh`` scripts that pull tools (uv, pytest) from
            # the public internet during evaluate().  Explicit allow-lists
            # would need per-task tuning.
            "network_block_all": False,
        }
        if not self.ephemeral:
            create_kwargs["auto_delete_interval"] = self.auto_delete_minutes

        params = CreateSandboxFromImageParams(**create_kwargs)
        # Retry on quota errors (parallel launches can transiently exceed tier
        # limits while prior sandboxes are still shutting down).
        for _attempt in range(3):
            try:
                sandbox = client.create(params, timeout=self.launch_timeout_seconds)
                break
            except Exception as exc:
                msg = str(exc).lower()
                if "memory limit exceeded" in msg or "quota" in msg or "limit exceeded" in msg:
                    logger.warning("Daytona quota hit on attempt %d/3 — waiting 30s: %s", _attempt + 1, exc)
                    time.sleep(30)
                    if _attempt == 2:
                        raise ContainerLaunchError(f"Daytona quota exceeded after 3 attempts: {exc}") from exc
                else:
                    raise

        # Wrap construction + bookkeeping so any failure after sandbox creation
        # deletes the sandbox — DaytonaContainer.__init__ calls create_session()
        # which can raise on transient API errors.
        try:
            container = DaytonaContainer(sandbox, client)
            logger.info("Daytona sandbox live: %s (%s)", container.id, resource.name)

            # Populate ResourceHandle bookkeeping on the container itself — a
            # DaytonaContainer IS a ResourceHandle (no wrapper).
            effective_ttl = (
                self.default_ttl_seconds if self.default_ttl_seconds is not None else resource.default_ttl_seconds
            )
            container.run_id = str(uuid.uuid4())
            container.resource = resource
            container.infra = self
            container.endpoint = None  # Daytona sandboxes don't expose an eager endpoint
            container.endpoints = {}
            container.created_at = datetime.now()
            container.expires_at = container.created_at + timedelta(seconds=effective_ttl) if effective_ttl else None
            return container
        except Exception:
            logger.exception("DaytonaContainer setup failed; deleting sandbox %s", getattr(sandbox, "id", "<unknown>"))
            try:
                client.delete(sandbox)
            except Exception as delete_exc:
                logger.warning("Failed to delete Daytona sandbox during cleanup: %s", delete_exc)
            raise

    def list_active(self, run_id: str | None = None) -> list[DaytonaContainer]:
        """Not implemented — Daytona's API doesn't tag sandboxes with our run_id today.

        Returns an empty list so ``cleanup_stale()`` won't try to iterate.  When we
        need durable cross-process cleanup we'll adopt Daytona labels.
        """
        return []

    def cleanup(self, run_id: str) -> None:
        """No-op — see ``list_active``."""
        logger.debug("DaytonaInfraConfig.cleanup(%s): no-op (labels not wired yet)", run_id)

    def cleanup_stale(self, max_age_seconds: int | None = None) -> list[str]:
        """No-op — ephemeral sandboxes self-destruct via Daytona's auto-stop/auto-delete."""
        return []


def _make_client() -> Daytona:
    """Build a Daytona client, letting the SDK pull credentials from env if not passed."""
    api_key = os.environ.get("DAYTONA_API_KEY")
    api_url = os.environ.get("DAYTONA_API_URL")
    target = os.environ.get("DAYTONA_TARGET")

    config_kwargs: dict[str, Any] = {}
    if api_key:
        config_kwargs["api_key"] = api_key
    if api_url:
        config_kwargs["api_url"] = api_url
    if target:
        config_kwargs["target"] = target

    return Daytona(DaytonaConfig(**config_kwargs) if config_kwargs else DaytonaConfig())
