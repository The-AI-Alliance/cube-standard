"""ModalInfraConfig — InfraConfig serving ``DockerServiceConfig(scope="task")`` as Modal Sandboxes.

Each ``launch()`` creates a Modal Sandbox from the resource's image and returns a
live ``ModalContainer`` — which IS a ``ResourceHandle`` and exposes the container
interface for the cube's tool layer.

Authentication: Modal reads credentials from ``~/.modal.toml`` (set by ``modal setup``)
or ``MODAL_TOKEN_ID`` / ``MODAL_TOKEN_SECRET`` env vars.  Credentials are never
stored on the config itself (would be serialised across process boundaries).
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta
from typing import Any

import modal

from cube.container import ContainerLaunchError
from cube.resource import (
    DockerServiceConfig,
    InfraConfig,
    ResourceConfig,
    UnsupportedResourceType,
)
from cube_infra_modal.container import ModalContainer

logger = logging.getLogger(__name__)


class ModalInfraConfig(InfraConfig):
    """Launches per-task Docker containers as Modal Sandboxes.

    Serves ``DockerServiceConfig(scope="task")`` only — multi-image stacks are
    rejected.  Modal Sandboxes are ephemeral by default and auto-terminate when
    the owning app is closed, so ``cleanup_stale`` is intentionally a no-op.

    Fields:
        app_name:        Modal app to attach sandboxes to.  Re-used across runs.
        timeout_seconds: Max sandbox runtime before Modal auto-terminates.
                         Default 1 hour; raise for longer evaluations.
    """

    app_name: str = "cube-container"
    timeout_seconds: int = 3600

    # ── InfraConfig interface ─────────────────────────────────────────────────

    def fingerprint(self) -> str:
        return f"modal:{self.app_name}"

    def capabilities(self) -> set[str]:
        return {"docker", "network:egress", "gpu:nvidia", "container:root"}

    def provision(self, resource: ResourceConfig) -> None:
        """Record a ProvisionStore entry.  Modal pulls images on-demand at Sandbox creation."""
        if not isinstance(resource, DockerServiceConfig):
            raise UnsupportedResourceType(resource, self)
        if len(resource.docker_images) != 1:
            raise ValueError(
                f"ModalInfraConfig only supports single-image resources, "
                f"got {len(resource.docker_images)} in {resource.name!r}."
            )

        from cube.provision_store import ProvisionStore

        ProvisionStore().put(resource, self, {"provisioned": True})
        logger.info("Registered %r with ModalInfraConfig (no upfront image pull)", resource.name)

    def launch(self, resource: ResourceConfig) -> ModalContainer:
        if not isinstance(resource, DockerServiceConfig):
            raise UnsupportedResourceType(resource, self)

        from cube.provision_store import ProvisionStore

        if ProvisionStore().get(resource, self) is None:
            self.provision(resource)

        image_ref = resource.docker_images[0]

        try:
            app = modal.App.lookup(self.app_name, create_if_missing=True)
        except Exception as exc:
            raise ContainerLaunchError(f"Failed to look up Modal app {self.app_name!r}: {exc}") from exc

        image = modal.Image.from_registry(image_ref)
        kwargs: dict[str, Any] = {
            "app": app,
            "image": image,
            "timeout": self.timeout_seconds,
            # Conservative defaults mirroring DaytonaInfraConfig — 2 CPU / 4 GiB.
            "cpu": 2,
            "memory": 4 * 1024,
        }

        logger.info("Creating Modal sandbox for %r (image=%s)…", resource.name, image_ref)
        try:
            sandbox = modal.Sandbox.create(**kwargs)
        except Exception as exc:
            raise ContainerLaunchError(f"Failed to create Modal sandbox from {image_ref!r}: {exc}") from exc

        container = ModalContainer(sandbox)
        logger.info("Modal sandbox live: %s (%s)", container.id, resource.name)

        # Populate ResourceHandle bookkeeping on the container itself.
        effective_ttl = (
            self.default_ttl_seconds if self.default_ttl_seconds is not None else resource.default_ttl_seconds
        )
        container.run_id = str(uuid.uuid4())
        container.resource = resource
        container.infra = self
        container.endpoint = None
        container.endpoints = {}
        container.created_at = datetime.now()
        container.expires_at = container.created_at + timedelta(seconds=effective_ttl) if effective_ttl else None
        return container

    def list_active(self, run_id: str | None = None) -> list[ModalContainer]:
        """Not implemented — Modal sandboxes aren't tagged with our run_id today."""
        return []

    def cleanup(self, run_id: str) -> None:
        """No-op — see ``list_active``."""
        logger.debug("ModalInfraConfig.cleanup(%s): no-op (labels not wired yet)", run_id)

    def cleanup_stale(self, max_age_seconds: int | None = None) -> list[str]:
        """No-op — Modal Sandboxes auto-terminate on app teardown / timeout."""
        return []
