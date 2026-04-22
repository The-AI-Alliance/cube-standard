"""ToolkitInfraConfig — InfraConfig serving ``DockerServiceConfig(scope="task")`` as EAI jobs.

Each ``launch()`` creates an ``eai job new -- sleep infinity`` with the resource's image,
polls until RUNNING, and returns a live ``ToolkitContainer`` — which IS a
``ResourceHandle`` and also provides the container interface for the cube's tool layer.

Authentication: the ``eai`` CLI reads its own config (``~/.eai/config``).  Pick the
cluster via ``profile=`` on the config or ``EAI_PROFILE`` env var.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timedelta
from typing import Literal

from cube.container import ContainerLaunchError
from cube.resource import (
    DockerServiceConfig,
    InfraConfig,
    ResourceConfig,
    UnsupportedResourceType,
)
from cube_infra_toolkit.container import ToolkitContainer, _run_eai

logger = logging.getLogger(__name__)


class ToolkitInfraConfig(InfraConfig):
    """Launches per-task Docker containers as EAI Toolkit jobs.

    Serves ``DockerServiceConfig(scope="task")`` — multi-image stacks are rejected.

    Fields:
        profile:                  EAI config profile (overrides ``EAI_PROFILE`` env).
        account:                  EAI account name (optional — defaults to profile's).
        preemptable:              If True, submit as a preemptable job (cheaper; may be
                                  killed mid-run).  Default False for deterministic tests.
        launch_timeout_seconds:   Max wait for the job to reach RUNNING state.
    """

    profile: str | None = None
    account: str | None = None
    preemptable: bool = False
    launch_timeout_seconds: int = 600
    # "sidecar" routes .exec() via an in-container HTTP server (bypasses the
    # eai-exec CLOSE_WAIT hang bug); "direct" keeps every .exec() on the flaky
    # eai job exec RPC.  Sidecar is strongly preferred; direct is a debug/fallback.
    exec_mode: Literal["sidecar", "direct"] = "sidecar"

    # ── InfraConfig interface ─────────────────────────────────────────────────

    def fingerprint(self) -> str:
        prof = self.profile or os.environ.get("EAI_PROFILE") or "default"
        return f"toolkit:{prof}"

    def capabilities(self) -> set[str]:
        return {"docker", "network:egress"}

    def provision(self, resource: ResourceConfig) -> None:
        """Record a ProvisionStore entry.  Toolkit pulls images on-demand at job submit."""
        if not isinstance(resource, DockerServiceConfig):
            raise UnsupportedResourceType(resource, self)
        if len(resource.docker_images) != 1:
            raise ValueError(
                f"ToolkitInfraConfig only supports single-image resources, "
                f"got {len(resource.docker_images)} in {resource.name!r}."
            )

        from cube.provision_store import ProvisionStore

        ProvisionStore().put(resource, self, {"provisioned": True})
        logger.info("Registered %r with ToolkitInfraConfig (no upfront image pull)", resource.name)

    def launch(self, resource: ResourceConfig) -> ToolkitContainer:
        if not isinstance(resource, DockerServiceConfig):
            raise UnsupportedResourceType(resource, self)

        from cube.provision_store import ProvisionStore

        if ProvisionStore().get(resource, self) is None:
            self.provision(resource)

        # Resolve the profile once — the eai CLI's EAI_PROFILE env var is ignored
        # by some subcommands (notably `job exec`), so we always pass --profile.
        profile = self.profile or os.environ.get("EAI_PROFILE")

        image = resource.docker_images[0]
        # DockerServiceConfig doesn't carry cpu/ram at launch time — stick to sensible
        # defaults matching DaytonaInfraConfig.  Cubes that need different resources
        # should override in a subclass or declare via a DockerServiceConfig extension.
        cpu, mem_gb = 2, 4

        cmd: list[str] = ["job", "new"]
        if self.preemptable:
            cmd.append("--preemptable")
        else:
            cmd.append("--non-preemptable")
        # --tunnel enables the sidecar gateway used by `eai job port-forward`,
        # which ToolkitContainer's sidecar exec_mode relies on.
        cmd += ["--tunnel"]
        cmd += ["--format", "json", "--no-header"]
        cmd += ["-i", image]
        cmd += ["--cpu", str(cpu)]
        cmd += ["--mem", str(mem_gb)]
        cmd += ["--", "sleep", "infinity"]

        logger.info("Submitting EAI job for %r (image=%s)…", resource.name, image)
        # retries=0: `eai job new` is not idempotent — a timeout mid-creation
        # may have actually created the job, and a retry would produce a duplicate.
        result = _run_eai(
            cmd, profile=profile, account=self.account, timeout=self.launch_timeout_seconds, retries=0,
        )
        if result.returncode != 0:
            raise ContainerLaunchError(f"Failed to submit EAI job: {result.stderr.strip()}")

        try:
            payload = json.loads(result.stdout)
            job_id = payload["id"]
        except (json.JSONDecodeError, KeyError) as exc:
            raise ContainerLaunchError(f"Could not parse job id from eai output: {result.stdout!r}") from exc

        logger.info("EAI job %s submitted — waiting for RUNNING…", job_id)
        _wait_for_running(job_id, profile=profile, account=self.account, timeout=self.launch_timeout_seconds)

        container = ToolkitContainer(
            job_id, profile=profile, account=self.account, exec_mode=self.exec_mode,
        )
        logger.info("EAI job %s RUNNING", job_id)

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
        container.expires_at = (
            container.created_at + timedelta(seconds=effective_ttl) if effective_ttl else None
        )
        return container

    def list_active(self, run_id: str | None = None) -> list[ToolkitContainer]:
        """Not implemented — EAI jobs aren't tagged with our run_id today."""
        return []

    def cleanup(self, run_id: str) -> None:
        """No-op — see ``list_active``."""
        logger.debug("ToolkitInfraConfig.cleanup(%s): no-op (labels not wired yet)", run_id)

    def cleanup_stale(self, max_age_seconds: int | None = None) -> list[str]:
        """No-op — EAI job TTLs are managed cluster-side."""
        return []


# ── helpers ─────────────────────────────────────────────────────────────────


def _wait_for_running(
    job_id: str,
    *,
    profile: str | None,
    account: str | None,
    timeout: int,
) -> None:
    deadline = time.monotonic() + timeout
    last_state = ""
    while time.monotonic() < deadline:
        result = _run_eai(
            ["job", "get", job_id, "--format", "json", "--no-header"],
            profile=profile,
            account=account,
            timeout=30,
        )
        try:
            info = json.loads(result.stdout)
            state = str(info.get("state", "")).lower()
        except (json.JSONDecodeError, AttributeError):
            state = ""

        if state == "running":
            return
        if state in ("failed", "cancelled", "killed"):
            _run_eai(["job", "kill", job_id], profile=profile, account=account, timeout=30)
            raise ContainerLaunchError(f"EAI job {job_id} entered terminal state: {state}")

        last_state = state
        logger.info("EAI job %s state=%s, waiting…", job_id[:8], state)
        time.sleep(5)

    _run_eai(["job", "kill", job_id], profile=profile, account=account, timeout=30)
    raise ContainerLaunchError(
        f"EAI job {job_id} did not reach RUNNING within {timeout}s (last state: {last_state!r})"
    )
