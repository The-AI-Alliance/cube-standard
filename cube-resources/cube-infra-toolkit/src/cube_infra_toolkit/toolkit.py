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
import secrets
import time
import uuid
from datetime import datetime, timedelta

from cube.container import ContainerLaunchError
from cube.resource import (
    DockerServiceConfig,
    InfraConfig,
    ResourceConfig,
    UnsupportedResourceType,
)
from cube_infra_toolkit.container import ToolkitContainer, _run_eai, relay_startup_args

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
    # Override when eai is not on PATH (e.g. installed in ~/bin via .zshrc).
    eai_path: str = "eai"
    # EAI data full name to mount as the exec-relay sidecar binary.  The data
    # must contain a single file named ``cube-sidecar`` at its root; it is
    # mounted read-only at ``/opt/cube-sidecar/``.  With the binary mounted,
    # the relay works on any image — no ``python3`` required in the container.
    #
    # Default points at the maintainer's personal account (``snow.allac``);
    # that's the only place we can publish today — ``snow.shared`` is admin-
    # locked.  Tracking ticket to migrate to a world-readable shared location:
    # TODO once admin approves the write grant.
    #
    # Republish with ``scripts/publish-cube-sidecar.sh`` after rebuilding the
    # Go binary in ``sidecar-go/``.
    sidecar_data: str | None = "snow.allac.cube_sidecar"
    # Optional EAI data full name mounted read-only at ``/opt/cube-assets/``
    # for cube-side helper binaries (e.g. ``uv``) that the harness can copy
    # into the container at runtime when the image lacks them.  Cubes consult
    # this path in their evaluator setup; unset → no asset mount.  Same
    # account caveat as ``sidecar_data``.
    assets_data: str | None = "snow.allac.cube_uv"

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

        # Generate a relay token now so we can embed it in the job startup command.
        # The relay server starts with the job — zero bootstrap eai execs.
        relay_token = secrets.token_urlsafe(32)

        cmd: list[str] = ["job", "new"]
        if self.preemptable:
            cmd.append("--preemptable")
        else:
            cmd.append("--non-preemptable")
        # --tunnel enables the port-forward gateway used by ToolkitContainer.
        cmd += ["--tunnel"]
        cmd += ["--format", "json", "--no-header"]
        cmd += ["-i", image]
        cmd += ["--cpu", str(cpu)]
        cmd += ["--mem", str(mem_gb)]
        if self.sidecar_data is not None:
            cmd += ["--data", f"{self.sidecar_data}:/opt/cube-sidecar:ro"]
        if self.assets_data is not None:
            cmd += ["--data", f"{self.assets_data}:/opt/cube-assets:ro"]
        # Embed relay startup + token into the job command; relay is up before
        # port-forward is established — no bootstrap eai execs needed.
        cmd += ["--"] + relay_startup_args(relay_token)

        logger.info("Submitting EAI job for %r (image=%s)…", resource.name, image)
        submit_started_at = datetime.now()
        # retries=0: `eai job new` is not idempotent — a timeout mid-creation
        # may have actually created the job, and a retry would produce a duplicate.
        try:
            result = _run_eai(
                cmd,
                eai_path=self.eai_path,
                profile=profile,
                account=self.account,
                timeout=self.launch_timeout_seconds,
                retries=0,
            )
        except Exception as exc:
            # _run_eai timed out or failed before returning. The job MAY have been
            # created server-side — we have no ID to kill it. Log loudly so ops
            # can find orphans via `eai job ls --mine` matching the submit window.
            logger.error(
                "EAI job submission failed at %s (profile=%s, account=%s, image=%s): %s. "
                "If a job was accepted server-side, it will orphan — check `eai job ls --mine` "
                "for jobs submitted around this timestamp and kill manually.",
                submit_started_at.isoformat(),
                profile,
                self.account,
                image,
                exc,
            )
            raise ContainerLaunchError(f"Failed to submit EAI job: {exc}") from exc

        if result.returncode != 0:
            # Non-zero from eai generally means the CLI rejected before submission,
            # but we can't be certain — warn so the window is auditable.
            logger.warning(
                "eai job new returned rc=%d at %s (stderr: %s). Assuming no job created; "
                "if an orphan appears, check `eai job ls --mine` for submissions around this time.",
                result.returncode,
                submit_started_at.isoformat(),
                result.stderr.strip(),
            )
            raise ContainerLaunchError(f"Failed to submit EAI job: {result.stderr.strip()}")

        try:
            payload = json.loads(result.stdout)
            job_id = payload["id"]
        except (json.JSONDecodeError, KeyError) as exc:
            logger.error(
                "eai job new returned rc=0 but payload was unparseable at %s: stdout=%r. "
                "A job was likely created — check `eai job ls --mine` and kill manually.",
                submit_started_at.isoformat(),
                result.stdout,
            )
            raise ContainerLaunchError(f"Could not parse job id from eai output: {result.stdout!r}") from exc

        # Once we have a job_id, any downstream failure must kill the job.
        try:
            logger.info("EAI job %s submitted — waiting for RUNNING…", job_id)
            _wait_for_running(
                job_id,
                eai_path=self.eai_path,
                profile=profile,
                account=self.account,
                timeout=self.launch_timeout_seconds,
            )

            container = ToolkitContainer(
                job_id,
                profile=profile,
                account=self.account,
                eai_path=self.eai_path,
                relay_prestarted_token=relay_token,
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
            container.expires_at = container.created_at + timedelta(seconds=effective_ttl) if effective_ttl else None
            return container
        except Exception:
            # _wait_for_running already kills on its own failure paths, but the
            # construction + bookkeeping below it do not. Be defensive.
            logger.exception("Post-submission setup failed for EAI job %s; killing to prevent leak", job_id)
            try:
                _run_eai(
                    ["job", "kill", job_id],
                    eai_path=self.eai_path,
                    profile=profile,
                    account=self.account,
                    timeout=30,
                )
            except Exception as kill_exc:
                logger.warning("Failed to kill EAI job %s during cleanup: %s", job_id, kill_exc)
            raise

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
    eai_path: str = "eai",
    profile: str | None,
    account: str | None,
    timeout: int,
) -> None:
    deadline = time.monotonic() + timeout
    last_state = ""
    while time.monotonic() < deadline:
        result = _run_eai(
            ["job", "get", job_id, "--format", "json", "--no-header"],
            eai_path=eai_path,
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
            _run_eai(["job", "kill", job_id], eai_path=eai_path, profile=profile, account=account, timeout=30)
            raise ContainerLaunchError(f"EAI job {job_id} entered terminal state: {state}")

        last_state = state
        logger.info("EAI job %s state=%s, waiting…", job_id[:8], state)
        time.sleep(5)

    _run_eai(["job", "kill", job_id], eai_path=eai_path, profile=profile, account=account, timeout=30)
    raise ContainerLaunchError(f"EAI job {job_id} did not reach RUNNING within {timeout}s (last state: {last_state!r})")
