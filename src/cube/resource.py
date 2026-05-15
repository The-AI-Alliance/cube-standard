"""
Resource lifecycle abstractions for CUBE.

Resource Lifetime Levels
------------------------
Resources in CUBE have three distinct lifetime levels. Harness developers must
understand which level each cleanup method targets:

    Level 1 — Provisioned images (long-lived, manual teardown)
        Created once per (resource, infra) pair by provision() or register().
        Examples: AWS AMI, Azure Compute Gallery image version, local qcow2 file.
        Persist indefinitely — shared across all runs on the same infra.
        Tracked in ProvisionStore (~/.cube/provisions/).
        Teardown: unprovision() — explicit and intentional only.

    Level 2 — Benchmark-scoped resources (mid-lived, automatic teardown)
        Shared server launched once for an entire benchmark run.
        Examples: WebArena server, WorkArena ServiceNow instance.
        Declared with scope="benchmark" on ResourceConfig.
        Created at benchmark.setup(), torn down at benchmark.close().
        Teardown: handle.close() at run end, or infra.cleanup(run_id) as catch-all.

    Level 3 — Task-scoped resources (short-lived, automatic teardown)
        Ephemeral resource created fresh for each individual task.
        Examples: individual OSWorld VMs, per-task Docker containers.
        Declared with scope="task" on ResourceConfig.
        Created by launch() at task start, deleted at task end.
        Teardown: handle.close() after each task, swept by cleanup_stale() on crash.

Cleanup Method Summary (for harness developers)
------------------------------------------------
    handle.close()              — L2/L3. Tear down one specific live resource.
                                  Call after each task (L3) or at run end (L2).

    infra.cleanup(run_id)       — L2/L3. Delete all resources for a run_id.
                                  Call at harness shutdown as a catch-all.

    infra.cleanup_stale()       — L2/L3. Delete expired resources across all runs
                                  by reading cloud tags. No local state needed.
                                  Call at harness startup to GC crashed runs.

    infra.unprovision(resource) — L1 only. Tear down the provisioned image and
                                  its ProvisionStore entry. Manual, intentional.
                                  Use when retiring a benchmark or re-provisioning.

Recommended harness lifecycle
------------------------------
    infra.cleanup_stale()                    # startup: GC orphans from prior crashes
    benchmark.setup()                        # creates L2 resource if needed
    for task in tasks:
        handle = infra.launch(resource)      # creates L3 resource
        try:
            run_episode(task, handle)
        finally:
            handle.close()                   # tears down L3 resource immediately
    infra.cleanup(run_id=run_id)             # shutdown: catch-all for the run
    benchmark.close()                        # tears down L2 resource

Core abstractions
-----------------
    ResourceConfig  — WHAT the benchmark needs (benchmark-owned, serializable)
    InfraConfig     — HOW to provision it (harness-owned, serializable + executable)
    ResourceHandle  — Live runtime object (not serializable, returned by launch())
    ProvisionStore  — Maps (resource, infra) → resource_info (~/.cube/provisions/)

Design reference: cube-standard/design/resource_lifecycle.md
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from pydantic import Field

from cube.core import TypedBaseModel, ValidatedConfig

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ── Exceptions ────────────────────────────────────────────────────────────────


class ResourceNotReadyError(RuntimeError):
    """Raised by launch() when no resource_info is registered for (resource, infra).

    The error message includes actionable instructions to resolve the issue.
    """

    def __init__(self, resource: "ResourceConfig", infra: "InfraConfig") -> None:
        super().__init__(
            f"{resource.name!r} is not registered for {infra.fingerprint()!r}.\n"
            f"  Run: infra.register(resource, {{...}})        # manual\n"
            f"  Or:  infra.provision(resource)               # automated, if supported"
        )


class UnsupportedResourceType(NotImplementedError):
    """Raised when an InfraConfig does not support a given ResourceConfig subtype."""

    def __init__(self, resource: "ResourceConfig", infra: "InfraConfig") -> None:
        super().__init__(
            f"{type(infra).__name__} does not support {type(resource).__name__} "
            f"({resource.name!r}). Check infra.capabilities() vs resource.requirements()."
        )


# ── ResourceConfig ────────────────────────────────────────────────────────────


class ResourceConfig(TypedBaseModel):
    """Declarative description of a single resource dependency — owned by the benchmark author.

    Describes WHAT is needed; the InfraConfig decides HOW to provision it.
    Stays a pure data object — no store or infra-aware methods.

    Attributes:
        name:                    Stable identifier, e.g. "osworld-ubuntu-vm".
        scope:                   "task" = per-task ephemeral; "benchmark" = shared
                                 server for the whole run (e.g. WebArena).
        max_concurrent_agents:   Capacity hint for benchmark-scoped resources; None = no limit.
        source_url:              Canonical image source (HuggingFace URL, Docker Hub, etc.).
        source_hash:             Content hash for informational purposes; not used for
                                 deduplication in v1.
        default_ttl_seconds:     Max lifetime before auto-cleanup. None = no expiry.
        bootstrap_script_extra:  Optional bash fragment injected into the infra's bootstrap
                                 script. Escape hatch for benchmark-specific VM setup.
                                 Must be declared in source — never fetched at runtime.
    """

    name: str
    scope: Literal["task", "benchmark"] = "task"
    max_concurrent_agents: int | None = None
    source_url: str | None = None
    source_hash: str | None = None
    default_ttl_seconds: int | None = 3600
    bootstrap_script_extra: str | None = None

    def requirements(self) -> set[str]:
        """Declare what the infra must support to run this resource.

        Checked against InfraConfig.capabilities() before provisioning or launch.
        Standard tokens: "kvm", "docker", "gpu:nvidia", "network:egress".
        """
        return set()


class VMResourceConfig(ResourceConfig):
    """VM-based resource (OSWorld, WindowsAgentArena, macOSWorld, AndroidWorld...)."""

    requires_kvm: bool = True
    os_disk_gb: int | None = None
    """Minimum OS disk size (GB) for task VMs. None means use the image's native size."""
    min_cpu_cores: int | None = None
    """Minimum vCPU count required. None defers to the infra's default instance size."""
    min_ram_gb: int | None = None
    """Minimum RAM (GB) required. None defers to the infra's default instance size."""
    uefi: bool = False
    """Boot with UEFI firmware. Required for Windows 11 / Trusted Launch images."""
    tpm: bool = False
    """Attach emulated TPM 2.0. Required for Windows 11."""
    os_type: Literal["linux", "windows"] = "linux"
    """Guest operating system. Infras branch on this to pick the correct os_profile
    (linux_configuration with SSH key vs windows_configuration with admin password),
    image os_type, and platform-specific bootstrap steps."""
    specialized: bool = False
    """If True, the image is a byte-for-byte clone of a specific VM (hostname, SID,
    user accounts, credentials all preserved). Azure deploys it without applying
    os_profile at launch — admin credentials and SSH keys must already be baked into
    the image. If False, the image was sysprep /generalize'd and Azure applies fresh
    os_profile at first boot."""
    forwarded_ports: list[int] = Field(default_factory=list)
    """Additional VM ports to expose on the host beyond the guest-agent port.
    Each port gets its own SSH tunnel from a free host port to the named VM port.
    The host-side URL appears in ResourceHandle.endpoints under the key
    ``"vm_port_{port}"`` (e.g. ``"vm_port_9222"`` → ``"http://localhost:54321"``).
    Use a freeport indirection — never assume host port == VM port — so multiple
    parallel workers can share a host without colliding on a fixed local port."""

    def requirements(self) -> set[str]:
        return {"kvm"} if self.requires_kvm else set()


class VolumeSpec(TypedBaseModel):
    """Named Docker volume, optionally pre-populated from a remote archive.

    Used during provision() to download data and extract it into Docker volumes
    that are baked into the VM snapshot.  At launch(), volumes are mounted into
    the container via their ``name`` and ``mount_path``.

    Volumes without a ``source_url`` are created empty (populated at container runtime).

    Example — pre-populated from a tarball::

        VolumeSpec(
            name="webarena_map_tile_db",
            mount_path="/data/database",
            source_url="https://example.com/osm_tile_server.tar",
            tar_subpath="projects/ogma3/docker/volumes/osm-data/_data",
            strip_components=6,
        )

    Example — empty volume::

        VolumeSpec(name="webarena_map_tiles", mount_path="/data/tiles")
    """

    name: str
    mount_path: str
    source_url: str | None = None
    tar_subpath: str | None = None
    strip_components: int = 0


class DockerServiceConfig(ResourceConfig):
    """Multi-container Docker service stack (WebArena, WorkArena, TheAgentCompany...).

    Attributes:
        docker_images:    Docker Hub images to pre-pull during provision (determines the
            provisioned snapshot content). Required — provision() will fail without them.
        services:         Maps service names to ports. Keys become the keys in
            ResourceHandle.endpoints. On cloud infra (AWS/Azure) launch() SSH-tunnels
            each guest port to a free local port. On LocalInfraConfig launch() uses
            these as direct host ports (the launch_script binds them on the host).
            Example: ``{"shopping_admin": 7780, "shopping_admin_ctrl": 7781}``
        launch_script:    Bash snippet run inside the VM at launch time to start services
            (e.g. ``docker run -d -p 7780:80 am1n3e/webarena-verified-shopping_admin``).
            Images must already be present from provision(); no docker pull at launch time.
        endpoint_to_site: Maps service names (keys in ``services``) to benchmark-specific
            site identifiers. The benchmark interprets this mapping;
            ``DockerServiceConfig`` itself does not. Only web-UI endpoints should appear
            here — control/API endpoints are omitted.
            Example: ``{"shopping_admin": "shopping_admin"}``
        volumes:          Data volumes to download and extract during provision().  Each
            ``VolumeSpec`` with a ``source_url`` is fetched and extracted into a Docker
            volume that is baked into the VM snapshot.  Empty volumes (no ``source_url``)
            are created for runtime use.  Archives referenced by multiple specs are
            downloaded once.
    """

    docker_images: list[str] = []
    services: dict[str, int] = {}
    launch_script: str = ""
    endpoint_to_site: dict[str, str] = {}
    volumes: list[VolumeSpec] = []

    def requirements(self) -> set[str]:
        return {"docker"}


class DockerImageConfig(ResourceConfig):
    """Single Docker image per task (SWE-bench, MLE-bench, CTF...).

    Resource requirements (ram_gb, cpu_cores, disk_gb, ports) are read by
    DockerInfraConfig.launch() to configure the container. gpu=True maps to
    the "gpu:nvidia" capability requirement token via requirements().
    """

    image: str
    ram_gb: float = 4.0
    cpu_cores: float = 2.0
    disk_gb: float = 10.0
    gpu: bool = False
    ports: list[int] | None = None

    def requirements(self) -> set[str]:
        reqs = {"docker"}
        if self.gpu:
            reqs.add("gpu:nvidia")
        return reqs


# ── ResourceHandle ────────────────────────────────────────────────────────────


@dataclass
class ResourceHandle(ABC):
    """Live handle to a running cloud/local resource. Not serializable.

    Returned by InfraConfig.launch(). Holds live state (subprocess, cloud client, etc.).

    Resource lifetime hierarchy
    ---------------------------
    Level 1 — Long-lived (L1):
        Provisioned images (AMI, Gallery image, local qcow2). Created once per
        (resource, infra) pair by provision() or register(). Persist indefinitely
        until explicitly removed by unprovision(). Shared across all runs.
        Tracked in ProvisionStore (~/.cube/provisions/).

    Level 2 — Benchmark-scoped (L2):
        Shared server for an entire benchmark run (e.g. WebArena, WorkArena).
        Created once at benchmark.setup(), shared across all tasks in the run.
        scope="benchmark" on ResourceConfig.
        Teardown: handle.close() or infra.cleanup(run_id) at run end.

    Level 3 — Task-scoped (L3):
        Per-task ephemeral resource (e.g. individual OSWorld VMs).
        Created by launch() at task start, deleted at task end.
        scope="task" on ResourceConfig.
        Teardown: handle.close() after each task.

    Cleanup method guide for harness developers
    -------------------------------------------
    handle.close()
        WHAT:    Tears down this specific live resource (VM + tunnel + NIC/IP).
        WHEN:    Call immediately after each task completes (L3), or once at
                 run end (L2). Use as a context manager for single-process flows.
        RECOVERS: If skipped, the orphaned resource is caught by cleanup_stale()
                 at the next harness startup via the cube:expires_at cloud tag.

    infra.cleanup(run_id)
        WHAT:    Deletes all live cloud resources tagged with run_id, regardless
                 of whether handle.close() was called.
        WHEN:    Call at harness shutdown (normal exit and signal handlers).
                 Safe to call even if all handles were already closed — no-ops
                 on already-deleted resources.
        RECOVERS: If skipped, resources linger until cleanup_stale() expires them
                 by TTL. Use this as the catch-all at run end.

    infra.cleanup_stale(max_age_seconds)
        WHAT:    Reads cube:expires_at tags directly from the cloud API and deletes
                 any resource past its TTL. No dependency on local state — works
                 after a full process crash or across machines.
        WHEN:    Call at harness STARTUP, before launching any new work, to GC
                 orphans left by previous crashed runs.
        RECOVERS: If never called, orphaned resources accumulate indefinitely,
                 causing cost leaks and quota exhaustion.

    infra.unprovision(resource)
        WHAT:    Tears down Level 1 long-lived artifacts: provisioned image
                 (AMI/Gallery image), VHD blobs, sentinels, and the ProvisionStore
                 entry. Does NOT affect any running VMs.
        WHEN:    Manual only. Use when retiring a benchmark from an infra, switching
                 regions, or forcing a full re-provision (e.g. new base image).
        RECOVERS: If skipped after retiring a benchmark, L1 artifacts (images,
                 blobs) remain in cloud storage and incur ongoing storage costs.

    Recommended harness lifecycle
    ------------------------------
        infra.cleanup_stale()                    # startup: GC from previous crashes
        benchmark.setup()                        # launches L2 resource if needed
        for task in tasks:
            handle = infra.launch(resource)      # L3: per-task VM
            try:
                run_episode(task, handle)
            finally:
                handle.close()                   # L3: delete VM immediately
        infra.cleanup(run_id=run_id)             # shutdown: catch-all for the run
        benchmark.close()                        # teardown L2 resource

    For multi-process use (e.g. Ray workers): pass run_id (a plain string) across
    process boundaries and call infra.cleanup(run_id) from any process — the handle
    itself is not serializable and must not be passed to workers.
    """

    # All fields default so ``cube.container.Container`` subclasses (e.g.
    # ``LocalContainer``) can construct without bookkeeping — the
    # ``InfraConfig.launch()`` path still populates them.
    run_id: str = ""
    resource: ResourceConfig | None = None
    infra: InfraConfig | None = None
    endpoint: str | None = None
    endpoints: dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime | None = None

    @abstractmethod
    def close(self) -> None:
        """Tear down this specific live resource (delete VM, stop container, etc.).

        Deletes all cloud sub-resources associated with this handle (VM instance,
        SSH tunnel, NIC, public IP, etc.). Idempotent — safe to call more than once.

        For single-process flows, prefer the context manager form:
            with infra.launch(resource) as handle:
                run_episode(task, handle.endpoint)

        If close() is not called (e.g. process crash), the orphaned resource will
        be swept by infra.cleanup_stale() at the next harness startup.
        """
        ...

    def __enter__(self) -> "ResourceHandle":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ── InfraConfig ───────────────────────────────────────────────────────────────


class InfraConfig(ValidatedConfig, ABC):
    """Harness-owned config + executor for resource provisioning and lifecycle.

    Extends TypedBaseModel for serializability (polymorphic via _type field —
    subclasses declare no _type field, it is injected automatically).
    Also carries launch/cleanup methods — instantiating the config IS the backend,
    following the existing VMBackend pattern in vm.py.

    Credentials are never stored in fields; resolved from env vars at runtime.

    Fields:
        default_ttl_seconds: Maximum resource lifetime before auto-cleanup.
                             Overrides resource.default_ttl_seconds when set.
                             Default is 1 day (86400s) — long enough to survive any
                             realistic evaluation run, short enough to avoid weeks-long
                             orphan accumulation. Set None to disable auto-expiry.

    Concrete subclasses must implement:
        fingerprint()   — stable key encoding provider + region/location only
        capabilities()  — set of supported capability tokens
        provision()     — L1: automated image prep (download → convert → upload → import)
        launch()        — L2/L3: resource instantiation from a provisioned image
        list_active()   — L2/L3: enumerate live resources
        cleanup()       — L2/L3: delete all resources for a run_id (call at shutdown)
        cleanup_stale() — L2/L3: delete expired resources across all runs (call at startup)

    Subclasses may optionally override:
        unprovision()   — L1: tear down provisioned image and ProvisionStore entry
                          Defaults to no-op. Override for infras that support it.

    See module docstring for the full level hierarchy and recommended harness lifecycle.
    """

    default_ttl_seconds: int | None = 86400
    """Max lifetime for launched resources. Overrides resource.default_ttl_seconds.
    1 day default prevents orphan accumulation without killing long-running jobs.
    Set None to disable auto-expiry (use cleanup() or handle.close() explicitly)."""

    image_name_suffix: str = ""
    """Suffix appended to image/AMI names and ProvisionStore keys.
    Use "-test" to isolate CI/test runs from production images without changing
    the resource name.  E.g. image_name_suffix="-test" and resource.name="foo"
    → image named "foo-test", store key "foo-test@<fingerprint>"."""

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def fingerprint(self) -> str:
        """Stable ProvisionStore key encoding provider + region/location.

        Must NOT encode performance knobs (instance size, CPU count) — those
        do not affect which image is needed. Two configs with the same fingerprint
        share the same provisioned image.

        Examples: "local", "aws:us-east-2", "azure:westus2", "docker:docker.io"
        """
        ...

    @abstractmethod
    def capabilities(self) -> set[str]:
        """Declare what this infra supports.

        Checked against resource.requirements() before provisioning or launch.
        Standard tokens: "kvm", "docker", "gpu:nvidia", "network:egress".
        """
        ...

    @abstractmethod
    def provision(self, resource: ResourceConfig) -> None:
        """Automated image prep (download → convert → upload → import → register).

        Idempotent — safe to call multiple times. Calls register() on completion.
        Raises UnsupportedResourceType if this infra cannot provision the resource type.
        """
        ...

    @abstractmethod
    def launch(self, resource: ResourceConfig) -> ResourceHandle:
        """Instantiate a resource and return a live handle.

        Reads resource_info from the ProvisionStore. Raises ResourceNotReadyError
        if no entry is found (i.e. register() or provision() was never called).

        run_id is generated internally (UUID4) and stored on the returned handle.
        TTL is resolved as: self.default_ttl_seconds ?? resource.default_ttl_seconds.
        Both are embedded as cube:expires_at tags on the cloud resource for recovery
        even if local state is lost.
        """
        ...

    @abstractmethod
    def list_active(self, run_id: str | None = None) -> list[ResourceHandle]:
        """List live resources, optionally filtered by run_id."""
        ...

    @abstractmethod
    def cleanup(self, run_id: str) -> None:
        """L2/L3: Delete all live resources associated with run_id.

        Targets all cloud resources tagged with cube:run_id=run_id, regardless
        of whether handle.close() was already called. Safe to call on already-deleted
        resources — implementations must no-op gracefully.

        When to call: at harness shutdown (normal exit and signal handlers) as a
        catch-all for any resources not explicitly closed via handle.close().
        Does NOT affect L1 provisioned images.
        """
        ...

    @abstractmethod
    def cleanup_stale(self, max_age_seconds: int | None = None) -> list[str]:
        """L2/L3: Delete expired resources across all runs by reading cloud tags.

        Reads cube:expires_at tags directly from the cloud provider API — no
        dependency on local state. Works after a full process crash or across
        machines. If max_age_seconds is set, also deletes resources older than
        that even if they have no cube:expires_at tag.

        When to call: at harness STARTUP, before launching any new work, to GC
        orphans left by previous crashed or abandoned runs.
        Does NOT affect L1 provisioned images.

        Returns: list of deleted resource identifiers (e.g. VM names or instance IDs).
        """
        ...

    def unprovision(self, resource: ResourceConfig) -> None:  # noqa: ARG002
        """L1: Tear down the provisioned image and its ProvisionStore entry.

        Deletes all Level 1 artifacts created by provision() for this
        (resource, infra) pair: the cloud image (AMI, Gallery image version, etc.),
        any intermediate blobs or snapshots, and the ProvisionStore entry.

        This is a manual, intentional operation — never called automatically by
        the harness. Use when:
          - Retiring a benchmark from this infra (free up storage costs)
          - Switching regions (unprovision here, provision in the new region)
          - Forcing a full re-provision of a new base image

        Does NOT affect any running L2/L3 resources (VMs, containers). To stop
        live resources use handle.close(), cleanup(run_id), or cleanup_stale().

        Default implementation is a no-op. Override in infras that manage L1 images
        (e.g. AWSInfraConfig, AzureInfraConfig). Infras where images are externally
        managed (e.g. public Docker Hub images) should leave this as a no-op.

        Raises:
            UnsupportedResourceType: if the resource type is not supported by this infra.
        """

    # ── Concrete helpers ──────────────────────────────────────────────────────

    def _image_name(self, resource: "ResourceConfig") -> str:
        """Effective image/AMI name: resource.name + image_name_suffix."""
        return resource.name + self.image_name_suffix

    def _resource_shim(self, resource: "ResourceConfig") -> "ResourceConfig":
        """Minimal shim with .name = _image_name(resource) for ProvisionStore keys.

        Allows image_name_suffix to redirect provision() and launch() to a
        suffixed store entry without changing the resource's own name.
        """
        import types

        return types.SimpleNamespace(name=self._image_name(resource))  # type: ignore[return-value]

    # ── Concrete store-backed methods ─────────────────────────────────────────

    def register(self, resource: ResourceConfig, resource_info: dict) -> None:
        """Record that a resource is available for this (resource, infra) pair.

        resource_info is an opaque dict interpreted by launch() — e.g.
        {"ami_id": "ami-..."} for AWS, {"image_path": "/..."} for local.

        Calling register() with new info overrides the existing entry.
        This is the only thing launch() depends on — provenance does not matter.
        """
        from cube.provision_store import ProvisionStore

        store = ProvisionStore()
        if store.get(resource, self) is not None:
            logger.warning(
                "Overriding existing registration for %r @ %r",
                resource.name,
                self.fingerprint(),
            )
        store.put(resource, self, resource_info)
        logger.info("Registered %r @ %r", resource.name, self.fingerprint())

    def provision_status(self, resource: ResourceConfig) -> Literal["ready", "needs_provisioning"]:
        """Query the ProvisionStore for this (resource, infra) pair.

        Returns "ready" if register() or provision() has been called,
        "needs_provisioning" otherwise.
        Respects image_name_suffix via _resource_shim().
        """
        from cube.provision_store import ProvisionStore

        store = ProvisionStore()
        return "ready" if store.get(self._resource_shim(resource), self) is not None else "needs_provisioning"

    def can_serve(self, resource: ResourceConfig) -> bool:
        """Return True if this infra's capabilities satisfy the resource's requirements."""
        return resource.requirements().issubset(self.capabilities())
