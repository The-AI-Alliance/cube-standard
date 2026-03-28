"""
Resource lifecycle abstractions for CUBE.

Three-level model:
    L1 — Install-time:    one-time image prep per infra/region    (slow, idempotent)
    L2 — Benchmark-wide:  shared server per benchmark run         (e.g. WebArena)
    L3 — Task-level:      per-task ephemeral resources            (e.g. individual VMs)

Core abstractions:
    ResourceConfig  — WHAT the benchmark needs (benchmark-owned, serializable)
    InfraConfig     — HOW to provision it (harness-owned, serializable + executable)
    ResourceHandle  — Live runtime object (not serializable, returned by launch())
    ProvisionStore  — Maps (resource, infra) → resource_info (~/.cube/provisions.json)

Design reference: cube-standard/design/resource_lifecycle.md
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from cube.core import TypedBaseModel

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
        scope:                   "task" = per-task ephemeral (L3); "benchmark" = shared
                                 server for the whole run (L2, e.g. WebArena).
        max_concurrent_agents:   Capacity hint for L2 resources; None = no limit.
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

    def requirements(self) -> set[str]:
        return {"kvm"} if self.requires_kvm else set()


class DockerServiceConfig(ResourceConfig):
    """Multi-container Docker Compose stack (WebArena, WorkArena, TheAgentCompany...)."""

    compose_url: str

    def requirements(self) -> set[str]:
        return {"docker"}


class DockerImageConfig(ResourceConfig):
    """Single Docker image per task (SWE-bench, MLE-bench, CTF...)."""

    image: str

    def requirements(self) -> set[str]:
        return {"docker"}


# ── ResourceHandle ────────────────────────────────────────────────────────────


@dataclass
class ResourceHandle(ABC):
    """Live handle to a running cloud/local resource. Not serializable.

    Returned by InfraConfig.launch(). Holds live state (subprocess, cloud client, etc.).

    close() is the primary API. The context manager is a convenience wrapper.
    For multi-process use cases, pass run_id (a plain string) across process
    boundaries and call infra.cleanup(run_id) from any process.
    """

    run_id: str
    resource: ResourceConfig
    infra: InfraConfig
    endpoint: str | None
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime | None = None

    @abstractmethod
    def close(self) -> None:
        """Tear down this resource (delete VM, stop container, etc.)."""
        ...

    def __enter__(self) -> "ResourceHandle":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ── InfraConfig ───────────────────────────────────────────────────────────────


class InfraConfig(TypedBaseModel, ABC):
    """Harness-owned config + executor for resource provisioning and lifecycle.

    Extends TypedBaseModel for serializability (polymorphic via _type field —
    subclasses declare no _type field, it is injected automatically).
    Also carries launch/cleanup methods — instantiating the config IS the backend,
    following the existing VMBackend pattern in vm.py.

    Credentials are never stored in fields; resolved from env vars at runtime.

    Concrete subclasses must implement:
        fingerprint()   — stable key encoding provider + region/location only
        capabilities()  — set of supported capability tokens
        provision()     — L1 automated image prep (download → convert → upload → import)
        launch()        — L2/L3 resource instantiation
        list_active()   — enumerate live resources
        cleanup()       — delete all resources for a run_id
        cleanup_stale() — delete resources past their expires_at
    """

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
        """L1: automated image prep (download → convert → upload → import → register).

        Idempotent — safe to call multiple times. Calls register() on completion.
        Raises UnsupportedResourceType if this infra cannot provision the resource type.
        """
        ...

    @abstractmethod
    def launch(
        self,
        resource: ResourceConfig,
        run_id: str,
        ttl_seconds: int | None = None,
    ) -> ResourceHandle:
        """L2/L3: instantiate a resource and return a live handle.

        Reads resource_info from the ProvisionStore. Raises ResourceNotReadyError
        if no entry is found (i.e. register() or provision() was never called).

        Tags the resource with run_id and expires_at for cleanup.
        ttl_seconds overrides resource.default_ttl_seconds when provided.
        """
        ...

    @abstractmethod
    def list_active(self, run_id: str | None = None) -> list[ResourceHandle]:
        """List live L2/L3 resources, optionally filtered by run_id."""
        ...

    @abstractmethod
    def cleanup(self, run_id: str) -> None:
        """Delete all resources associated with run_id."""
        ...

    @abstractmethod
    def cleanup_stale(self, max_age_seconds: int | None = None) -> list[str]:
        """Delete resources past their expires_at.

        If max_age_seconds is set, also deletes resources older than that
        even if they have no expires_at tag.
        Returns list of deleted resource identifiers.
        """
        ...

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

    def provision_status(
        self, resource: ResourceConfig
    ) -> Literal["ready", "needs_provisioning"]:
        """Query the ProvisionStore for this (resource, infra) pair.

        Returns "ready" if register() or provision() has been called,
        "needs_provisioning" otherwise.
        """
        from cube.provision_store import ProvisionStore

        store = ProvisionStore()
        return "ready" if store.get(resource, self) is not None else "needs_provisioning"

    def can_serve(self, resource: ResourceConfig) -> bool:
        """Return True if this infra's capabilities satisfy the resource's requirements."""
        return resource.requirements().issubset(self.capabilities())
