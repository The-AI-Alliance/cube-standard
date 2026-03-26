"""
VM abstraction for desktop-automation benchmarks.

Mirrors the container.py pattern — separates WHAT the task needs (VMConfig)
from HOW to provision it (VMBackend) and the live runtime handle (VM).

The core resource is an HTTP endpoint, not a specific hypervisor.
Every VMBackend implementation converges to the same HTTP interface that
ComputerBase (in cube-computer-tool) talks to.

Design reference: cube-standard/design/vm_backend.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from cube.core import TypedBaseModel


class VMConfig(TypedBaseModel):
    """Declarative description of *what* the task needs — owned by the benchmark.

    These are task requirements, not infrastructure details. The VMBackend
    is responsible for mapping these onto its specific provisioning mechanism
    (QEMU flags, EC2 instance type, Azure image, etc.).

    Attributes:
        snapshot_name: Named snapshot to restore between tasks.
        cpu_cores: Number of vCPUs required.
        ram_gb: RAM in gigabytes.
        screen_size: (width, height) resolution in pixels.
        os_type: Guest operating system type.
    """

    snapshot_name: str = "init_state"
    cpu_cores: int = 4
    ram_gb: int = 4
    screen_size: tuple[int, int] = (1920, 1080)
    os_type: Literal["Ubuntu", "Windows"] = "Ubuntu"


class VM(ABC):
    """Runtime handle to a running desktop VM. Not serializable.

    The VM exposes a single HTTP endpoint that accepts screenshot/execute
    requests. ComputerBase (cube-computer-tool) uses this endpoint to drive
    the desktop without knowing which hypervisor is underneath.
    """

    def __repr__(self) -> str:
        return f"<{type(self).__name__} endpoint={self.endpoint}>"

    @property
    @abstractmethod
    def endpoint(self) -> str:
        """Base URL of the in-VM HTTP agent: ``http://host:port``."""

    @abstractmethod
    def restore_snapshot(self, name: str) -> None:
        """Revert the VM to a named snapshot. Called between tasks.

        Implementations vary by backend:
        - LocalQEMU: delete overlay + reboot (~30s, RESTART isolation)
        - Cloud prebuit AMI: stop + re-launch (~2-4 min, NEW_INSTANCE isolation)
        - VM savestate: qemu savevm/loadvm (~5s, SNAPSHOT isolation)
        """

    @abstractmethod
    def stop(self) -> None:
        """Shut down and release this VM's resources. Idempotent."""


class VMBackend(TypedBaseModel, ABC):
    """Serializable description of *how* to provision VMs — owned by the harness user.

    Defined once per experiment run, shared across all tasks.
    Image references, credentials, and cloud-specific parameters live here,
    not in VMConfig.

    Subclasses are responsible for:
    - ensure_resource(): idempotent one-time setup (image download/import)
    - launch(): blocking call that returns a ready VM handle
    - close(): release all VMs managed by this backend
    """

    @abstractmethod
    def ensure_resource(self, config: VMConfig) -> None:
        """Idempotent: ensure the backend-specific image exists for *config*.

        First call: download qcow2 → convert/upload → import as snapshot/AMI.
        Subsequent calls: no-op (cached).

        Called automatically by launch(), but can be called explicitly during
        benchmark.setup() to front-load the one-time setup cost.
        """

    @abstractmethod
    def launch(self, config: VMConfig) -> VM:
        """Calls ensure_resource(), then blocks until the HTTP endpoint is ready.

        Returns a VM handle. The caller owns the lifecycle (call vm.stop() when done).
        """

    @abstractmethod
    def close(self) -> None:
        """Release all resources managed by this backend (VMs, pools, etc.)."""
