"""VM abstraction for desktop/OS benchmark tasks.

Separates *what* a task needs from a VM (VMConfig) from *how* to provision
and run that VM (VMBackend / VM).

    VMConfig  — declarative description of VM requirements owned by the benchmark.
    VMBackend — serializable config describing how to provision a VM (e.g. local
                Docker+QEMU, cloud EC2, etc.). Concrete implementations live in
                cube-tools/cube-computer-tool/.
    VM        — runtime handle to a running VM instance. Not serializable.

Following the same what/how pattern as container.py (ContainerConfig /
ContainerBackend / Container).

Usage example:

    from cube.vm import VMConfig, VMBackend, VM

    class OSWorldBenchmark(Benchmark):
        vm_config: VMConfig = VMConfig()
        # vm_backend is provided by the harness user, not the benchmark

    backend = LocalQEMUVMBackend(vm_image_path="/path/to/ubuntu.qcow2", docker_image="...")
    vm: VM = backend.launch(benchmark.vm_config)
    try:
        vm.restore_snapshot("init_state")
        ...
    finally:
        vm.stop()
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from cube.core import TypedBaseModel


class VMConfig(TypedBaseModel):
    """Declarative description of *what* a task needs from a VM.

    Owned by the benchmark, not the harness. Only includes fields that
    benchmarks genuinely care about — backend-specific settings (CPU cores,
    RAM, disk size, Docker image) belong on VMBackend subclasses.
    """

    snapshot_name: str = "init_state"
    """Name of the snapshot to restore at the start of each task."""

    screen_size: tuple[int, int] = (1920, 1080)
    """Display resolution (width, height) in pixels."""


class VMBackend(TypedBaseModel, ABC):
    """Serializable config describing *how* to provision and launch a VM.

    Subclass to add backend-specific fields (e.g. docker_image, vm_image_path,
    headless, AWS region, etc.). Concrete implementations live in
    cube-tools/cube-computer-tool/.

    Pattern mirrors ContainerBackend in container.py.
    """

    @abstractmethod
    def launch(self, config: VMConfig) -> VM:
        """Launch a VM described by *config*. Blocks until the Flask server is ready.

        Args:
            config: Declarative VM requirements from the benchmark.

        Returns:
            A VM runtime handle ready to accept requests.
        """

    def close(self) -> None:
        """Optional: clean up any backend-level resources (e.g. Docker client)."""


class VM(ABC):
    """Runtime handle to a running VM instance.

    Not serializable — created by VMBackend.launch() and used for the duration
    of one benchmark episode. Provides the HTTP endpoint for tool communication
    and snapshot management.
    """

    @property
    @abstractmethod
    def endpoint(self) -> str:
        """Base URL of the Flask server running inside the VM container.

        Example: ``"http://localhost:54321"``
        """

    @abstractmethod
    def restore_snapshot(self, name: str) -> None:
        """Restore the VM to a named snapshot.

        For Docker+QEMU backends this typically means stopping and restarting
        the container (the qcow2 image is mounted read-only, so a fresh
        container always starts from the base state).

        Args:
            name: Snapshot name (e.g. ``"init_state"``). Interpretation is
                  backend-specific.
        """

    @abstractmethod
    def stop(self) -> None:
        """Stop and clean up the VM. Idempotent."""
