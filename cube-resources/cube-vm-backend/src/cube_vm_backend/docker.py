"""LocalDockerVMBackend — VMBackend that runs QEMU inside a Docker container.

Uses ``happysixd/osworld-docker``, which wraps QEMU in a container so the host
only needs Docker Desktop (no QEMU installation, works on macOS).

The qcow2 base image is still required — it is mounted read-only at
``/System.qcow2`` inside the container. Benchmarks that need auto-download
(e.g. OSWorld) subclass this backend and override ensure_resource().

Reset strategy: stop + remove container, start fresh container with the same
read-only qcow2 mount (NEW_INSTANCE isolation, ~30-60s). Ports are stable
across resets (reserved before first launch, held until stop()).
"""

from __future__ import annotations

import logging
from pathlib import Path

from cube.vm import VM, VMBackend, VMConfig

from cube_vm_backend.docker_manager import OSWORLD_DOCKER_IMAGE, DockerConfig, DockerManager

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path.home() / ".cube" / "vm_data"


class LocalDockerVM(VM):
    """Runtime handle to a QEMU-in-Docker VM managed by LocalDockerVMBackend.

    Not serializable. The caller owns the lifecycle via stop().
    """

    def __init__(self, manager: DockerManager) -> None:
        self._manager = manager

    @property
    def endpoint(self) -> str:
        """Base URL of the in-VM HTTP agent: ``http://localhost:<port>``."""
        return f"http://localhost:{self._manager.server_port}"

    @property
    def chromium_port(self) -> int:
        """Host port forwarded to guest Chromium DevTools (9222)."""
        return self._manager.chromium_port

    @property
    def vlc_port(self) -> int:
        """Host port forwarded to guest VLC HTTP (8080)."""
        return self._manager.vlc_port

    @property
    def server_port(self) -> int:
        """Host port forwarded to guest Flask agent (5000)."""
        return self._manager.server_port

    def restore_snapshot(self, name: str) -> None:
        """Restore the VM to its initial state.

        Implementation: stop + remove container, start fresh container with the
        same read-only qcow2 mount (NEW_INSTANCE isolation, ~30-60s). The ``name``
        argument is accepted for API compatibility but ignored — only the base image
        state is available.
        """
        logger.info("Restoring Docker VM snapshot '%s' (NEW_INSTANCE reset)", name)
        self._manager.reset()

    def stop(self) -> None:
        """Stop the container and release port reservations."""
        self._manager.stop()


class LocalDockerVMBackend(VMBackend):
    """VMBackend that runs QEMU inside Docker on the local host.

    Works on macOS via Docker Desktop — no QEMU installation required.
    The qcow2 base image is still needed and mounted into the container.

    Attributes
    ----------
    cache_dir : str
        Directory for the base qcow2 image (overlays are inside the container).
    path_to_vm : str | None
        Explicit path to a pre-existing qcow2 base image.
        If None, subclasses should override ensure_resource() to acquire it.
    memory : str
        RAM allocation (e.g. ``"4G"``).
    cpus : int
        Number of vCPUs.
    disk_size : str
        Disk size passed to the container (e.g. ``"32G"``).
    pull_policy : str
        Docker image pull policy: ``"missing"`` (default), ``"always"``, ``"never"``.
    """

    cache_dir: str = str(_DEFAULT_CACHE_DIR)
    path_to_vm: str | None = None
    image: str = OSWORLD_DOCKER_IMAGE
    memory: str = "4G"
    cpus: int = 4
    disk_size: str = "32G"
    pull_policy: str = "missing"

    def ensure_resource(self, config: VMConfig) -> None:
        """Validate or prepare the base qcow2 image before launch.

        Override in subclasses to add auto-download behaviour (e.g. OSWorld).
        Default implementation raises if path_to_vm is not set.
        """
        if self.path_to_vm is None:
            raise ValueError(
                "path_to_vm must be set on LocalDockerVMBackend. "
                "Provide a path to an existing qcow2 image, or use a subclass "
                "that handles image acquisition (e.g. OSWorldDockerVMBackend)."
            )

    def launch(self, config: VMConfig) -> LocalDockerVM:
        """Ensure image exists, then start a Docker container and return a live handle.

        Blocks until the in-VM HTTP agent is reachable.
        """
        self.ensure_resource(config)

        docker_config = DockerConfig(
            path_to_vm=Path(self.path_to_vm),  # type: ignore[arg-type]
            image=self.image,
            memory=self.memory,
            cpus=self.cpus,
            disk_size=self.disk_size,
            pull_policy=self.pull_policy,
        )
        manager = DockerManager(docker_config)
        manager.start()
        logger.info("Docker VM launched at endpoint http://localhost:%d", manager.server_port)
        return LocalDockerVM(manager)

    def close(self) -> None:
        """No-op at the backend level — each VM is stopped individually via vm.stop()."""
        pass
