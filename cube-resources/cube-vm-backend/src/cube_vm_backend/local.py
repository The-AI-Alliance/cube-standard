"""LocalQEMUVMBackend — VMBackend implementation using QEMU/KVM on the local host.

Uses a read-only base qcow2 image + per-task copy-on-write overlays.
Reset strategy: delete overlay + reboot (ResetIsolation.RESTART, ~30s).

Image:
    path_to_vm must point to an existing qcow2 base image.
    Benchmarks that need auto-download (e.g. OSWorld) subclass this backend
    and override ensure_resource() to fetch the image before launch.

Port forwarding:
    SLIRP user-mode networking — no root or bridge required.
    Each VM gets unique host ports forwarded to guest ports 5000/9222/8006/8080.
"""

from __future__ import annotations

import logging
from pathlib import Path

from cube.vm import VM, VMBackend, VMConfig

from cube_vm_backend.qemu_manager import QEMUConfig, QEMUManager

logger = logging.getLogger(__name__)

# Default cache location — can be overridden via cache_dir field
_DEFAULT_CACHE_DIR = Path.home() / ".cube" / "vm_data"


class LocalQEMUVM(VM):
    """Runtime handle to a QEMU/KVM VM managed by LocalQEMUVMBackend.

    Not serializable. The caller owns the lifecycle via stop().
    """

    def __init__(self, manager: QEMUManager) -> None:
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

        Implementation: delete overlay + create fresh overlay + reboot QEMU.
        This provides RESTART isolation (~30s). The ``name`` argument is
        accepted for API compatibility but ignored — only one snapshot state
        (the base image) is available with the overlay strategy.
        """
        logger.info("Restoring VM snapshot '%s' (overlay reset)", name)
        self._manager.reset()

    def stop(self) -> None:
        """Shut down the VM and clean up overlay and socket files."""
        self._manager.stop()


class LocalQEMUVMBackend(VMBackend):
    """VMBackend that runs QEMU/KVM directly on the local host.

    Attributes
    ----------
    cache_dir : str
        Directory for the base qcow2 image and per-task overlays.
    path_to_vm : str | None
        Explicit path to a pre-existing qcow2 base image.
        If None, the image is auto-downloaded to cache_dir on first use.
    headless : bool
        Suppress the graphical display (default True).
    memory : str
        RAM allocation passed to QEMU -m (e.g. "4G").
    cpus : int
        Number of vCPUs.
    """

    cache_dir: str = str(_DEFAULT_CACHE_DIR)
    path_to_vm: str | None = None
    headless: bool = True
    memory: str = "4G"
    cpus: int = 4
    arch: str = "x86_64"

    def ensure_resource(self, config: VMConfig) -> None:
        """Validate or prepare the base qcow2 image before launch.

        Override in subclasses to add auto-download behaviour (e.g. OSWorld).
        Default implementation raises if path_to_vm is not set.
        """
        if self.path_to_vm is None:
            raise ValueError(
                "path_to_vm must be set on LocalQEMUVMBackend. "
                "Provide a path to an existing qcow2 image, or use a subclass "
                "that handles image acquisition (e.g. OSWorldQEMUVMBackend)."
            )

    def launch(self, config: VMConfig) -> LocalQEMUVM:
        """Ensure image exists, then start a QEMU VM and return a live handle.

        Blocks until the in-VM HTTP agent is reachable.
        """
        self.ensure_resource(config)

        vm_dir = Path(self.cache_dir)
        base_image = Path(self.path_to_vm)  # type: ignore[arg-type]

        qemu_config = QEMUConfig(
            base_image=base_image,
            overlay_dir=vm_dir / "overlays",
            memory=self.memory,
            cpus=self.cpus,
            headless=self.headless,
            screen_width=config.screen_size[0],
            screen_height=config.screen_size[1],
            arch=self.arch,
        )
        manager = QEMUManager(qemu_config)
        manager.start()
        logger.info("VM launched at endpoint http://localhost:%d", manager.server_port)
        return LocalQEMUVM(manager)

    def close(self) -> None:
        """No-op at the backend level — each VM is stopped individually via vm.stop()."""
        pass
