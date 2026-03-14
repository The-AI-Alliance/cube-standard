"""LocalQEMUVMBackend — VMBackend implementation using QEMU/KVM on the local host.

Uses a read-only base qcow2 image + per-task copy-on-write overlays.
Reset strategy: delete overlay + reboot (ResetIsolation.RESTART, ~30s).

Image download:
    ensure_resource() downloads the qcow2 from HuggingFace on first use (~4 GB).
    Subsequent calls are no-ops (cached image detected by filename).

Port forwarding:
    SLIRP user-mode networking — no root or bridge required.
    Each VM gets unique host ports forwarded to guest ports 5000/9222/8006/8080.
"""

from __future__ import annotations

import logging
from pathlib import Path

from cube.vm import VM, VMBackend, VMConfig

from cube_vm_backend.qemu_manager import QEMUConfig, QEMUManager, ensure_base_image

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

    def ensure_resource(self, config: VMConfig) -> None:
        """Download the base qcow2 image if not already present.

        Idempotent — subsequent calls are no-ops once the image exists.
        """
        if self.path_to_vm is not None:
            logger.info("Using explicit VM image: %s", self.path_to_vm)
            return
        vm_dir = Path(self.cache_dir)
        base_image = ensure_base_image(vm_dir, config.os_type)
        logger.info("Base image ready: %s", base_image)

    def launch(self, config: VMConfig) -> LocalQEMUVM:
        """Ensure image exists, then start a QEMU VM and return a live handle.

        Blocks until the in-VM HTTP agent is reachable.
        """
        self.ensure_resource(config)

        vm_dir = Path(self.cache_dir)
        if self.path_to_vm is not None:
            base_image = Path(self.path_to_vm)
        else:
            base_image = ensure_base_image(vm_dir, config.os_type)

        qemu_config = QEMUConfig(
            base_image=base_image,
            overlay_dir=vm_dir / "overlays",
            memory=self.memory,
            cpus=self.cpus,
            headless=self.headless,
            screen_width=config.screen_size[0],
            screen_height=config.screen_size[1],
        )
        manager = QEMUManager(qemu_config)
        manager.start()
        logger.info("VM launched at endpoint http://localhost:%d", manager.server_port)
        return LocalQEMUVM(manager)

    def close(self) -> None:
        """No-op at the backend level — each VM is stopped individually via vm.stop()."""
        pass
