"""cube-vm-backend: VM backend implementations for CUBE desktop-automation benchmarks."""

from cube_vm_backend.local import LocalQEMUVM, LocalQEMUVMBackend
from cube_vm_backend.qemu_manager import QEMUConfig, QEMUManager, ensure_base_image

__all__ = [
    "LocalQEMUVM",
    "LocalQEMUVMBackend",
    "QEMUConfig",
    "QEMUManager",
    "ensure_base_image",
]
