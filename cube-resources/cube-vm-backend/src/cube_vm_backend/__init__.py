"""cube-vm-backend: VM backend implementations for CUBE desktop-automation benchmarks."""

from cube_vm_backend.docker import LocalDockerVM, LocalDockerVMBackend
from cube_vm_backend.docker_manager import DockerConfig, DockerManager
from cube_vm_backend.local import LocalQEMUVM, LocalQEMUVMBackend
from cube_vm_backend.qemu_manager import QEMUConfig, QEMUManager

__all__ = [
    "LocalQEMUVM",
    "LocalQEMUVMBackend",
    "QEMUConfig",
    "QEMUManager",
    "LocalDockerVM",
    "LocalDockerVMBackend",
    "DockerConfig",
    "DockerManager",
]
