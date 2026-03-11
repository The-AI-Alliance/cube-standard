"""Local Docker+QEMU VM backend.

Runs a VM inside a Docker container using QEMU hardware acceleration (KVM).
The VM disk image is mounted read-only; stopping and restarting the container
always produces a clean VM state, making container restart the effective
"snapshot restore" mechanism.

Usage:

    from cube_computer_tool.backends.local_qemu import LocalQEMUVMBackend

    backend = LocalQEMUVMBackend(
        vm_image_path="/path/to/ubuntu.qcow2",
        docker_image="happysixd/osworld-docker",
    )
    vm = backend.launch(VMConfig())
    try:
        # vm.endpoint is e.g. "http://localhost:54321"
        ...
    finally:
        vm.stop()
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING

import requests
from pydantic import Field

from cube import get_cache_dir
from cube.vm import VM, VMBackend, VMConfig

if TYPE_CHECKING:
    import docker as docker_module

logger = logging.getLogger(__name__)

_FLASK_PORT = 5000  # Container-side Flask server port
_WAIT_TIMEOUT = 300  # seconds to wait for VM to boot
_WAIT_INTERVAL = 5   # seconds between health-check polls


def _wait_for_vm(endpoint: str, timeout: int = _WAIT_TIMEOUT) -> None:
    """Poll GET /screenshot until the Flask server responds (VM is ready)."""
    deadline = time.monotonic() + timeout
    last_exc: Exception | None = None
    while time.monotonic() < deadline:
        try:
            resp = requests.get(f"{endpoint}/screenshot", timeout=10)
            if resp.status_code == 200:
                logger.info("VM Flask server is ready at %s", endpoint)
                return
        except requests.RequestException as exc:
            last_exc = exc
        time.sleep(_WAIT_INTERVAL)
    raise TimeoutError(
        f"VM Flask server did not become ready within {timeout}s. "
        f"Last error: {last_exc}"
    )


class LocalQEMUVMBackend(VMBackend):
    """Launch a QEMU-based VM inside a Docker container locally.

    Fields:
        vm_image_path: Absolute path to the .qcow2 disk image on the host.
        docker_image:  Docker image tag that bundles QEMU + the Flask control server.
        headless:      Run QEMU without a graphical display (default True).
        cache_dir:     Directory for any cached backend artifacts.
    """

    vm_image_path: str
    docker_image: str
    headless: bool = True
    cache_dir: str = Field(default_factory=lambda: str(get_cache_dir("vm")))

    def launch(self, config: VMConfig) -> "LocalQEMUVM":
        """Start a new Docker container and wait until the Flask server is ready.

        The qcow2 disk is mounted *read-only*, so every container launch
        produces a fresh VM state.  No explicit snapshot management is needed.
        """
        import docker

        environment: dict[str, str] = {
            "DISK_SIZE": "32G",
            "RAM_SIZE": "4G",
            "CPU_CORES": "4",
        }

        devices: list[str] = []
        if os.path.exists("/dev/kvm"):
            devices.append("/dev/kvm")
            logger.info("KVM device found — using hardware acceleration")
        else:
            environment["KVM"] = "N"
            logger.warning("KVM not available — VM will run without hardware acceleration (slow)")

        client = docker.from_env()
        container = client.containers.run(
            self.docker_image,
            environment=environment,
            cap_add=["NET_ADMIN"],
            devices=devices,
            volumes={
                os.path.abspath(self.vm_image_path): {
                    "bind": "/System.qcow2",
                    "mode": "ro",
                }
            },
            ports={f"{_FLASK_PORT}/tcp": None},  # dynamically assigned host port
            detach=True,
        )
        container.reload()

        host_port = container.ports[f"{_FLASK_PORT}/tcp"][0]["HostPort"]
        endpoint = f"http://localhost:{host_port}"
        logger.info("Container started (id=%s), waiting for VM at %s", container.short_id, endpoint)

        try:
            _wait_for_vm(endpoint)
        except TimeoutError:
            container.stop()
            container.remove()
            raise

        return LocalQEMUVM(
            container=container,
            _endpoint=endpoint,
            _backend=self,
            _config=config,
        )

    def close(self) -> None:
        pass


class LocalQEMUVM(VM):
    """Runtime handle to a running Docker+QEMU VM.

    restore_snapshot() stops and removes the current container, then relaunches
    a fresh one via the stored backend.  Because the disk image is mounted
    read-only the new container always starts from the same base state.
    """

    def __init__(
        self,
        container,
        _endpoint: str,
        _backend: LocalQEMUVMBackend,
        _config: VMConfig,
    ) -> None:
        self._container = container
        self._endpoint = _endpoint
        self._backend = _backend
        self._config = _config

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def restore_snapshot(self, name: str) -> None:
        """Restart the container to restore a clean VM state.

        The snapshot *name* is accepted for API compatibility but ignored —
        the qcow2 read-only mount guarantees the same base state on every
        container start.
        """
        logger.info("Restoring snapshot '%s' — restarting container", name)
        self._container.stop()
        self._container.remove()

        new_vm = self._backend.launch(self._config)
        self._container = new_vm._container
        self._endpoint = new_vm._endpoint

    def stop(self) -> None:
        """Stop and remove the Docker container."""
        try:
            self._container.stop()
            self._container.remove()
        except Exception as exc:
            logger.warning("Error stopping container: %s", exc)
