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
_WAIT_TIMEOUT = 1800  # seconds to wait for VM to boot (30 min; software emulation is slow)
_WAIT_INTERVAL = 5    # seconds between health-check polls


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
        ram_size:      RAM to allocate to the QEMU VM (e.g. "4G", "2G"). Lower this
                       on memory-constrained hosts such as Colima with default 2 GB.
        cpu_cores:     Number of vCPUs to pass to QEMU.
        disk_size:     Overlay disk size (not the qcow2 image itself).
        cache_dir:     Directory for any cached backend artifacts.
    """

    vm_image_path: str
    docker_image: str
    headless: bool = True
    ram_size: str = "4G"
    cpu_cores: int = 4
    disk_size: str = "32G"
    cache_dir: str = Field(default_factory=lambda: str(get_cache_dir("vm")))

    def launch(self, config: VMConfig) -> "LocalQEMUVM":
        """Start a new Docker container and wait until the Flask server is ready.

        The qcow2 disk is mounted *read-only*, so every container launch
        produces a fresh VM state.  No explicit snapshot management is needed.
        """
        import docker

        environment: dict[str, str] = {
            "DISK_SIZE": self.disk_size,
            "RAM_SIZE": self.ram_size,
            "CPU_CORES": str(self.cpu_cores),
        }

        devices: list[str] = []
        if os.path.exists("/dev/kvm"):
            devices.append("/dev/kvm")
            logger.info("KVM device found — using hardware acceleration")
        else:
            environment["KVM"] = "N"
            logger.warning("KVM not available — VM will run without hardware acceleration (slow)")

        # When ip_tables is unavailable (e.g. ARM-emulated containers), qemus/qemu-docker
        # falls back to QEMU usermode networking.  In that mode hostfwd rules handle port
        # forwarding and port 5000 (Flask control server) must be added explicitly via
        # USER_PORTS; on tap/bridge networking this is handled by iptables instead.
        environment["USER_PORTS"] = str(_FLASK_PORT)

        # Mount the parent directory instead of the file directly.  Some Docker
        # runtimes (e.g. Colima on macOS) silently create an empty directory when
        # a host *file* is bind-mounted into a container; mounting the containing
        # directory avoids this problem.
        image_abs = os.path.abspath(self.vm_image_path)
        image_dir = os.path.dirname(image_abs)
        image_name = os.path.basename(image_abs)

        # Bypass the container's tini init (tini -s fails when PR_SET_CHILD_SUBREAPER
        # is unavailable, e.g. on amd64-emulated containers on ARM hosts).
        #
        # install.sh uses `find -type f` to locate System.qcow2, so a symlink is
        # insufficient.  Instead we create a real qcow2 overlay at /boot.qcow2
        # that references the host-mounted image as its backing file.  install.sh's
        # findFile("qcow2") finds /boot.qcow2 as a regular file and uses it as the
        # boot disk; all VM writes go to the in-container overlay while the host
        # qcow2 stays read-only.
        entrypoint = [
            "/bin/bash", "-c",
            (
                f"qemu-img create -f qcow2 "
                f"-b /vm_image_src/{image_name} -F qcow2 /boot.qcow2 "
                f"&& exec /run/entry.sh"
            ),
        ]

        client = docker.from_env()
        container = client.containers.run(
            self.docker_image,
            environment=environment,
            cap_add=["NET_ADMIN"],
            devices=devices,
            entrypoint=entrypoint,
            volumes={
                image_dir: {
                    "bind": "/vm_image_src",
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
