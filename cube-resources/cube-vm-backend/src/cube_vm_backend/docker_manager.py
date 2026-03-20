"""Docker-wrapped QEMU VM lifecycle manager.

Runs QEMU inside a Docker container (``happysixd/osworld-docker``), which is the
macOS-friendly alternative to bare-metal QEMU. Docker handles the QEMU installation
and setup; the host only needs Docker Desktop.

The container requires the qcow2 base image mounted read-only at ``/System.qcow2``.
It starts the same Flask guest agent on port 5000 as the bare-metal QEMU approach,
so the rest of the stack (ComputerBase, OSWorldTask, etc.) works unchanged.

Port strategy: explicit host ports allocated via O_EXCL file reservation (same
race-free approach as QEMUManager) — Docker does not auto-assign when ports are
passed explicitly.

Snapshot strategy: stop + remove container, start fresh container with the same
read-only qcow2 mount (NEW_INSTANCE isolation, ~30-60s).
"""

import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import docker
import docker.errors
import requests
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from cube_vm_backend.qemu_manager import _release_port_reservation, _reserve_free_port

logger = logging.getLogger(__name__)

_VM_READY_TIMEOUT = 300  # seconds to wait for guest /screenshot endpoint
_VM_READY_POLL_INTERVAL = 2  # seconds between readiness polls

_retry_launch = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    reraise=True,
    retry=retry_if_not_exception_type(RuntimeError),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)

# The OSWorld Docker image that wraps QEMU
OSWORLD_DOCKER_IMAGE = "happysixd/osworld-docker"


class DockerConfig:
    """Configuration for a Docker-wrapped QEMU virtual machine.

    Parameters
    ----------
    path_to_vm : Path
        Path to the read-only base qcow2 disk image. Mounted into the container
        at ``/System.qcow2``.
    image : str
        Docker image name. Defaults to ``happysixd/osworld-docker``.
    memory : str
        RAM allocation passed to the container via ``RAM_SIZE`` env var (e.g. ``"4G"``).
    cpus : int
        Number of vCPUs passed via ``CPU_CORES`` env var.
    disk_size : str
        Disk size passed via ``DISK_SIZE`` env var (e.g. ``"32G"``).
    pull_policy : str
        ``"always"`` — always pull before launch.
        ``"missing"`` — pull only if not present locally (default).
        ``"never"`` — never pull; fail if not present.
    """

    def __init__(
        self,
        path_to_vm: Path,
        image: str = OSWORLD_DOCKER_IMAGE,
        memory: str = "4G",
        cpus: int = 4,
        disk_size: str = "32G",
        pull_policy: str = "missing",
    ) -> None:
        self.path_to_vm = Path(path_to_vm)
        self.image = image
        self.memory = memory
        self.cpus = cpus
        self.disk_size = disk_size
        self.pull_policy = pull_policy


class DockerManager:
    """Manages the lifecycle of a QEMU VM running inside a Docker container.

    Usage::

        manager = DockerManager(config)
        manager.start()   # pull image, allocate ports, launch container, wait for readiness
        manager.reset()   # stop + remove container, start fresh container
        manager.stop()    # stop + remove container, release ports, close Docker client

    After ``start()`` the following properties are available:

    - :attr:`server_port`   — host port forwarded to guest :5000  (Flask agent)
    - :attr:`chromium_port` — host port forwarded to guest :9222  (Chromium DevTools)
    - :attr:`vnc_port`      — host port forwarded to guest :8006  (VNC/noVNC)
    - :attr:`vlc_port`      — host port forwarded to guest :8080  (VLC HTTP)

    Note: unlike the QEMU approach, port values are stable across ``reset()`` calls
    because ports are reserved before the container starts and held for the lifetime
    of this manager.

    Parameters
    ----------
    config : DockerConfig
        Container configuration (qcow2 path, image, memory, CPUs).
    """

    def __init__(self, config: DockerConfig) -> None:
        self.config = config
        self._client: Optional[docker.DockerClient] = None
        self._container: Optional[docker.models.containers.Container] = None

        self._server_port: Optional[int] = None
        self._chromium_port: Optional[int] = None
        self._vnc_port: Optional[int] = None
        self._vlc_port: Optional[int] = None

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def server_port(self) -> int:
        if self._server_port is None:
            raise RuntimeError("Container not started — call start() first")
        return self._server_port

    @property
    def chromium_port(self) -> int:
        if self._chromium_port is None:
            raise RuntimeError("Container not started — call start() first")
        return self._chromium_port

    @property
    def vnc_port(self) -> int:
        if self._vnc_port is None:
            raise RuntimeError("Container not started — call start() first")
        return self._vnc_port

    @property
    def vlc_port(self) -> int:
        if self._vlc_port is None:
            raise RuntimeError("Container not started — call start() first")
        return self._vlc_port

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Pull image (if needed), allocate ports, launch container, wait for readiness."""
        self._client = docker.from_env()
        self._pull_image_if_needed()

        self._server_port = _reserve_free_port(5000)
        self._chromium_port = _reserve_free_port(9222)
        self._vnc_port = _reserve_free_port(8006)
        self._vlc_port = _reserve_free_port(8080)

        self._launch_container()
        self._wait_for_ready()

    def reset(self) -> None:
        """Restore the VM to its initial state.

        Stops and removes the running container, then launches a fresh one using
        the same read-only qcow2 mount (NEW_INSTANCE isolation, ~30-60s).
        Ports are preserved — they were reserved before the first launch.
        """
        self._stop_container()
        self._launch_container()
        self._wait_for_ready()

    def stop(self) -> None:
        """Stop the container, release port reservations, close Docker client."""
        self._stop_container()
        for port in (self._server_port, self._chromium_port, self._vnc_port, self._vlc_port):
            _release_port_reservation(port)
        self._server_port = None
        self._chromium_port = None
        self._vnc_port = None
        self._vlc_port = None
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pull_image_if_needed(self) -> None:
        """Pull the Docker image according to pull_policy."""
        if self.config.pull_policy == "never":
            return
        if self.config.pull_policy == "always" or not _image_exists(self._client, self.config.image):
            logger.info("Pulling image %s ...", self.config.image)
            try:
                self._client.images.pull(self.config.image)
            except docker.errors.APIError as exc:
                raise RuntimeError(f"Failed to pull image '{self.config.image}': {exc}") from exc

    @_retry_launch
    def _launch_container(self) -> None:
        """Launch the Docker container with the qcow2 mounted and explicit port bindings."""
        name = f"cube_vm_{uuid.uuid4().hex[:8]}"
        env = _build_env(self.config)
        devices = ["/dev/kvm"] if os.path.exists("/dev/kvm") else []

        try:
            self._container = self._client.containers.run(  # nosemgrep: docker-arbitrary-container-run
                self.config.image,  # trusted: DockerConfig is authored by the benchmark developer
                detach=True,
                name=name,
                environment=env,
                cap_add=["NET_ADMIN"],
                devices=devices,
                volumes={
                    str(self.config.path_to_vm.resolve()): {
                        "bind": "/System.qcow2",
                        "mode": "ro",
                    }
                },
                ports={
                    8006: self._vnc_port,
                    5000: self._server_port,
                    9222: self._chromium_port,
                    8080: self._vlc_port,
                },
            )
        except docker.errors.APIError as exc:
            raise RuntimeError(f"Failed to launch container from '{self.config.image}': {exc}") from exc

        # Wait for 'running' state
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            self._container.reload()
            if self._container.status == "running":
                break
            time.sleep(0.5)
        else:
            self._stop_container()
            raise RuntimeError(f"Container did not reach 'running' state within 30s (status: {self._container.status})")
        logger.info(
            "Container '%s' launched (server_port=%d, kvm=%s)",
            name,
            self._server_port,
            bool(devices),
        )

    def _wait_for_ready(self) -> None:
        """Poll the guest's /screenshot endpoint until the VM inside the container is ready."""
        deadline = time.time() + _VM_READY_TIMEOUT
        url = f"http://localhost:{self._server_port}/screenshot"
        logger.info("Waiting for guest agent to be ready at %s ...", url)
        while time.time() < deadline:
            try:
                resp = requests.get(url, timeout=(5, 5))
                if resp.status_code == 200:
                    logger.info("Guest agent is ready")
                    return
            except Exception:
                pass
            logger.debug("Guest agent not ready yet, retrying in %ds...", _VM_READY_POLL_INTERVAL)
            time.sleep(_VM_READY_POLL_INTERVAL)
        raise TimeoutError(f"Guest agent failed to become ready within {_VM_READY_TIMEOUT}s")

    def _stop_container(self) -> None:
        """Stop and remove the container (ports remain reserved)."""
        if self._container is None:
            return
        try:
            self._container.stop(timeout=10)
        except docker.errors.NotFound:
            pass
        except docker.errors.APIError as exc:
            logger.warning("Container stop failed: %s", exc)
        try:
            self._container.remove(force=True)
        except docker.errors.NotFound:
            pass
        except docker.errors.APIError as exc:
            logger.warning("Container remove failed: %s", exc)
        self._container = None


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _build_env(config: DockerConfig) -> dict[str, str]:
    """Build the environment dict passed to the container."""
    env: dict[str, str] = {
        "DISK_SIZE": config.disk_size,
        "RAM_SIZE": config.memory,
        "CPU_CORES": str(config.cpus),
    }
    if not os.path.exists("/dev/kvm"):
        env["KVM"] = "N"
        logger.warning("KVM not available — container will run QEMU without hardware acceleration (slow)")
    return env


def _image_exists(client: docker.DockerClient, image: str) -> bool:
    """Return True if the image is present in the local Docker daemon."""
    try:
        client.images.get(image)
        return True
    except docker.errors.ImageNotFound:
        return False
