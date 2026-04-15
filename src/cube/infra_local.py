"""
LocalInfraConfig — runs CUBE resources locally via QEMU or Docker.

No cloud credentials required. This is the default infra and the reference
implementation of the InfraConfig interface.

Supported resource types:
    VMResourceConfig    — boots a QEMU VM from a local qcow2 image.
    DockerImageConfig   — runs a Docker container from a pulled image.

Image storage: CUBE_LOCAL_IMAGE_DIR env var, defaults to ~/.cube/images/.
Active resource tracking: ~/.cube/active.json.
  Each entry has a "type" field: "vm" (QEMU) or "docker".
  VM entries carry: pid, overlay_path, port, endpoint, created_at, expires_at.
  Docker entries carry: container_name, port_map, endpoint, created_at, expires_at.

System requirements (VM):
    qemu-img             — image conversion (brew install qemu / apt install qemu-utils)
    qemu-system-x86_64  — VM execution  (brew install qemu / apt install qemu-system-x86)

System requirements (Docker):
    docker               — container runtime (docker.com/get-started)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from cube.resource import (
    DockerImageConfig,
    InfraConfig,
    ResourceConfig,
    ResourceHandle,
    ResourceNotReadyError,
    UnsupportedResourceType,
    VMResourceConfig,
)

logger = logging.getLogger(__name__)

_IMAGE_DIR = Path(os.environ.get("CUBE_LOCAL_IMAGE_DIR", str(Path.home() / ".cube" / "images")))
_ACTIVE_JSON = Path(os.environ.get("CUBE_CACHE_DIR", str(Path.home() / ".cube"))) / "active.json"


# ── Active resource store (PID-based, file-backed) ────────────────────────────


def _load_active() -> dict:
    if not _ACTIVE_JSON.exists():
        return {}
    with open(_ACTIVE_JSON) as f:
        return json.load(f)


def _save_active(data: dict) -> None:
    _ACTIVE_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(_ACTIVE_JSON, "w") as f:
        json.dump(data, f, indent=2)


def _register_active(entry_id: str, entry: dict) -> None:
    data = _load_active()
    data[entry_id] = entry
    _save_active(data)


def _deregister_active(entry_id: str) -> None:
    data = _load_active()
    data.pop(entry_id, None)
    _save_active(data)


# ── VM Helpers ───────────────────────────────────────────────────────────────────


def _free_port(start: int = 15000, count: int = 200) -> int:
    for port in range(start, start + count):
        try:
            with socket.socket() as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No free port in {start}–{start + count - 1}")


def _wait_for_http(endpoint: str, timeout: int = 300) -> None:
    """Poll {endpoint}/screenshot until HTTP 200 or timeout."""
    try:
        import requests as _requests
    except ImportError as e:
        raise ImportError("cube-infra-local requires 'requests'. Install it with: pip install requests") from e

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = _requests.get(f"{endpoint}/screenshot", timeout=5)
            if r.status_code == 200 and len(r.content) > 0:
                logger.info("Guest agent ready at %s (%d bytes)", endpoint, len(r.content))
                return
        except Exception:
            pass
        remaining = int(deadline - time.time())
        logger.debug("Waiting for guest agent… (%ds left)", remaining)
        time.sleep(5)
    raise TimeoutError(f"Guest agent not ready after {timeout}s at {endpoint}")


def _download(url: str, dest: Path) -> None:
    """Download url to dest with a tqdm progress bar.

    Handles .zip archives — extracts the first .qcow2 file found inside.
    """
    import zipfile

    try:
        import requests as _requests
        from tqdm import tqdm
    except ImportError as e:
        raise ImportError("cube-infra-local requires 'requests' and 'tqdm'.") from e

    dest.parent.mkdir(parents=True, exist_ok=True)

    # Download to a temp file first, then move on success.
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    logger.info("Downloading %s → %s", url, dest)
    with _requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) or None
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))

    # Unzip if needed.
    if url.endswith(".zip") or zipfile.is_zipfile(tmp):
        logger.info("Extracting zip archive…")
        with zipfile.ZipFile(tmp) as zf:
            qcow2_names = [n for n in zf.namelist() if n.endswith(".qcow2")]
            if not qcow2_names:
                raise RuntimeError(f"No .qcow2 file found in archive from {url}")
            qcow2_name = qcow2_names[0]
            logger.info("Extracting %s", qcow2_name)
            zf.extract(qcow2_name, dest.parent)
            extracted = dest.parent / qcow2_name
            extracted.rename(dest)
        tmp.unlink(missing_ok=True)
    else:
        tmp.rename(dest)

    logger.info("Downloaded %s (%.1f GB)", dest.name, dest.stat().st_size / 1024**3)


def _convert_to_qcow2(src: Path, dst: Path) -> None:
    """Convert src image to qcow2 format using qemu-img."""
    if dst.exists():
        logger.info("Converted image already exists at %s — skipping", dst)
        return
    logger.info("Converting %s → %s", src.name, dst.name)
    subprocess.run(
        ["qemu-img", "convert", "-O", "qcow2", str(src), str(dst)],
        check=True,
    )
    logger.info("Conversion done (%.1f GB)", dst.stat().st_size / 1024**3)


def _create_overlay(base_image: Path, run_id: str) -> Path:
    """Create a qcow2 copy-on-write overlay over base_image for this run."""
    overlay_dir = base_image.parent / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    overlay = overlay_dir / f"{base_image.stem}_{run_id[:8]}.qcow2"
    subprocess.run(
        [
            "qemu-img",
            "create",
            "-f",
            "qcow2",
            "-b",
            str(base_image.resolve()),
            "-F",
            "qcow2",
            str(overlay),
        ],
        check=True,
        capture_output=True,
    )
    logger.debug("Created overlay %s", overlay)
    return overlay


# ── Docker Helpers ───────────────────────────────────────────────────────────────────


def _stop_docker_container(container_name: str) -> None:
    """Stop and remove a Docker container by name. Idempotent."""
    for cmd in (["docker", "stop", container_name], ["docker", "rm", container_name]):
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 and "No such container" not in result.stderr:
            logger.warning(f"{' '.join(cmd)} failed: {result.stderr.strip()}")
    logger.info(f"Stopped and removed Docker container {container_name}")


def _wait_for_docker_http(url: str, timeout: int = 60) -> None:
    """Poll url until HTTP status < 500 or timeout expires.

    Accepts any 1xx–4xx response as "container is up" — only 5xx and
    connection errors are treated as not-yet-ready.
    """
    import urllib.error
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(urllib.request.Request(url), timeout=5) as resp:
                if resp.status < 500:
                    logger.info(f"Container ready at {url} (status {resp.status})")
                    return
        except urllib.error.HTTPError as e:
            if e.code < 500:
                logger.info(f"Container ready at {url} (status {e.code}: {e.reason})")
                return
        except Exception:
            pass
        remaining = int(deadline - time.time())
        logger.debug(f"Waiting for container… ({remaining}s left)")
        time.sleep(2)
    raise TimeoutError(f"Container not ready after {timeout}s at {url}")


# ── LocalResourceHandle ───────────────────────────────────────────────────────


@dataclass
class LocalResourceHandle(ResourceHandle):
    """ResourceHandle for a locally-running QEMU VM."""

    _entry_id: str = field(default="", repr=False)
    _qemu_proc: subprocess.Popen | None = field(default=None, repr=False)
    _overlay_path: Path | None = field(default=None, repr=False)

    def close(self) -> None:
        """Terminate the QEMU process and remove the overlay image."""
        if self._qemu_proc is not None:
            try:
                self._qemu_proc.terminate()
                self._qemu_proc.wait(timeout=10)
            except Exception:
                try:
                    self._qemu_proc.kill()
                except Exception:
                    pass
            self._qemu_proc = None
            logger.info("Terminated QEMU process for run %s", self.run_id[:8])

        if self._overlay_path and self._overlay_path.exists():
            self._overlay_path.unlink()
            logger.debug("Removed overlay %s", self._overlay_path)
            self._overlay_path = None

        _deregister_active(self._entry_id)


@dataclass
class LocalDockerResourceHandle(ResourceHandle):
    """ResourceHandle for a locally-running Docker container."""

    _entry_id: str = field(default="", repr=False)
    _container_name: str = field(default="", repr=False)
    port_map: dict[int, int] = field(default_factory=dict, repr=False)
    """Maps container port → allocated host port."""

    def close(self) -> None:
        """Stop and remove the Docker container."""
        if self._container_name:
            _stop_docker_container(self._container_name)
        _deregister_active(self._entry_id)


# ── LocalInfraConfig ──────────────────────────────────────────────────────────


class LocalInfraConfig(InfraConfig):
    """Runs CUBE resources locally — QEMU VMs or Docker containers.

    No cloud credentials required. The default infra when no InfraConfig is provided.

    Fields:
        cpu_cores:    vCPU count for launched VMs (default 4).
        ram_gb:       RAM in GB for launched VMs (default 4).
        screen_width: Horizontal resolution (default 1920).
        screen_height: Vertical resolution (default 1080).
        headless:     Run QEMU without a display window (default True).
        enable_kvm:   Use KVM hardware acceleration if available (default True).
        image_dir:    Where to store downloaded/converted images.
                      Defaults to CUBE_LOCAL_IMAGE_DIR or ~/.cube/images.
    """

    cpu_cores: int = 4
    ram_gb: int = 4
    screen_width: int = 1920
    screen_height: int = 1080
    headless: bool = True
    enable_kvm: bool = True
    image_dir: str = str(_IMAGE_DIR)

    # ── InfraConfig interface ─────────────────────────────────────────────────

    def fingerprint(self) -> str:
        return "local"

    def capabilities(self) -> set[str]:
        caps: set[str] = set()
        if shutil.which("qemu-system-x86_64"):
            # kvm capability only when the device is accessible
            kvm_available = self.enable_kvm and Path("/dev/kvm").exists()
            if kvm_available:
                caps.add("kvm")
            else:
                # QEMU TCG (no KVM) can still run VMs, just slower.
                # We advertise "kvm" only if hardware acceleration is available.
                pass
        if shutil.which("docker"):
            caps.add("docker")
        return caps

    def provision(self, resource: ResourceConfig) -> None:
        """Download/prepare the image locally, then register it.

        For VMResourceConfig: downloads the qcow2 (or zip of qcow2) from
        source_url and stores it at image_dir/{resource.name}.qcow2.
        Idempotent — skips download/convert if the image already exists.

        For DockerImageConfig: pulls the image from the registry.
        Idempotent — docker pull is a no-op when the image is already present.
        """
        if isinstance(resource, DockerImageConfig):
            logger.info(f"Pulling Docker image {resource.image!r}...")
            subprocess.run(["docker", "pull", resource.image], check=True)
            self.register(resource, {"image": resource.image})
            logger.info(f"Pulled Docker image {resource.image!r}")
            return

        if not isinstance(resource, VMResourceConfig):
            raise UnsupportedResourceType(resource, self)

        image_dir = Path(self.image_dir)
        image_dir.mkdir(parents=True, exist_ok=True)
        dest = image_dir / f"{resource.name}.qcow2"

        if dest.exists():
            logger.info("Image already at %s (%.1f GB) — skipping download", dest, dest.stat().st_size / 1024**3)
        elif resource.source_url:
            # Download to a staging path, then convert to qcow2 if needed.
            raw_dest = image_dir / f"{resource.name}.raw"
            _download(resource.source_url, raw_dest)
            if raw_dest.suffix != ".qcow2" or raw_dest != dest:
                _convert_to_qcow2(raw_dest, dest)
                if raw_dest != dest:
                    raw_dest.unlink(missing_ok=True)
        else:
            raise ValueError(
                f"Cannot provision {resource.name!r}: no source_url provided "
                f"and image not found at {dest}.\n"
                f'  Call infra.register(resource, {{"image_path": "/path/to/image.qcow2"}}) '
                f"to register an existing image."
            )

        self.register(resource, {"image_path": str(dest)})

    def launch(self, resource: ResourceConfig) -> ResourceHandle:
        """Instantiate a resource locally and return a live handle.

        For VMResourceConfig: boots a QEMU VM with a copy-on-write overlay.
        For DockerImageConfig: runs the container with dynamically-allocated host ports.

        Reads resource_info from the ProvisionStore. Raises ResourceNotReadyError
        if no entry is found (i.e. provision() or register() was never called).

        run_id is generated internally; TTL resolves as
        self.default_ttl_seconds ?? resource.default_ttl_seconds.
        """
        if isinstance(resource, DockerImageConfig):
            return self._launch_docker(resource)
        elif isinstance(resource, VMResourceConfig):
            return self._launch_vm(resource)
        else:
            raise UnsupportedResourceType(resource, self)

    def _launch_vm(self, resource: VMResourceConfig) -> LocalResourceHandle:
        """Boot a QEMU VM and return a handle with the guest agent endpoint.

        Reads image_path from the ProvisionStore. Raises ResourceNotReadyError
        if no entry is found.

        The VM is isolated via a copy-on-write overlay so the base image is
        never modified. run_id is generated internally; TTL resolves as
        self.default_ttl_seconds ?? resource.default_ttl_seconds.
        """
        from cube.provision_store import ProvisionStore

        resource_info = ProvisionStore().get(resource, self)
        if resource_info is None:
            raise ResourceNotReadyError(resource, self)

        image_path = Path(resource_info["image_path"])
        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found at {image_path}. Re-run infra.provision(resource) to re-download."
            )

        run_id = str(uuid.uuid4())
        port = _free_port()
        overlay = _create_overlay(image_path, run_id)
        entry_id = f"cube-{run_id[:8]}-vm-{uuid.uuid4().hex[:6]}"

        cmd = [
            "qemu-system-x86_64",
            "-m",
            f"{self.ram_gb}G",
            "-smp",
            str(self.cpu_cores),
            "-drive",
            f"file={overlay},format=qcow2,if=virtio",
            "-netdev",
            f"user,id=net0,hostfwd=tcp:127.0.0.1:{port}-:5000",
            "-device",
            "virtio-net-pci,netdev=net0",
            "-vga",
            "virtio",
        ]
        if self.headless:
            cmd += ["-display", "none"]
        if self.enable_kvm and Path("/dev/kvm").exists():
            cmd += ["-enable-kvm", "-cpu", "host"]

        logger.info(f"Starting QEMU VM for {resource.name!r} (run={run_id[:8]}, port={port})")
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        endpoint = f"http://127.0.0.1:{port}"
        effective_ttl = (
            self.default_ttl_seconds if self.default_ttl_seconds is not None else resource.default_ttl_seconds
        )
        created_at = datetime.utcnow()
        expires_at = created_at + timedelta(seconds=effective_ttl) if effective_ttl else None

        _register_active(
            entry_id,
            {
                "type": "vm",
                "run_id": run_id,
                "resource_name": resource.name,
                "infra_fingerprint": self.fingerprint(),
                "pid": proc.pid,
                "overlay_path": str(overlay),
                "port": port,
                "endpoint": endpoint,
                "created_at": created_at.isoformat(),
                "expires_at": expires_at.isoformat() if expires_at else None,
            },
        )

        try:
            _wait_for_http(endpoint)
        except TimeoutError:
            proc.kill()
            overlay.unlink(missing_ok=True)
            _deregister_active(entry_id)
            raise

        return LocalResourceHandle(
            run_id=run_id,
            resource=resource,
            infra=self,
            endpoint=endpoint,
            created_at=created_at,
            expires_at=expires_at,
            _entry_id=entry_id,
            _qemu_proc=proc,
            _overlay_path=overlay,
        )

    def _launch_docker(self, resource: DockerImageConfig) -> LocalDockerResourceHandle:
        """Run a Docker container and return a handle with the primary endpoint.

        Allocates a free host port for each container port declared in resource.ports.
        The endpoint is http://127.0.0.1:{first_host_port}. A health check is
        performed on {endpoint}{resource.health_check_path} unless health_check_path
        is empty. On failure the container is removed and the error is re-raised.
        """
        from cube.provision_store import ProvisionStore

        resource_info = ProvisionStore().get(resource, self)
        if resource_info is None:
            raise ResourceNotReadyError(resource, self)

        run_id = str(uuid.uuid4())
        container_name = f"cube-{resource.name}-{run_id[:8]}"
        entry_id = f"cube-{run_id[:8]}-docker-{uuid.uuid4().hex[:6]}"

        # Allocate one free host port per declared container port.
        port_map: dict[int, int] = {}
        if resource.ports:
            for container_port in resource.ports:
                port_map[container_port] = _free_port()

        cmd = ["docker", "run", "-d", "--name", container_name]
        for container_port, host_port in port_map.items():
            cmd += ["-p", f"127.0.0.1:{host_port}:{container_port}"]
        if resource.gpu:
            cmd += ["--gpus", "all"]
        cmd.append(resource_info["image"])

        logger.info(f"Starting Docker container {container_name!r} (run={run_id[:8]})")
        subprocess.run(cmd, check=True, capture_output=True)

        endpoint: str | None = None
        if port_map:
            first_host_port = next(iter(port_map.values()))
            endpoint = f"http://127.0.0.1:{first_host_port}"

        effective_ttl = (
            self.default_ttl_seconds if self.default_ttl_seconds is not None else resource.default_ttl_seconds
        )
        created_at = datetime.utcnow()
        expires_at = created_at + timedelta(seconds=effective_ttl) if effective_ttl else None

        _register_active(
            entry_id,
            {
                "type": "docker",
                "run_id": run_id,
                "resource_name": resource.name,
                "infra_fingerprint": self.fingerprint(),
                "container_name": container_name,
                "port_map": {str(k): v for k, v in port_map.items()},
                "endpoint": endpoint,
                "created_at": created_at.isoformat(),
                "expires_at": expires_at.isoformat() if expires_at else None,
            },
        )

        if endpoint and resource.health_check_path:
            health_url = f"{endpoint}{resource.health_check_path}"
            try:
                _wait_for_docker_http(health_url, timeout=resource.health_check_timeout)
            except TimeoutError:
                _stop_docker_container(container_name)
                _deregister_active(entry_id)
                raise

        return LocalDockerResourceHandle(
            run_id=run_id,
            resource=resource,
            infra=self,
            endpoint=endpoint,
            created_at=created_at,
            expires_at=expires_at,
            _entry_id=entry_id,
            _container_name=container_name,
            port_map=port_map,
        )

    def list_active(self, run_id: str | None = None) -> list[ResourceHandle]:
        """Return handles for all active local resources, optionally filtered by run_id.

        Reconstructed from ~/.cube/active.json. Stale entries (dead process or
        stopped container) are cleaned up and excluded from the result.
        """
        data = _load_active()
        handles: list[ResourceHandle] = []
        stale_ids = []

        for entry_id, entry in data.items():
            if entry.get("infra_fingerprint") != self.fingerprint():
                continue
            if run_id and entry.get("run_id") != run_id:
                continue

            entry_type = entry.get("type")
            if entry_type == "docker":
                handle = self._get_docker_handle(entry_id, entry)
            elif entry_type == "vm":
                handle = self._get_vm_handle(entry_id, entry)
            else:
                raise ValueError(f"Unknown active.json entry type: {entry_type!r}")

            if handle is None:
                stale_ids.append(entry_id)
            else:
                handles.append(handle)

        if stale_ids:
            data = _load_active()
            for sid in stale_ids:
                data.pop(sid, None)
            _save_active(data)

        return handles

    def _get_docker_handle(self, entry_id: str, entry: dict) -> LocalDockerResourceHandle | None:
        """Reconstruct a LocalDockerResourceHandle from an active.json entry.

        Returns None if the container is no longer running (stale entry).
        """
        container_name = entry.get("container_name", "")
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", container_name],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or result.stdout.strip() != "true":
            return None

        created_at = datetime.fromisoformat(entry["created_at"])
        expires_at = datetime.fromisoformat(entry["expires_at"]) if entry.get("expires_at") else None
        port_map = {int(k): v for k, v in entry.get("port_map", {}).items()}
        resource = DockerImageConfig(name=entry["resource_name"], image="")

        return LocalDockerResourceHandle(
            run_id=entry["run_id"],
            resource=resource,
            infra=self,
            endpoint=entry.get("endpoint"),
            created_at=created_at,
            expires_at=expires_at,
            _entry_id=entry_id,
            _container_name=container_name,
            port_map=port_map,
        )

    def _get_vm_handle(self, entry_id: str, entry: dict) -> LocalResourceHandle | None:
        """Reconstruct a LocalResourceHandle from an active.json entry.

        Returns None if the QEMU process is no longer alive (stale entry).
        """
        pid = entry.get("pid")
        if pid and not _pid_alive(pid):
            return None

        resource = VMResourceConfig(name=entry["resource_name"])
        created_at = datetime.fromisoformat(entry["created_at"])
        expires_at = datetime.fromisoformat(entry["expires_at"]) if entry.get("expires_at") else None

        return LocalResourceHandle(
            run_id=entry["run_id"],
            resource=resource,
            infra=self,
            endpoint=entry["endpoint"],
            created_at=created_at,
            expires_at=expires_at,
            _entry_id=entry_id,
            _qemu_proc=None,  # can't reconstruct Popen; cleanup uses PID
            _overlay_path=Path(entry["overlay_path"]) if entry.get("overlay_path") else None,
        )

    def cleanup(self, run_id: str) -> None:
        """Tear down all local resources (VMs and Docker containers) for run_id."""
        data = _load_active()
        to_remove = []
        for entry_id, entry in list(data.items()):
            if entry.get("run_id") != run_id:
                continue
            if entry.get("infra_fingerprint") != self.fingerprint():
                continue
            _kill_entry(entry)
            to_remove.append(entry_id)

        for entry_id in to_remove:
            data.pop(entry_id, None)
        if to_remove:
            _save_active(data)
            logger.info("Cleaned up %d resource(s) for run %s", len(to_remove), run_id[:8])

    def cleanup_stale(self, max_age_seconds: int | None = None) -> list[str]:
        """Kill expired resources and resources older than max_age_seconds.

        Returns list of deleted entry_ids.
        """
        data = _load_active()
        now = datetime.utcnow()
        to_remove = []

        for entry_id, entry in list(data.items()):
            if entry.get("infra_fingerprint") != self.fingerprint():
                continue

            expired = False
            expires_at_str = entry.get("expires_at")
            if expires_at_str:
                expired = datetime.fromisoformat(expires_at_str) < now

            too_old = False
            if not expired and max_age_seconds is not None:
                created_at_str = entry.get("created_at")
                if created_at_str:
                    age = (now - datetime.fromisoformat(created_at_str)).total_seconds()
                    too_old = age > max_age_seconds

            if expired or too_old:
                _kill_entry(entry)
                to_remove.append(entry_id)

        for entry_id in to_remove:
            data.pop(entry_id, None)
        if to_remove:
            _save_active(data)
            logger.info("cleanup_stale: removed %d stale resource(s)", len(to_remove))

        return to_remove


# ── Internal helpers ──────────────────────────────────────────────────────────


def _pid_alive(pid: int) -> bool:
    """Return True if the process with the given PID is still running."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _kill_entry(entry: dict) -> None:
    """Tear down a single active.json entry (VM or Docker container)."""
    entry_type = entry.get("type")

    if entry_type == "docker":
        container_name = entry.get("container_name")
        if container_name:
            _stop_docker_container(container_name)

    elif entry_type == "vm":
        pid = entry.get("pid")
        if pid and _pid_alive(pid):
            try:
                os.kill(pid, 15)  # SIGTERM
                time.sleep(1)
                if _pid_alive(pid):
                    os.kill(pid, 9)  # SIGKILL
            except Exception:
                pass
            logger.debug(f"Killed PID {pid}")

        overlay_path = entry.get("overlay_path")
        if overlay_path:
            p = Path(overlay_path)
            if p.exists():
                p.unlink()
                logger.debug(f"Removed overlay {p}")

    else:
        raise ValueError(f"Unknown active.json entry type: {entry_type!r}")
