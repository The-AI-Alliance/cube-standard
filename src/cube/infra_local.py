"""
LocalInfraConfig — runs CUBE resources locally via QEMU or Docker.

No cloud credentials required. This is the default infra and the reference
implementation of the InfraConfig interface.

Supported resource types:
    VMResourceConfig      — boots a QEMU VM from a local qcow2 image.
    DockerServiceConfig   — pulls images, sets up volumes, runs launch_script locally.

Image storage: CUBE_LOCAL_IMAGE_DIR env var, defaults to ~/.cube/images/.
Active resource tracking: ~/.cube/active.json (PID + port per run_id entry).

System requirements (VM):
    qemu-img             — image conversion (brew install qemu / apt install qemu-utils)
    qemu-system-x86_64  — VM execution  (brew install qemu / apt install qemu-system-x86)

System requirements (Docker):
    docker               — container runtime (docker.com/get-started)
"""

from __future__ import annotations

import fcntl
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
from typing import Any

from cube.infra_utils import build_volume_setup_script
from cube.resource import (
    DockerServiceConfig,
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


_RATE_LIMIT_MARKERS = ("toomanyrequests", "rate limit", "429")
# Docker Hub rate-limits anonymous pulls (100/6 h per IP; 200/6 h authenticated).
# Workers that share an IP back off (6 attempts, 30 s → 5 min capped) so a burst
# clears instead of every worker failing at once.
_PULL_MAX_ATTEMPTS = 6
_PULL_BACKOFF_BASE = 30.0
_PULL_BACKOFF_CAP = 300.0


# auto-fix(174)↓
def _active_docker_context_host() -> str | None:
    """Best-effort daemon endpoint of the active ``docker context``, or ``None``.

    The Docker SDK honours ``DOCKER_HOST`` but NOT ``docker context``; shelling
    out to the CLI is the only reliable way to recover the endpoint on
    Colima / Docker-Desktop, whose socket is not ``/var/run/docker.sock``.
    """
    try:
        out = subprocess.run(
            ["docker", "context", "inspect", "--format", "{{.Endpoints.docker.Host}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    host = out.stdout.strip()
    return host or None


def _make_docker_client(docker: Any) -> Any:
    """Construct a Docker SDK client robustly across Podman / Colima / Docker Desktop.

    Resolution order:

    1. ``DOCKER_HOST`` if it names a real socket — normalising Podman's
       ``http+unix://<path>`` to the ``unix://<path>`` the SDK accepts. A bare
       ``http+unix://`` / ``unix://`` (no path) is malformed and treated as unset.
    2. The active ``docker context`` endpoint (covers Colima / Docker-Desktop).
    3. ``docker.from_env()``.

    Raises an actionable ``RuntimeError`` if the client cannot reach the daemon,
    instead of the SDK's opaque ``FileNotFoundError``.
    """
    host = os.environ.get("DOCKER_HOST", "").strip()
    if host.startswith("http+unix://"):
        host = host[len("http+") :]  # podman advertises http+unix://; SDK wants unix://
    if host in ("", "unix://", "tcp://"):  # unset or malformed (scheme without a path)
        host = _active_docker_context_host() or ""

    try:
        client = docker.DockerClient(base_url=host) if host else docker.from_env()
        client.ping()
    except Exception as exc:
        raise RuntimeError(
            f"Cannot reach the Docker daemon (tried base_url="
            f"{host or '(docker.from_env)'!r}). Set DOCKER_HOST to a valid socket "
            "or check `docker context ls` — Colima / Docker-Desktop use a "
            "non-default socket the Docker SDK does not auto-discover."
        ) from exc
    return client


# /auto-fix(174)


def _docker_pull(image: str) -> None:
    """``docker pull`` with exponential backoff on registry rate limits.

    Non-rate-limit failures are raised immediately (no retry); the worker
    should not spin on a genuinely missing or unauthorized image.
    """
    for attempt in range(1, _PULL_MAX_ATTEMPTS + 1):
        proc = subprocess.run(["docker", "pull", image], capture_output=True, text=True)
        if proc.returncode == 0:
            return
        combined = f"{proc.stdout}\n{proc.stderr}".lower()
        rate_limited = any(marker in combined for marker in _RATE_LIMIT_MARKERS)
        if not rate_limited or attempt == _PULL_MAX_ATTEMPTS:
            raise RuntimeError(f"docker pull {image!r} failed (exit {proc.returncode}): {proc.stderr.strip()}")
        delay = min(_PULL_BACKOFF_BASE * 2 ** (attempt - 1), _PULL_BACKOFF_CAP)
        logger.warning(
            "docker pull %r rate-limited (attempt %d/%d); retrying in %.0fs",
            image,
            attempt,
            _PULL_MAX_ATTEMPTS,
            delay,
        )
        time.sleep(delay)


# ── Active resource store (PID-based, file-backed) ────────────────────────────


def _load_active() -> dict:
    if not _ACTIVE_JSON.exists():
        return {}
    try:
        with open(_ACTIVE_JSON) as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError, OSError):
        return {}


def _save_active(data: dict) -> None:
    tmp = _ACTIVE_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(_ACTIVE_JSON)


def _register_active(entry_id: str, entry: dict) -> None:
    _ACTIVE_JSON.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _ACTIVE_JSON.with_suffix(".lock")
    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        data = _load_active()
        data[entry_id] = entry
        _save_active(data)


def _deregister_active(entry_id: str) -> None:
    _ACTIVE_JSON.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _ACTIVE_JSON.with_suffix(".lock")
    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        data = _load_active()
        data.pop(entry_id, None)
        _save_active(data)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _free_port(start: int = 15000, count: int = 200) -> int:
    for port in range(start, start + count):
        try:
            with socket.socket() as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No free port in {start}–{start + count - 1}")


def _wait_for_http(endpoint: str, timeout: int = 600) -> None:
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


_OVMF_CODE = Path("/usr/share/OVMF/OVMF_CODE_4M.ms.fd")
_OVMF_VARS = Path("/usr/share/OVMF/OVMF_VARS_4M.ms.fd")


def _launch_swtpm(run_id: str) -> tuple[subprocess.Popen, Path, Path]:
    """Start a swtpm daemon and return (proc, state_dir, socket_path)."""
    if not shutil.which("swtpm"):
        raise RuntimeError("swtpm not found — install with `apt install swtpm swtpm-tools`.")
    state_dir = Path(f"/tmp/cube-swtpm-{run_id[:8]}")
    state_dir.mkdir(parents=True, exist_ok=True)
    sock = state_dir / "sock"
    proc = subprocess.Popen(
        [
            "swtpm",
            "socket",
            "--tpmstate",
            f"dir={state_dir}",
            "--ctrl",
            f"type=unixio,path={sock}",
            "--tpm2",
            "--log",
            f"file={state_dir}/swtpm.log,level=20",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Wait briefly for socket to appear
    for _ in range(50):
        if sock.exists():
            break
        time.sleep(0.1)
    else:
        proc.kill()
        raise RuntimeError(f"swtpm socket never appeared at {sock}")
    return proc, state_dir, sock


def _create_pflash_vars(run_id: str) -> Path:
    """Copy OVMF_VARS to a per-VM writable file."""
    if not _OVMF_CODE.exists() or not _OVMF_VARS.exists():
        raise RuntimeError(f"OVMF firmware not found at {_OVMF_CODE.parent} — install with `apt install ovmf`.")
    dst = Path(f"/tmp/cube-ovmf-vars-{run_id[:8]}.fd")
    shutil.copy(_OVMF_VARS, dst)
    return dst


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


# ── LocalResourceHandle ───────────────────────────────────────────────────────


@dataclass
class LocalResourceHandle(ResourceHandle):
    """ResourceHandle for a locally-running QEMU VM."""

    _entry_id: str = field(default="", repr=False)
    _qemu_proc: subprocess.Popen | None = field(default=None, repr=False)
    _overlay_path: Path | None = field(default=None, repr=False)
    _swtpm_proc: subprocess.Popen | None = field(default=None, repr=False)
    _swtpm_dir: Path | None = field(default=None, repr=False)
    _pflash_vars: Path | None = field(default=None, repr=False)

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

        if self._swtpm_proc is not None:
            try:
                self._swtpm_proc.terminate()
                self._swtpm_proc.wait(timeout=5)
            except Exception:
                try:
                    self._swtpm_proc.kill()
                except Exception:
                    pass
            self._swtpm_proc = None

        if self._swtpm_dir and self._swtpm_dir.exists():
            shutil.rmtree(self._swtpm_dir, ignore_errors=True)
            self._swtpm_dir = None

        if self._pflash_vars and self._pflash_vars.exists():
            self._pflash_vars.unlink()
            self._pflash_vars = None

        if self._overlay_path and self._overlay_path.exists():
            self._overlay_path.unlink()
            logger.debug("Removed overlay %s", self._overlay_path)
            self._overlay_path = None

        _deregister_active(self._entry_id)


# ── LocalDockerServiceHandle ──────────────────────────────────────────────────


@dataclass
class LocalDockerServiceHandle(ResourceHandle):
    """ResourceHandle for a locally-running DockerServiceConfig.

    Tracks the containers started by the launch_script so close() can stop them.
    ``endpoints`` maps service names (from DockerServiceConfig.services) to
    ``http://127.0.0.1:{port}`` URLs — the same ports declared in the resource,
    since the launch_script binds them directly on the host.

    ``container`` returns a ``cube.container.Container`` wrapping the first started
    container — convenient for task-scoped (L3) single-container resources where
    the cube's tool layer needs ``.exec()`` semantics. Only available when exactly
    one container was started; raises otherwise.
    """

    _entry_id: str = field(default="", repr=False)
    _container_ids: list[str] = field(default_factory=list, repr=False)
    _container_cache: object | None = field(default=None, repr=False)

    @property
    def container(self):
        """Return a ``Container`` wrapper for single-container L3 resources.

        Built lazily on first access.  Caller must not call ``stop()`` on the wrapper
        directly — this handle owns the lifecycle; use ``self.close()`` instead.
        """
        if self._container_cache is not None:
            return self._container_cache
        if len(self._container_ids) != 1:
            raise RuntimeError(
                f"LocalDockerServiceHandle.container is only defined for single-container "
                f"resources (got {len(self._container_ids)} containers)."
            )
        # Import lazily to avoid a hard ``docker`` dependency for VM-only usage.
        import docker  # type: ignore

        from cube.local_container import LocalContainer

        client = _make_docker_client(docker)
        docker_container = client.containers.get(self._container_ids[0])
        # remove_on_close=False — this handle, not the wrapper, owns the lifecycle.
        self._container_cache = LocalContainer(docker_container, client, remove_on_close=False)
        return self._container_cache

    def close(self) -> None:
        """Stop and remove all containers started by the launch_script."""
        for cid in self._container_ids:
            for cmd in (["docker", "stop", cid], ["docker", "rm", cid]):
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0 and "No such container" not in result.stderr:
                    logger.warning("%s failed: %s", " ".join(cmd), result.stderr.strip())
        if self._container_ids:
            logger.info("Stopped %d container(s) for run %s", len(self._container_ids), self.run_id[:8])
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
        """Download/prepare the resource locally, then register it.

        For VMResourceConfig: downloads the qcow2 (or zip of qcow2) from
        source_url and stores it at image_dir/{resource.name}.qcow2.
        Idempotent — skips download/convert if the image already exists.

        For DockerServiceConfig: pulls all declared images and runs
        build_volume_setup_script() to download + extract any VolumeSpec data.
        Idempotent — docker pull is a no-op when images are already present.
        """
        if isinstance(resource, DockerServiceConfig):
            return self._provision_docker_service(resource)

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
        For DockerServiceConfig: runs the launch_script locally via bash and
        returns a handle whose endpoints map service names to localhost URLs.
        """
        if isinstance(resource, DockerServiceConfig):
            return self._launch_docker_service(resource)
        if not isinstance(resource, VMResourceConfig):
            raise UnsupportedResourceType(resource, self)

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

        ram_gb = max(self.ram_gb, resource.min_ram_gb or 0)
        cpu_cores = max(self.cpu_cores, resource.min_cpu_cores or 0)

        swtpm_proc: subprocess.Popen | None = None
        swtpm_dir: Path | None = None
        pflash_vars: Path | None = None

        machine = "q35,smm=on" if resource.uefi else "q35"
        cmd = [
            "qemu-system-x86_64",
            "-machine",
            machine,
            "-m",
            f"{ram_gb}G",
            "-smp",
            str(cpu_cores),
        ]
        if resource.uefi:
            pflash_vars = _create_pflash_vars(run_id)
            cmd += [
                "-drive",
                f"if=pflash,format=raw,readonly=on,file={_OVMF_CODE}",
                "-drive",
                f"if=pflash,format=raw,file={pflash_vars}",
            ]
        if resource.tpm:
            swtpm_proc, swtpm_dir, tpm_sock = _launch_swtpm(run_id)
            cmd += [
                "-chardev",
                f"socket,id=chrtpm,path={tpm_sock}",
                "-tpmdev",
                "emulator,id=tpm0,chardev=chrtpm",
                "-device",
                "tpm-tis,tpmdev=tpm0",
            ]
        cmd += [
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

        logger.info(
            "Starting QEMU VM for %r (run=%s, port=%d, ram=%dG, uefi=%s, tpm=%s)",
            resource.name,
            run_id[:8],
            port,
            ram_gb,
            resource.uefi,
            resource.tpm,
        )
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        endpoint = f"http://127.0.0.1:{port}"
        effective_ttl = (
            self.default_ttl_seconds if self.default_ttl_seconds is not None else resource.default_ttl_seconds
        )
        created_at = datetime.utcnow()
        expires_at = created_at + timedelta(seconds=effective_ttl) if effective_ttl else None

        # Persist to active.json for cross-process cleanup.
        _register_active(
            entry_id,
            {
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
            if swtpm_proc is not None:
                swtpm_proc.kill()
            if swtpm_dir and swtpm_dir.exists():
                shutil.rmtree(swtpm_dir, ignore_errors=True)
            if pflash_vars and pflash_vars.exists():
                pflash_vars.unlink()
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
            _swtpm_proc=swtpm_proc,
            _swtpm_dir=swtpm_dir,
            _pflash_vars=pflash_vars,
        )

    def _provision_docker_service(self, resource: DockerServiceConfig) -> None:
        """Pull images and set up VolumeSpec data for a DockerServiceConfig."""
        from cube.provision_store import ProvisionStore

        for image in resource.docker_images:
            logger.info("Pulling Docker image %r…", image)
            _docker_pull(image)

        volume_script = build_volume_setup_script(resource.volumes)
        if volume_script:
            logger.info("Setting up volumes for %r…", resource.name)
            subprocess.run(["bash", "-c", volume_script], check=True)

        ProvisionStore().put(resource, self, {"provisioned": True})

    def _launch_docker_service(self, resource: DockerServiceConfig) -> LocalDockerServiceHandle:
        """Run the launch_script locally and return a handle tracking started containers.

        The launch_script is executed via bash on the local machine — it handles
        ``docker run``, health checks, and warmup exactly as it would on a cloud VM.

        Endpoints: ``{service_name: "http://127.0.0.1:{port}"}`` using the ports
        declared in ``resource.services``.  These are the host ports the launch_script
        binds directly (no SSH tunneling needed locally).

        Container tracking: the launch_script is run with ``CUBE_LAUNCH_ID=<run_id>``
        in its environment.  Scripts built by ``cube.task_infra.build_docker_run_script``
        propagate this into ``docker run --label cube.launch=$CUBE_LAUNCH_ID``, which
        lets us identify only *our* containers via ``docker ps --filter label=…``.
        This is race-safe under concurrent launches (pytest-xdist, Ray workers, etc.)
        where the previous before/after ``docker ps -q`` diff would capture peers'
        containers as our own.
        """
        from cube.provision_store import ProvisionStore

        if ProvisionStore().get(resource, self) is None:
            raise ResourceNotReadyError(resource, self)

        run_id = str(uuid.uuid4())
        entry_id = f"cube-{run_id[:8]}-dss-{uuid.uuid4().hex[:6]}"

        # Snapshot containers before launch so we can fall back to a diff for scripts
        # that don't propagate CUBE_LAUNCH_ID (legacy / hand-rolled launch scripts).
        before = set(subprocess.run(["docker", "ps", "-q"], capture_output=True, text=True, check=True).stdout.split())

        if resource.launch_script:
            logger.info("Running launch_script for %r (run=%s)…", resource.name, run_id[:8])
            env = {**os.environ, "CUBE_LAUNCH_ID": run_id}
            subprocess.run(["bash", "-c", resource.launch_script], check=True, env=env)

        # Prefer label-based identification — race-safe under concurrency.
        label_result = subprocess.run(
            ["docker", "ps", "-q", "--filter", f"label=cube.launch={run_id}"],
            capture_output=True,
            text=True,
            check=True,
        )
        container_ids = label_result.stdout.split()

        if not container_ids:
            # Launch script didn't tag containers — fall back to the before/after diff.
            after = set(
                subprocess.run(["docker", "ps", "-q"], capture_output=True, text=True, check=True).stdout.split()
            )
            container_ids = list(after - before)

        logger.info("launch_script started %d container(s) for %r", len(container_ids), resource.name)

        endpoints = {name: f"http://127.0.0.1:{port}" for name, port in resource.services.items()}
        primary_endpoint = next(iter(endpoints.values()), None) if endpoints else None

        effective_ttl = (
            self.default_ttl_seconds if self.default_ttl_seconds is not None else resource.default_ttl_seconds
        )
        created_at = datetime.utcnow()
        expires_at = created_at + timedelta(seconds=effective_ttl) if effective_ttl else None

        _register_active(
            entry_id,
            {
                "type": "docker_service",
                "run_id": run_id,
                "resource_name": resource.name,
                "infra_fingerprint": self.fingerprint(),
                "container_ids": container_ids,
                "endpoints": endpoints,
                "endpoint": primary_endpoint,
                "created_at": created_at.isoformat(),
                "expires_at": expires_at.isoformat() if expires_at else None,
            },
        )

        return LocalDockerServiceHandle(
            run_id=run_id,
            resource=resource,
            infra=self,
            endpoint=primary_endpoint,
            endpoints=endpoints,
            created_at=created_at,
            expires_at=expires_at,
            _entry_id=entry_id,
            _container_ids=container_ids,
        )

    def list_active(self, run_id: str | None = None) -> list[LocalResourceHandle]:
        """Return handles for all active local resources, optionally filtered by run_id.

        Reconstructed from ~/.cube/active.json.  Supports both VM entries (type
        absent / "vm") and Docker service entries (type "docker_service").
        VM entries whose PID has died are silently dropped and removed.
        """

        data = _load_active()
        handles: list[LocalResourceHandle] = []
        stale_ids = []

        for entry_id, entry in data.items():
            if entry.get("infra_fingerprint") != self.fingerprint():
                continue
            if run_id and entry.get("run_id") != run_id:
                continue

            created_at = datetime.fromisoformat(entry["created_at"])
            expires_at = datetime.fromisoformat(entry["expires_at"]) if entry.get("expires_at") else None

            if entry.get("type") == "docker_service":
                resource = DockerServiceConfig(name=entry["resource_name"], scope="task")
                handles.append(
                    LocalDockerServiceHandle(
                        run_id=entry["run_id"],
                        resource=resource,
                        infra=self,
                        endpoint=entry.get("endpoint"),
                        endpoints=entry.get("endpoints", {}),
                        created_at=created_at,
                        expires_at=expires_at,
                        _entry_id=entry_id,
                        _container_ids=entry.get("container_ids", []),
                    )
                )
            else:
                # VM entry — check the QEMU process is still alive.
                pid = entry.get("pid")
                if pid and not _pid_alive(pid):
                    stale_ids.append(entry_id)
                    continue

                resource = VMResourceConfig(name=entry["resource_name"], scope="task")
                handles.append(
                    LocalResourceHandle(
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
                )

        # Clean up dead VM entries.
        if stale_ids:
            _ACTIVE_JSON.parent.mkdir(parents=True, exist_ok=True)
            lock_path = _ACTIVE_JSON.with_suffix(".lock")
            with open(lock_path, "w") as lock_fd:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                data = _load_active()
                for sid in stale_ids:
                    data.pop(sid, None)
                _save_active(data)

        return handles

    def cleanup(self, run_id: str) -> None:
        """Kill all QEMU processes and remove overlays for run_id."""
        _ACTIVE_JSON.parent.mkdir(parents=True, exist_ok=True)
        lock_path = _ACTIVE_JSON.with_suffix(".lock")
        with open(lock_path, "w") as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
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
        _ACTIVE_JSON.parent.mkdir(parents=True, exist_ok=True)
        lock_path = _ACTIVE_JSON.with_suffix(".lock")
        with open(lock_path, "w") as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
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
    """Stop containers / kill QEMU process and clean up for a single active.json entry."""
    # Docker service entries: stop + remove all tracked containers.
    container_ids = entry.get("container_ids", [])
    for cid in container_ids:
        for cmd in (["docker", "stop", cid], ["docker", "rm", cid]):
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0 and "No such container" not in result.stderr:
                logger.warning("%s failed: %s", " ".join(cmd), result.stderr.strip())
    if container_ids:
        logger.debug("Stopped %d container(s) from stale entry", len(container_ids))

    # VM entries: SIGTERM/SIGKILL the QEMU process.
    pid = entry.get("pid")
    if pid and _pid_alive(pid):
        try:
            os.kill(pid, 15)  # SIGTERM
            time.sleep(1)
            if _pid_alive(pid):
                os.kill(pid, 9)  # SIGKILL
        except Exception:
            pass
        logger.debug("Killed PID %d", pid)

    overlay_path = entry.get("overlay_path")
    if overlay_path:
        p = Path(overlay_path)
        if p.exists():
            p.unlink()
            logger.debug("Removed overlay %s", p)


# === auto-fix notes ===  (spec: openspec/specs/auto-fix/spec.md)
# auto-fix-note(174) {class=L1 issue=174 hash=PENDING ctx=colima/macos-arm64/n-a/cube-standard@0e91ae1}
#   symptoms:  tbench2 Auto-CUBE shakeout on a macOS/arm64 Colima box; malformed
#              DOCKER_HOST=http+unix:// (no socket path) -> broken unix:// client,
#              and from_env() missed Colima's non-default socket. Infra-specific:
#              docker backend + OS + DOCKER_HOST are the load-bearing context.
#   invariant: a Docker SDK client is built that reaches the daemon across
#              Podman/Colima/Docker-Desktop, or fails with an actionable error.
#   why:       single construction site; normalize http+unix only with a real
#              path -> docker context endpoint -> from_env -> actionable error.
#   tested:    tests/test_infra_local.py docker-client cases.
#   hash=PENDING: stamped by scripts/auto_fix_lint.py (Tier-1) on first run.
