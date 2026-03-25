"""
AzureVMBackend — experimental implementation of cube.vm.VMBackend for Azure.

Bridges:
  cube_azure_pipeline.py  (low-level Azure ops, validated 2026-03-25)
  cube.vm.VMBackend / VM  (official CUBE API)

ARCHITECTURE
------------
The VM image is expected to already expose an HTTP server on port 5000
(the OSWorld Flask server, or any compatible server).  AzureVMBackend's job
is purely infrastructure: get the image into Azure, provision a VM from it,
and return a vm.endpoint the harness can talk to.

No agent injection happens here — the image owns its server.

  AzureVMBackend.launch(config)
    → gallery image → Azure VM
    → SSH tunnel: localhost:{port} → vm:5000
    → AzureVM.endpoint = "http://localhost:{port}"
    → harness talks to endpoint directly

SAFETY
------
stop() / restore_snapshot() only delete resources created at launch time:
  cube-vm-{uid}, cube-nic-{uid}, cube-ip-{uid}
Gallery images, blobs, managed disks, and any pre-existing resources
are never touched by the VM lifecycle methods.

USAGE
-----
    from cube.vm import VMConfig
    from azure_vm_backend import AzureVMBackend

    config = VMConfig(snapshot_name="cube-ubuntu-22-04")
    backend = AzureVMBackend()
    vm = backend.launch(config)
    print(vm.endpoint)                 # http://localhost:15001
    vm.restore_snapshot("init_state")  # stop + relaunch (~3-4 min)
    vm.stop()
"""

from __future__ import annotations

import logging
import subprocess
import time

import cube_azure_pipeline as pipeline
from azure.identity import AzureCliCredential
from azure.mgmt.compute import ComputeManagementClient

from cube.vm import VM, VMBackend, VMConfig

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _download_hf(hf_uri: str) -> str:
    """Download a file from HuggingFace Hub.

    hf_uri: "hf://repo_id/filename"
    e.g.    "hf://xlangai/ubuntu_osworld/Ubuntu.qcow2"

    Returns local path to the downloaded file.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError("pip install huggingface_hub") from e

    assert hf_uri.startswith("hf://"), f"Expected hf:// URI, got: {hf_uri}"
    path_part = hf_uri[len("hf://"):]
    slash = path_part.index("/")
    repo_id = path_part[:slash]
    filename = path_part[slash + 1:]

    logger.info("[ensure_resource] Downloading %s from HuggingFace repo %s ...", filename, repo_id)
    local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    logger.info("[ensure_resource] Downloaded to %s", local_path)
    return local_path


def _resolve_image_source(hf_qcow2: str) -> str:
    """Return a local file path from an hf:// URI or a plain local path."""
    if hf_qcow2.startswith("hf://"):
        return _download_hf(hf_qcow2)
    from pathlib import Path
    p = Path(hf_qcow2)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    return str(p)


def _gallery_image_exists(subscription_id: str, resource_group: str, gallery_name: str, image_name: str) -> bool:
    """Return True if the gallery has at least one succeeded version of image_name."""
    compute = ComputeManagementClient(AzureCliCredential(), subscription_id)
    try:
        versions = list(
            compute.gallery_image_versions.list_by_gallery_image(resource_group, gallery_name, image_name)
        )
    except Exception:
        return False
    return any(v.provisioning_state not in ("Failed", "Deleting") for v in versions)


def _wait_for_endpoint(endpoint: str, timeout: int = 300) -> None:
    """Poll {endpoint}/screenshot until HTTP 200 with non-empty body.

    Uses /screenshot because it is universal across all compatible servers
    (OSWorld Flask server and the mini test agent both expose it).
    """
    import requests

    url = f"{endpoint}/screenshot"
    deadline = time.time() + timeout
    logger.info("[launch] Waiting for HTTP server at %s ...", url)
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200 and len(r.content) > 0:
                logger.info("[launch] HTTP server ready (%d bytes)", len(r.content))
                return
        except Exception:
            pass
        remaining = int(deadline - time.time())
        logger.debug("[launch] Not ready yet (%ds left)", remaining)
        time.sleep(10)
    raise TimeoutError(f"HTTP server not ready after {timeout}s at {endpoint}")


# ── AzureVM ──────────────────────────────────────────────────────────────────


class AzureVM(VM):
    """Runtime handle to a running Azure VM.

    Created by AzureVMBackend.launch() — do not instantiate directly.

    The HTTP endpoint is reached via SSH tunnel:
        http://localhost:{port}  →  SSH  →  azurevm:5000

    This bypasses Zscaler and other corporate proxies (SSH port 22 is always
    open) without exposing the VM's port 5000 to the internet.
    """

    def __init__(
        self,
        backend: AzureVMBackend,
        config: VMConfig,
        vm_name: str,
        pip_name: str,
        nic_name: str,
        public_ip: str,
        endpoint: str,
        tunnel: subprocess.Popen | None,
    ) -> None:
        self._backend = backend
        self._config = config
        self._vm_name = vm_name
        self._pip_name = pip_name
        self._nic_name = nic_name
        self._public_ip = public_ip
        self._endpoint = endpoint
        self._tunnel = tunnel

    @property
    def endpoint(self) -> str:
        return self._endpoint

    @property
    def public_ip(self) -> str:
        return self._public_ip

    def restore_snapshot(self, name: str) -> None:
        """Reset to clean state: delete this VM and launch a fresh one from gallery.

        The `name` argument is accepted for API compatibility.
        On cloud backends the entire VM IS the snapshot — restoring means
        deleting the current instance and launching a fresh one from the
        gallery image (which always represents the base state).

        Time: ~3-4 min on Azure.

        Safety: only the resources created at this VM's launch are deleted.
        The gallery image and all other pre-existing resources are untouched.
        """
        logger.info("[restore_snapshot] '%s' on %s", name, self._vm_name)
        self._terminate_tunnel()
        pipeline.stop(self._vm_name, pip_name=self._pip_name, nic_name=self._nic_name)

        new_vm = self._backend.launch(self._config)

        # Update state in-place so the caller's reference stays valid
        self._vm_name = new_vm._vm_name
        self._pip_name = new_vm._pip_name
        self._nic_name = new_vm._nic_name
        self._public_ip = new_vm._public_ip
        self._endpoint = new_vm._endpoint
        self._tunnel = new_vm._tunnel
        logger.info("[restore_snapshot] Done — new endpoint: %s", self._endpoint)

    def stop(self) -> None:
        """Shut down and delete this VM's resources. Idempotent.

        Deletes: VM, NIC, public IP (all named cube-vm/nic/ip-{uid}).
        Does NOT touch: gallery images, blobs, managed disks, or any
        resource that existed before this VM was launched.
        """
        logger.info("[stop] %s", self._vm_name)
        self._terminate_tunnel()
        pipeline.stop(self._vm_name, pip_name=self._pip_name, nic_name=self._nic_name)

    def _terminate_tunnel(self) -> None:
        if self._tunnel is not None and self._tunnel.poll() is None:
            self._tunnel.terminate()
            self._tunnel = None

    def __repr__(self) -> str:
        return f"<AzureVM {self._vm_name} @ {self._public_ip} endpoint={self._endpoint}>"


# ── AzureVMBackend ───────────────────────────────────────────────────────────


class AzureVMBackend(VMBackend):
    """VMBackend that provisions VMs from an Azure Compute Gallery image.

    ensure_resource() — one-time setup per subscription:
        download image (HuggingFace or local) → convert to fixed VHD
        → upload to Blob Storage → import as Managed Disk
        → publish to Compute Gallery

    launch() — per eval (~4 min):
        Gallery image → VM (createOption: FromImage, SSH key via os_profile)
        → SSH tunnel → http://localhost:{port}

    restore_snapshot() on the returned VM — between tasks (~3-4 min):
        stop() + launch() from gallery

    Attributes
    ----------
    subscription_id : str
    resource_group : str
    location : str
    gallery_name : str
    vm_size : str
    hf_qcow2 : str | None
        Image source: "hf://repo_id/filename" or "/local/path/image.qcow2".
        If None, image must already exist in the gallery.
    ssh_privkey : str
        Path to SSH private key for tunnel and os_profile injection.
    ssh_pubkey : str
        Path to SSH public key.
    agent_timeout : int
        Seconds to wait for the VM's HTTP server to become ready.
    """

    subscription_id: str = pipeline.SUBSCRIPTION
    resource_group: str = pipeline.RESOURCE_GROUP
    location: str = pipeline.LOCATION
    gallery_name: str = pipeline.GALLERY_NAME
    vm_size: str = pipeline.VM_SIZE

    hf_qcow2: str | None = None

    ssh_privkey: str = pipeline.SSH_PRIVKEY
    ssh_pubkey: str = pipeline.SSH_PUBKEY

    agent_timeout: int = 300

    def ensure_resource(self, config: VMConfig) -> None:
        """Idempotent: ensure gallery image `config.snapshot_name` exists.

        If the gallery image already exists, returns immediately (no-op).
        Otherwise runs the full pipeline: download → convert → upload → gallery.

        The gallery image is permanent and reused across all launches.
        It is never deleted by launch() / stop() / restore_snapshot().
        """
        image_name = config.snapshot_name

        if _gallery_image_exists(self.subscription_id, self.resource_group, self.gallery_name, image_name):
            logger.info("[ensure_resource] Gallery image '%s' already exists — skipping.", image_name)
            return

        if self.hf_qcow2 is None:
            raise ValueError(
                f"Gallery image '{image_name}' not found and hf_qcow2 is not set. "
                "Either provide hf_qcow2 to download the image, or run ensure_resource manually."
            )

        local_image = _resolve_image_source(self.hf_qcow2)
        vhd_path = pipeline.convert_to_vhd(local_image)
        pipeline.ensure_resource(vhd_path, image_name)
        logger.info("[ensure_resource] '%s' is now in gallery.", image_name)

    def launch(self, config: VMConfig) -> AzureVM:
        """Ensure gallery image exists, provision a VM, open SSH tunnel, wait for HTTP.

        Returns an AzureVM whose endpoint is ready for ComputerBase / GuestAgent.
        Blocks for ~4 min on first boot (2 min provisioning + 2 min server startup).
        """
        self.ensure_resource(config)

        result = pipeline.launch(config.snapshot_name, open_tunnel=True)

        vm = AzureVM(
            backend=self,
            config=config,
            vm_name=result["vm_name"],
            pip_name=result["pip_name"],
            nic_name=result["nic_name"],
            public_ip=result["public_ip"],
            endpoint=result["endpoint"],
            tunnel=result.get("tunnel"),
        )

        _wait_for_endpoint(vm.endpoint, timeout=self.agent_timeout)
        logger.info("[launch] Ready: %s", vm)
        return vm

    def close(self) -> None:
        pass


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """Full lifecycle test: ensure_resource (skip) → launch → probe → restore → probe → stop.

    Uses cube-ubuntu-22-04 which is already in the gallery (Generalized, v1.0.0).
    ensure_resource() detects it exists and skips the upload pipeline.

    Run:
        uv run --extra cube python experiments/azure-vm-backend/azure_vm_backend.py
    """
    import sys

    import requests

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = VMConfig(snapshot_name="cube-ubuntu-22-04")
    backend = AzureVMBackend()

    print("\n=== AzureVMBackend full lifecycle test ===")

    print("\n[1] ensure_resource()")
    backend.ensure_resource(config)

    print("\n[2] launch()")
    vm = backend.launch(config)
    print(f"    {vm}")

    print("\n[3] probe endpoints")
    for path, method, body in [
        ("/screenshot", "GET", None),
        ("/execute",    "POST", {"command": ["uname", "-a"]}),
    ]:
        r = requests.get(f"{vm.endpoint}{path}", timeout=10) if method == "GET" \
            else requests.post(f"{vm.endpoint}{path}", json=body, timeout=10)
        status = f"HTTP {r.status_code}  {len(r.content)} bytes"
        detail = r.json().get("stdout", "").strip() if r.headers.get("content-type", "").startswith("application/json") else ""
        print(f"    {method} {path} → {status}  {detail}")

    print("\n[4] restore_snapshot()")
    vm.restore_snapshot("init_state")
    print(f"    {vm}")

    print("\n[5] probe after restore")
    r = requests.get(f"{vm.endpoint}/screenshot", timeout=10)
    print(f"    GET /screenshot → HTTP {r.status_code}  {len(r.content)} bytes")

    print("\n[6] stop()")
    vm.stop()
    print("    Done — all runtime resources deleted.")
    print("    Gallery image, blobs, and pre-existing resources untouched.")
    sys.exit(0)
