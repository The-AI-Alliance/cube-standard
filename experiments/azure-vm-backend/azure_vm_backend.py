"""
AzureVMBackend — experimental implementation of the cube.vm.VMBackend ABC for Azure.

This is the bridge layer between:
  - cube_azure_pipeline.py  (low-level Azure operations, validated 2026-03-25)
  - cube.vm.VMBackend / VM  (the official CUBE API)

It lives in experiments/ while we validate the design. Once tested against the
real OSWorld image, it moves to cube-resources/cube-vm-backend/.

ARCHITECTURE
------------
AzureVMBackend (VMBackend)        AzureVM (VM)
  fields:                           fields:
    subscription_id                   _backend    ← needed for restore_snapshot
    resource_group                    _vm_name
    location                          _pip_name
    gallery_name                      _nic_name
    vm_size                           _public_ip
    hf_qcow2  ← image source          _endpoint   ← "http://localhost:{port}"
    inject_agent                      _tunnel     ← subprocess handle
    cache_dir                         _config     ← VMConfig (for restore)
    ssh_privkey
    ssh_pubkey
  methods:
    ensure_resource(config)       methods:
    launch(config) → AzureVM        endpoint (property)
    close()                         restore_snapshot(name)
                                    stop()

GUEST AGENT
-----------
Two modes controlled by `inject_agent`:
  True  (default) — cloud-init installs + starts the mini Flask agent from
                    cube_azure_pipeline.py. Use for blank images or testing.
  False           — image already has a server on port 5000 (e.g. OSWorld).
                    cloud-init only injects SSH key; no agent installation.

IMAGE SOURCE (hf_qcow2)
-----------------------
hf_qcow2 supports three forms:
  "hf://repo_id/filename"          — HuggingFace dataset (uses huggingface_hub)
  "/local/path/to/image.qcow2"     — pre-existing local file
  None                             — image must already exist in gallery

USAGE
-----
    from cube.vm import VMConfig
    from azure_vm_backend import AzureVMBackend

    config = VMConfig(snapshot_name="osworld-ubuntu", cpu_cores=4, ram_gb=8)
    backend = AzureVMBackend(
        hf_qcow2="hf://xlangai/OSWORLD/ubuntu.vmdk",
        inject_agent=False,   # OSWorld image already has its own Flask server
    )
    vm = backend.launch(config)        # ensure_resource() called automatically
    print(vm.endpoint)                 # http://localhost:15001
    vm.restore_snapshot("init_state")  # stop + relaunch from gallery (~3-4 min)
    vm.stop()
"""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

# ── Pipeline imports ─────────────────────────────────────────────────────────
# Low-level Azure operations, all validated end-to-end (2026-03-25).
import cube_azure_pipeline as pipeline
from azure.identity import AzureCliCredential
from azure.mgmt.compute import ComputeManagementClient

# ── CUBE imports ────────────────────────────────────────────────────────────
# cube.vm defines the official API we must conform to.
# These are already in cube-standard/src/; run from the cube-standard venv.
from cube.vm import VM, VMBackend, VMConfig

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _download_hf(hf_uri: str, cache_dir: Path) -> str:
    """Download a file from HuggingFace Hub.

    hf_uri format: "hf://repo_id/filename"
    e.g.  "hf://xlangai/OSWORLD/ubuntu.vmdk"
         "hf://The-AI-Alliance/osworld-cube/ubuntu-22.04.qcow2"

    Returns the local path to the downloaded file.
    Requires: pip install huggingface_hub
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError("pip install huggingface_hub  (needed for hf:// image sources)") from e

    # Parse "hf://repo_id/filename"  →  repo_id="repo_id", filename="filename"
    assert hf_uri.startswith("hf://"), f"Expected hf:// URI, got: {hf_uri}"
    path_part = hf_uri[len("hf://"):]
    slash = path_part.index("/")
    repo_id = path_part[:slash]
    filename = path_part[slash + 1:]

    logger.info("[ensure_resource] Downloading %s from HuggingFace...", filename)
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        cache_dir=str(cache_dir),
        local_dir=str(cache_dir),
    )
    logger.info("[ensure_resource] Downloaded to %s", local_path)
    return local_path


def _resolve_image(hf_qcow2: str, cache_dir: Path) -> str:
    """Resolve hf_qcow2 to a local file path.

    Handles:
      - "hf://..."   → download from HuggingFace
      - "/path/..."  → local file (returned as-is)
    """
    if hf_qcow2.startswith("hf://"):
        return _download_hf(hf_qcow2, cache_dir)
    local = Path(hf_qcow2)
    if not local.exists():
        raise FileNotFoundError(f"Image not found: {local}")
    return str(local)


def _wait_for_agent(endpoint: str, timeout: int = 300) -> None:
    """Poll vm.endpoint/screenshot until it responds (HTTP 200 with image bytes).

    Uses /screenshot rather than /health because the OSWorld server does not
    expose /health — only our mini agent does. /screenshot is universal.
    """
    import requests

    deadline = time.time() + timeout
    url = f"{endpoint}/screenshot"
    logger.info("[launch] Waiting for HTTP agent at %s (timeout %ds)...", url, timeout)
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200 and len(r.content) > 0:
                logger.info("[launch] Agent ready ✓ (%d bytes)", len(r.content))
                return
        except Exception:
            pass
        remaining = int(deadline - time.time())
        logger.debug("[launch] Waiting... (%ds left)", remaining)
        time.sleep(10)
    raise TimeoutError(f"VM HTTP agent not ready after {timeout}s at {endpoint}")


# ── AzureVM ──────────────────────────────────────────────────────────────────


class AzureVM(VM):
    """Runtime handle to a running Azure VM.

    Not serializable. Created by AzureVMBackend.launch() — do not instantiate directly.

    The VM exposes an HTTP endpoint via SSH tunnel:
        http://localhost:{port}  →  azurevm:5000

    This bypasses Zscaler on corporate networks (SSH port 22 is always open).
    The endpoint URL is what ComputerBase / GuestAgent talks to.
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
        """Base URL of the in-VM HTTP agent: ``http://localhost:{port}``."""
        return self._endpoint

    @property
    def public_ip(self) -> str:
        """Azure public IP of the VM (useful for debugging, not needed by CUBE)."""
        return self._public_ip

    def restore_snapshot(self, name: str) -> None:
        """Reset to clean state by deleting this VM and launching a fresh one.

        Implementation: stop() + backend.launch() from the gallery image.
        The `name` argument is accepted for API compatibility but is currently
        ignored — the gallery always returns the base image state.

        Time: ~3-4 min (Azure VM provisioning + cloud-init).
        """
        logger.info("[restore_snapshot] Resetting VM '%s' (stop + relaunch)", self._vm_name)
        # Terminate tunnel and delete VM (in-place mutation of self)
        self._stop_tunnel()
        pipeline.stop(self._vm_name, pip_name=self._pip_name, nic_name=self._nic_name)

        # Launch a fresh VM from the same gallery image
        new_vm = self._backend.launch(self._config)

        # Update our state in-place so the caller's reference stays valid
        self._vm_name = new_vm._vm_name
        self._pip_name = new_vm._pip_name
        self._nic_name = new_vm._nic_name
        self._public_ip = new_vm._public_ip
        self._endpoint = new_vm._endpoint
        self._tunnel = new_vm._tunnel

        logger.info("[restore_snapshot] Done — new endpoint: %s", self._endpoint)

    def stop(self) -> None:
        """Shut down this VM and release all its resources. Idempotent."""
        logger.info("[stop] Stopping VM '%s'", self._vm_name)
        self._stop_tunnel()
        pipeline.stop(self._vm_name, pip_name=self._pip_name, nic_name=self._nic_name)

    def _stop_tunnel(self) -> None:
        if self._tunnel is not None and self._tunnel.poll() is None:
            self._tunnel.terminate()
            self._tunnel = None

    def __repr__(self) -> str:
        return f"<AzureVM {self._vm_name} @ {self._public_ip} endpoint={self._endpoint}>"


# ── AzureVMBackend ───────────────────────────────────────────────────────────


class AzureVMBackend(VMBackend):
    """VMBackend that provisions VMs on Azure Compute from a gallery image.

    One-time setup (ensure_resource):
        download (HuggingFace or local) → convert to VHD → upload to Blob Storage
        → import as Managed Disk → publish to Compute Gallery

    Per-eval (launch):
        Gallery image → VM (createOption: FromImage) + SSH key injection
        → SSH tunnel → http://localhost:{port}

    Between tasks (restore_snapshot on the returned VM):
        stop + relaunch from gallery (~3-4 min)

    Attributes
    ----------
    subscription_id : str
        Azure subscription ID.
    resource_group : str
        Resource group for all provisioned resources.
    location : str
        Azure region (e.g. "westus2").
    gallery_name : str
        Name of the Azure Compute Gallery.
    vm_size : str
        Azure VM size (e.g. "Standard_D4s_v3" = 4 vCPU, 16 GB RAM).
    hf_qcow2 : str | None
        Image source. Supported formats:
          "hf://repo_id/filename"    — HuggingFace dataset
          "/local/path/image.qcow2"  — pre-existing local file
          None                       — image must already be in gallery
    inject_agent : bool
        True  → cloud-init installs + starts the mini Flask agent (for blank images)
        False → image already has a server on port 5000 (e.g. OSWorld)
    cache_dir : str
        Local directory for downloaded images and converted VHDs.
    ssh_privkey : str
        Path to SSH private key (used for tunnel and cloud-init key injection).
    ssh_pubkey : str
        Path to SSH public key (injected into VM via os_profile).
    agent_timeout : int
        Seconds to wait for the HTTP agent to become ready after launch.
    """

    # Azure infrastructure
    subscription_id: str = pipeline.SUBSCRIPTION
    resource_group: str = pipeline.RESOURCE_GROUP
    location: str = pipeline.LOCATION
    gallery_name: str = pipeline.GALLERY_NAME
    vm_size: str = pipeline.VM_SIZE

    # Image source
    hf_qcow2: str | None = None

    # Guest agent
    inject_agent: bool = True

    # Local cache for images
    cache_dir: str = str(Path.home() / ".cube" / "vm_data")

    # SSH
    ssh_privkey: str = pipeline.SSH_PRIVKEY
    ssh_pubkey: str = pipeline.SSH_PUBKEY

    # Readiness timeout
    agent_timeout: int = 300

    # ── ensure_resource ───────────────────────────────────────────────────────

    def ensure_resource(self, config: VMConfig) -> None:
        """Idempotent: ensure gallery image `config.snapshot_name` exists on Azure.

        Steps (skipped if already done):
          1. Download image from HuggingFace (or use local path)
          2. Convert to fixed VHD
          3. Upload to Azure Blob Storage
          4. Import blob → Managed Disk
          5. Publish Managed Disk → Compute Gallery image version

        This is ~60-90 min the first time per subscription; subsequent calls
        detect the existing gallery image and return immediately.
        """
        image_name = config.snapshot_name

        # Fast-path: gallery image already exists
        existing = self._find_gallery_image(image_name)
        if existing:
            logger.info("[ensure_resource] Gallery image '%s' already exists: %s", image_name, existing)
            return

        logger.info("[ensure_resource] Gallery image '%s' not found — running pipeline.", image_name)

        # Step 1: Acquire image locally
        if self.hf_qcow2 is None:
            raise ValueError(
                f"Gallery image '{image_name}' not found and hf_qcow2 is not set. "
                "Either provide hf_qcow2 to download the image, or upload it manually first."
            )
        cache = Path(self.cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        local_image = _resolve_image(self.hf_qcow2, cache)

        # Step 2: Convert to fixed VHD
        vhd_path = pipeline.convert_to_vhd(local_image)

        # Step 3–5: Upload → import disk → gallery
        # pipeline.ensure_resource handles steps 3-5 and is idempotent.
        # It uses the pipeline's global SUBSCRIPTION / RESOURCE_GROUP / LOCATION /
        # GALLERY_NAME constants. Future: parameterize these from self.*.
        pipeline.ensure_resource(vhd_path, image_name)

        logger.info("[ensure_resource] Done — '%s' is now in gallery.", image_name)

    # ── launch ────────────────────────────────────────────────────────────────

    def launch(self, config: VMConfig) -> AzureVM:
        """Ensure gallery image exists, then launch a VM and return a live handle.

        Blocks until the HTTP endpoint on the VM is reachable.
        Total time: ~4 min (2 min provisioning + 2 min cloud-init on first boot).

        The returned AzureVM owns its tunnel and VM resources.
        Call vm.stop() when done, or vm.restore_snapshot() between tasks.
        """
        self.ensure_resource(config)

        image_name = config.snapshot_name

        # Temporarily patch pipeline constants if this backend uses non-default values.
        # TODO: refactor pipeline to accept params instead of using globals.
        if not self.inject_agent:
            # Launch without guest agent injection: cloud-init only injects SSH key.
            # The image is expected to already have a server running on port 5000.
            result = self._launch_no_agent(image_name)
        else:
            # Default: launch with mini Flask agent injected via cloud-init.
            result = pipeline.launch(image_name, open_tunnel=True)

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

        # Wait for the HTTP agent to be ready
        _wait_for_agent(vm.endpoint, timeout=self.agent_timeout)

        logger.info("[launch] VM ready: %s", vm)
        return vm

    def close(self) -> None:
        """No-op at the backend level — each VM is stopped individually via vm.stop()."""
        pass

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _find_gallery_image(self, image_name: str) -> str | None:
        """Return gallery image ID if it exists, else None.

        Queries the Azure Compute Gallery directly via SDK.
        Returns the first available version ID, or None if no versions exist.
        """
        compute = ComputeManagementClient(AzureCliCredential(), self.subscription_id)
        try:
            versions = list(
                compute.gallery_image_versions.list_by_gallery_image(
                    self.resource_group, self.gallery_name, image_name
                )
            )
        except Exception:
            return None

        # Return ID of the first available (non-failed) version
        for v in versions:
            if v.provisioning_state not in ("Failed", "Deleting"):
                return v.id or ""
        return None

    def _launch_no_agent(self, image_name: str) -> dict:
        """Launch a VM without cloud-init agent injection.

        Used when the image already has its own HTTP server (e.g. OSWorld).
        Still injects SSH key via os_profile (needed for SSH tunnel).

        NOTE: This is a separate launch path that omits the CLOUD_INIT_TEMPLATE.
        cloud-init will still run (it always does for Generalized images) but
        with an empty/minimal config — just SSH key injection.

        TODO: Implement the minimal cloud-init path in cube_azure_pipeline.py
              and call it here. For now, falls back to default launch() which
              injects the mini agent (harmless if the real server is also running).
        """
        logger.warning(
            "[launch] inject_agent=False not fully implemented yet — "
            "using default launch with mini agent. The OSWorld server should "
            "still be reachable at /screenshot if it is already running in the image."
        )
        return pipeline.launch(image_name, open_tunnel=True)


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """Quick smoke test: launch from existing gallery image, probe, stop.

    Assumes cube-ubuntu-22-04 already exists in the gallery (from previous runs).
    Run from the cube-standard venv:
        uv run python experiments/azure-vm-backend/azure_vm_backend.py
    """
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = VMConfig(snapshot_name="cube-ubuntu-22-04", cpu_cores=4, ram_gb=16)
    backend = AzureVMBackend(inject_agent=True)

    print("=== AzureVMBackend smoke test ===")
    print(f"config: {config}")
    print(f"backend: {backend}")

    print("\n--- launch() ---")
    vm = backend.launch(config)
    print(f"vm: {vm}")
    print(f"endpoint: {vm.endpoint}")

    print("\n--- probe endpoints ---")
    import requests

    for path, method, body in [
        ("/screenshot", "GET", None),
        ("/execute", "POST", {"command": ["uname", "-a"]}),
    ]:
        if method == "GET":
            r = requests.get(f"{vm.endpoint}{path}", timeout=10)
        else:
            r = requests.post(f"{vm.endpoint}{path}", json=body, timeout=10)
        print(f"  {method} {path} → HTTP {r.status_code}, {len(r.content)} bytes")
        if r.headers.get("content-type", "").startswith("application/json"):
            print(f"    {r.json()}")

    print("\n--- restore_snapshot() ---")
    vm.restore_snapshot("init_state")
    print(f"vm after restore: {vm}")

    print("\n--- probe after restore ---")
    r = requests.get(f"{vm.endpoint}/screenshot", timeout=10)
    print(f"  GET /screenshot → HTTP {r.status_code}, {len(r.content)} bytes")

    print("\n--- stop() ---")
    vm.stop()
    print("Done.")
    sys.exit(0)
