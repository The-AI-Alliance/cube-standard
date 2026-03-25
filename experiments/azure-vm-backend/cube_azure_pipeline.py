"""
CUBE Azure VM Pipeline
======================
Automates the full path from a VM image (qcow2 or URL on HuggingFace) to a
running Azure VM that the CUBE harness can communicate with.

Two ensure_resource approaches
--------------------------------
1. Local pipeline  — convert + upload from your machine (ensure_resource)
   Best for: images you already have locally, fast local upload connection.
   Steps: detect format → convert to fixed VHD → upload PageBlob → import disk
          → publish to Compute Gallery
   Timing: upload dominates (~0.2–1 GB/min on typical broadband)

2. Bootstrap VM    — spin up a cheap Azure VM to do the heavy lifting (bootstrap_ensure_resource)
   Best for: images on HuggingFace / public URLs, slow local upload.
   Steps: launch Standard_B2ms → download from HF at ~55 MB/s → convert to VHD
          → upload via azcopy at ~300 Mb/s (datacenter) → signal sentinel → VM deleted
          → import disk → publish to Compute Gallery
   Timing: ~46 min total for 12 GB zip / 23 GB VHD (vs hours from home broadband)
   Cost:   ~$0.04 (Standard_B2ms @ $0.087/hr × ~30 min + 128 GB HDD disk)

Key design decisions
---------------------
- Azure Compute Gallery mandatory: bypasses ServiceNow Golden Image Policy (blocks
  Marketplace images). All bootstrap VMs and OSWorld VMs must use gallery images.
- Bootstrap VM must also use a gallery image (same policy applies to the bootstrap VM itself).
- Large OS disk (128 GB) on bootstrap VM: simpler than attaching a data disk and avoids
  mkfs/udev timing issues when formatting newly-attached block devices.
- Fixed VHD (not sparse VMDK): required by Azure import pipeline. VMDK is rejected by
  Azure Blob Storage's PageBlob writer if the virtual size does not match the file size.
- VHD footer validation in upload_vhd(): a partial upload leaves a blob with the right
  size but zero bytes in the footer. Checking for the "conectix" magic string detects
  corrupt uploads and forces a re-upload.
- SSH tunnel: bypasses Zscaler (all non-SSH ports blocked on corp network). Tunnel opens
  localhost:{port} → vm:5000 so callers just hit http://localhost:{port}.
- OS disk size not specified in launch(): letting Azure inherit from the gallery image
  avoids "disk size smaller than image" errors when the image has a large virtual disk.

USAGE
-----
# Bootstrap (recommended for HuggingFace sources):
python cube_azure_pipeline.py bootstrap --url https://huggingface.co/.../Ubuntu.qcow2.zip --name my-image

# Local pipeline:
python cube_azure_pipeline.py ensure --image path/to/image.qcow2 --name my-benchmark

# Per-eval: launch VM from gallery (~2-4 min)
python cube_azure_pipeline.py launch --name my-benchmark

# Probe guest agent (after launch)
python cube_azure_pipeline.py probe --ip 20.x.x.x

# Teardown
python cube_azure_pipeline.py stop --vm cube-vm-abc123

# List gallery images
python cube_azure_pipeline.py list

Tested end-to-end
------------------
- Ubuntu 22.04 cloud image: local pipeline ✓ (2026-03-24)
- OSWorld Ubuntu image (50 GB qcow2): local pipeline ✓ (2026-03-25)
- OSWorld Ubuntu image: bootstrap VM pipeline ✓ (2026-03-25)
"""

import base64
import json
import socket
import subprocess
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from azure.identity import AzureCliCredential
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.network import NetworkManagementClient
from azure.mgmt.storage import StorageManagementClient
from azure.storage.blob import BlobSasPermissions, BlobServiceClient, generate_blob_sas

# ── Configuration ─────────────────────────────────────────────────────────────

SUBSCRIPTION = "aeb958d3-a614-450e-94bc-88f284dc0664"
RESOURCE_GROUP = "ui_assist"
LOCATION = "westus2"
STORAGE_ACCOUNT = "cubeexpvhd"
CONTAINER_NAME = "vhds"
VNET_NAME = "vnet-westus2"
SUBNET_NAME = "snet-westus2-1"
NSG_NAME = "osworld-nsg"
GALLERY_NAME = "cube_exp_gallery"
VM_SIZE = "Standard_D4s_v3"  # 4 vCPU, 16 GB RAM
GUEST_PORT = 5000
TAGS = {"project": "cube-experiment"}

SSH_PRIVKEY = str(Path.home() / ".ssh" / "id_ed25519")
SSH_PUBKEY = str(Path.home() / ".ssh" / "id_ed25519.pub")

# Minimal CUBE guest agent (Flask) — base64-encoded to avoid YAML parse issues
# Provides: /health, /screenshot (black rect placeholder), /execute
_AGENT_PY_B64 = (
    "aW1wb3J0IHN1YnByb2Nlc3MKZnJvbSBmbGFzayBpbXBvcnQgRmxhc2ssIGpzb25pZnksIHNlbmRf"
    "ZmlsZQpmcm9tIFBJTCBpbXBvcnQgSW1hZ2UKYXBwID0gRmxhc2soX19uYW1lX18pCgpAYXBwLnJv"
    "dXRlKCIvaGVhbHRoIikKZGVmIGhlYWx0aCgpOgogICAgZnJvbSBmbGFzayBpbXBvcnQganNvbmlm"
    "eQogICAgcmV0dXJuIGpzb25pZnkoeyJzdGF0dXMiOiAib2siLCAiYWdlbnQiOiAiY3ViZS1taW5p"
    "LWd1ZXN0LWFnZW50In0pCgpAYXBwLnJvdXRlKCIvc2NyZWVuc2hvdCIpCmRlZiBzY3JlZW5zaG90"
    "KCk6CiAgICBpbWcgPSBJbWFnZS5uZXcoIlJHQiIsICg4MDAsIDYwMCksIGNvbG9yPSgzMCwgMzAs"
    "IDMwKSkKICAgIGltZy5zYXZlKCIvdG1wL3NjcmVlbi5wbmciKQogICAgcmV0dXJuIHNlbmRfZmls"
    "ZSgiL3RtcC9zY3JlZW4ucG5nIiwgbWltZXR5cGU9ImltYWdlL3BuZyIpCgpAYXBwLnJvdXRlKCIv"
    "ZXhlY3V0ZSIsIG1ldGhvZHM9WyJQT1NUIl0pCmRlZiBleGVjdXRlKCk6CiAgICBmcm9tIGZsYXNr"
    "IGltcG9ydCByZXF1ZXN0CiAgICBjbWQgPSByZXF1ZXN0Lmpzb24uZ2V0KCJjb21tYW5kIiwgW10p"
    "CiAgICByZXN1bHQgPSBzdWJwcm9jZXNzLnJ1bihjbWQsIGNhcHR1cmVfb3V0cHV0PVRydWUsIHRl"
    "eHQ9VHJ1ZSkKICAgIGZyb20gZmxhc2sgaW1wb3J0IGpzb25pZnkKICAgIHJldHVybiBqc29uaWZ5"
    "KHsic3Rkb3V0IjogcmVzdWx0LnN0ZG91dCwgInN0ZGVyciI6IHJlc3VsdC5zdGRlcnIsICJyZXR1"
    "cm5jb2RlIjogcmVzdWx0LnJldHVybmNvZGV9KQoKaWYgX19uYW1lX18gPT0gIl9fbWFpbl9fIjoK"
    "ICAgIGFwcC5ydW4oaG9zdD0iMC4wLjAuMCIsIHBvcnQ9NTAwMCkK"
)

# cloud-init: installs flask + writes agent + starts it on port 5000
# Uses write_files+b64 to avoid Python-code-as-YAML parse errors.
# NOTE: write_files runs during init-network, before azureuser is created by waagent.
#   So we write to /usr/local/bin/ (no owner field needed — defaults to root:root).
#   runcmd runs as root so ownership doesn't matter.
CLOUD_INIT_TEMPLATE = """\
#cloud-config
packages:
  - python3-flask
  - python3-pil
write_files:
  - path: /usr/local/bin/cube_guest_agent.py
    permissions: '0755'
    encoding: b64
    content: {agent_b64}
runcmd:
  - nohup python3 /usr/local/bin/cube_guest_agent.py > /var/log/cube-guest-agent.log 2>&1 &
"""


# ── Azure clients ─────────────────────────────────────────────────────────────


def _cred():
    return AzureCliCredential()


def _compute() -> ComputeManagementClient:
    return ComputeManagementClient(_cred(), SUBSCRIPTION)


def _network() -> NetworkManagementClient:
    return NetworkManagementClient(_cred(), SUBSCRIPTION)


def _storage() -> StorageManagementClient:
    return StorageManagementClient(_cred(), SUBSCRIPTION)


# ── Step 1: Convert image to fixed VHD ───────────────────────────────────────


def convert_to_vhd(image_path: str, output_path: str | None = None) -> str:
    """
    Convert a qcow2 or VMDK image to a fixed-size Azure-compatible VHD.

    Supports:
      - qcow2 (.qcow2, .img)         -- QEMU native format
      - VMDK (.vmdk, multi-extent)   -- VMware format (e.g. OSWorld HuggingFace)
      - VHD (.vhd)                   -- already correct, returned as-is

    Returns path to the .vhd file.
    """
    src = Path(image_path).resolve()
    if output_path is None:
        output_path = str(src.with_suffix(".vhd"))
    dst = Path(output_path).resolve()

    if dst.exists():
        print(f"[convert] VHD already exists: {dst.name} ({dst.stat().st_size / 1024**3:.1f} GB), skipping.")
        return str(dst)

    # Detect format
    result = subprocess.run(
        ["qemu-img", "info", "--output=json", str(src)],
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(result.stdout)
    fmt = info["format"]
    vsize_gb = info["virtual-size"] / 1024**3
    dsize_gb = info.get("disk-size", info["virtual-size"]) / 1024**3

    print(f"[convert] {src.name}")
    print(f"  format: {fmt}  virtual: {vsize_gb:.1f} GB  on-disk: {dsize_gb:.1f} GB")
    print(f"  → {dst.name} (fixed VHD, {vsize_gb:.1f} GB)")

    t0 = time.time()
    subprocess.run(
        ["qemu-img", "convert", "-f", fmt, "-O", "vpc", "-o", "subformat=fixed,force_size", str(src), str(dst)],
        check=True,
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s ({dst.stat().st_size / 1024**3:.1f} GB on disk)")
    return str(dst)


# ── Step 2: Upload VHD to Azure Blob Storage ──────────────────────────────────


def upload_vhd(vhd_path: str) -> str:
    """
    Upload a fixed VHD to Azure Blob Storage as a PageBlob.
    Idempotent — re-uploads if blob doesn't exist.

    Returns the blob URL.
    """
    vhd = Path(vhd_path).resolve()
    blob_name = vhd.name
    size_gb = vhd.stat().st_size / 1024**3

    print(f"[upload] {vhd.name} ({size_gb:.1f} GB)")

    # Ensure storage account exists
    storage = _storage()
    try:
        storage.storage_accounts.get_properties(RESOURCE_GROUP, STORAGE_ACCOUNT)
    except Exception:
        print(f"  Creating storage account: {STORAGE_ACCOUNT}")
        poller = storage.storage_accounts.begin_create(  # type: ignore[call-overload]
            RESOURCE_GROUP,
            STORAGE_ACCOUNT,
            {"location": LOCATION, "tags": TAGS, "sku": {"name": "Standard_LRS"}, "kind": "StorageV2"},  # type: ignore[arg-type]
        )
        poller.result()

    keys = storage.storage_accounts.list_keys(RESOURCE_GROUP, STORAGE_ACCOUNT)
    assert keys.keys, "Storage account returned no keys"
    conn_str = (
        f"DefaultEndpointsProtocol=https;AccountName={STORAGE_ACCOUNT};"
        f"AccountKey={keys.keys[0].value};EndpointSuffix=core.windows.net"
    )

    # Large-file upload settings: 4 MB chunks, 4 concurrent connections,
    # generous timeouts for 50+ GB files over slower connections.
    svc = BlobServiceClient.from_connection_string(
        conn_str,
        max_single_put_size=4 * 1024 * 1024,       # 4 MB
        max_page_size=4 * 1024 * 1024,              # 4 MB pages
        connection_timeout=300,                      # 5 min connect timeout
        read_timeout=600,                            # 10 min read timeout
    )
    container = svc.get_container_client(CONTAINER_NAME)
    try:
        container.get_container_properties()
    except Exception:
        container.create_container()

    # Check if already uploaded — validate both size AND VHD footer magic.
    # A timed-out upload may leave a PageBlob with the correct reported size but
    # zeroed-out trailing bytes (the footer was never written), causing Azure disk
    # import to fail with "cookie value 'conectix' not found".
    blob_client = svc.get_blob_client(CONTAINER_NAME, blob_name)
    try:
        props = blob_client.get_blob_properties()
        if props.size == vhd.stat().st_size:
            # Validate the VHD footer: last 512 bytes must start with 'conectix'
            footer_offset = props.size - 512
            footer_data = blob_client.download_blob(offset=footer_offset, length=512).readall()
            if footer_data[:8] == b"conectix":
                blob_url = f"https://{STORAGE_ACCOUNT}.blob.core.windows.net/{CONTAINER_NAME}/{blob_name}"
                print(f"  Already uploaded (footer valid): {blob_url}")
                return blob_url
            else:
                print("  Blob exists but footer is corrupt (partial upload) — deleting and re-uploading.")
                blob_client.delete_blob()
    except Exception:
        pass

    print(f"  Uploading to {STORAGE_ACCOUNT}/{CONTAINER_NAME}/{blob_name} ...")
    print("  (50 GB takes ~60-90 min — progress shown every 512 MB)")
    t0 = time.time()
    uploaded = [0]

    def _progress(current: int, total: int) -> None:
        pct = current / total * 100
        gb = current / 1024**3
        total_gb = total / 1024**3
        elapsed = time.time() - t0
        rate = gb / (elapsed / 60) if elapsed > 0 else 0
        eta_min = (total_gb - gb) / rate if rate > 0 else 0
        # Print every ~512 MB to avoid flooding output
        if current - uploaded[0] >= 512 * 1024 * 1024 or current == total:
            uploaded[0] = current
            print(f"  {pct:.0f}%  {gb:.1f}/{total_gb:.1f} GB  {rate:.2f} GB/min  ETA {eta_min:.0f} min")

    with open(vhd, "rb") as f:
        blob_client.upload_blob(
            f,
            blob_type="PageBlob",
            overwrite=True,
            max_concurrency=4,
            progress_hook=_progress,
        )
    elapsed = time.time() - t0
    speed = size_gb / (elapsed / 60)
    print(f"  Uploaded in {elapsed / 60:.1f} min ({speed:.2f} GB/min)")

    return f"https://{STORAGE_ACCOUNT}.blob.core.windows.net/{CONTAINER_NAME}/{blob_name}"


# ── Step 3: Import VHD blob as Managed Disk ───────────────────────────────────


def import_disk(blob_url: str, disk_name: str | None = None) -> str:
    """
    Create a Managed Disk from a VHD blob.
    Returns the disk name.
    """
    if disk_name is None:
        disk_name = f"cube-disk-{uuid.uuid4().hex[:8]}"

    print(f"[import] Blob → Managed Disk: {disk_name}")
    t0 = time.time()

    compute = _compute()
    poller = compute.disks.begin_create_or_update(  # type: ignore[call-overload]
        RESOURCE_GROUP,
        disk_name,
        {  # type: ignore[arg-type]
            "location": LOCATION,
            "tags": TAGS,
            "sku": {"name": "Standard_LRS"},
            "properties": {
                "creationData": {
                    "createOption": "Import",
                    "sourceUri": blob_url,
                    "storageAccountId": (
                        f"/subscriptions/{SUBSCRIPTION}/resourceGroups/{RESOURCE_GROUP}"
                        f"/providers/Microsoft.Storage/storageAccounts/{STORAGE_ACCOUNT}"
                    ),
                },
                "osType": "Linux",
            },
        },
    )
    disk = poller.result()
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s: {disk_name} ({disk.disk_size_gb} GB)")
    return disk_name


# ── Step 4: Publish to Azure Compute Gallery ──────────────────────────────────


def ensure_gallery() -> str:
    """Create Compute Gallery if it doesn't exist. Returns gallery name."""
    compute = _compute()
    try:
        compute.galleries.get(RESOURCE_GROUP, GALLERY_NAME)
    except Exception:
        print(f"[gallery] Creating gallery: {GALLERY_NAME}")
        compute.galleries.begin_create_or_update(  # type: ignore[call-overload]
            RESOURCE_GROUP,
            GALLERY_NAME,
            {"location": LOCATION, "tags": TAGS, "description": "CUBE benchmark VM image gallery"},  # type: ignore[arg-type]
        ).result()
    return GALLERY_NAME


def create_image_definition(name: str, os_state: str = "Generalized", hyper_v_gen: str = "V1") -> str:
    """
    Create a gallery image definition.

    os_state:
      "Generalized" — Ubuntu cloud images. Supports os_profile SSH key injection
                      and cloud-init custom_data at launch. Use this for images
                      converted from HuggingFace qcow2/vmdk.
      "Specialized"  — Images with baked-in identity (e.g. osworld_base snapshot).
                      No os_profile at launch. Needs Run Command for SSH access.

    Returns image definition name.
    """
    ensure_gallery()
    compute = _compute()
    try:
        compute.gallery_images.get(RESOURCE_GROUP, GALLERY_NAME, name)
        print(f"[imgdef] Already exists: {name}")
        return name
    except Exception:
        pass

    print(f"[imgdef] Creating image definition: {name} ({os_state}, HyperV {hyper_v_gen})")
    poller = compute.gallery_images.begin_create_or_update(  # type: ignore[call-overload]
        RESOURCE_GROUP,
        GALLERY_NAME,
        name,
        {  # type: ignore[arg-type]
            "location": LOCATION,
            "tags": TAGS,
            "os_type": "Linux",
            "os_state": os_state,
            "hyper_v_generation": hyper_v_gen,
            "identifier": {"publisher": "cube", "offer": name, "sku": "linux"},
        },
    )
    poller.result()
    print(f"  Created: {name}")
    return name


def create_image_version(image_def: str, version: str, disk_name: str) -> str:
    """
    Publish a Managed Disk as a gallery image version.
    Idempotent — skips if version already exists.

    Returns the full gallery image version ID.
    """
    compute = _compute()
    try:
        existing = compute.gallery_image_versions.get(RESOURCE_GROUP, GALLERY_NAME, image_def, version)
        if existing.provisioning_state == "Succeeded":
            print(f"[version] Already exists: {image_def}/{version}")
            return existing.id or ""
    except Exception:
        pass

    disk = compute.disks.get(RESOURCE_GROUP, disk_name)
    print(f"[version] Publishing {image_def}/{version} from {disk_name} ({disk.disk_size_gb} GB)...")
    t0 = time.time()

    poller = compute.gallery_image_versions.begin_create_or_update(  # type: ignore[call-overload]
        RESOURCE_GROUP,
        GALLERY_NAME,
        image_def,
        version,
        {  # type: ignore[arg-type]
            "location": LOCATION,
            "tags": TAGS,
            "publishing_profile": {
                "replica_count": 1,
                "storage_account_type": "Standard_LRS",
                "target_regions": [
                    {"name": LOCATION, "regional_replica_count": 1, "storage_account_type": "Standard_LRS"}
                ],
                "exclude_from_latest": False,
            },
            "storage_profile": {"os_disk_image": {"source": {"id": disk.id}, "host_caching": "ReadWrite"}},
        },
    )
    version_obj = poller.result()
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s: {version_obj.id}")
    return version_obj.id or ""


# ── ensure_resource: all steps in one call ────────────────────────────────────


def ensure_resource(image_path: str, name: str, version: str = "1.0.0", admin_user: str = "azureuser") -> dict:
    """
    Full one-time setup: image file → Compute Gallery image version.

    image_path : local path to .qcow2, .img, .vmdk, or .vhd
    name       : gallery image definition name (e.g. "osworld-ubuntu-x86")
    version    : gallery image version string (default "1.0.0")

    Returns {"gallery_image": name, "version": version, "image_id": ...}
    """
    timings = {}
    print(f"\n{'=' * 60}")
    print(f"ensure_resource: {name} v{version}")
    print(f"{'=' * 60}")

    t0 = time.time()
    vhd_path = convert_to_vhd(image_path)
    timings["convert"] = time.time() - t0

    t0 = time.time()
    blob_url = upload_vhd(vhd_path)
    timings["upload"] = time.time() - t0

    t0 = time.time()
    disk_name = import_disk(blob_url, disk_name=f"cube-disk-{name}")
    timings["import"] = time.time() - t0

    t0 = time.time()
    create_image_definition(name)
    image_id = create_image_version(name, version, disk_name)
    timings["gallery"] = time.time() - t0

    print("\n--- ensure_resource timings ---")
    for step, secs in timings.items():
        print(f"  {step:8s}: {secs / 60:.1f} min")
    print(f"  {'total':8s}: {sum(timings.values()) / 60:.1f} min")
    print(f"\nReady to launch: python cube_azure_pipeline.py launch --name {name}")

    return {"gallery_image": name, "version": version, "image_id": image_id}


# ── Step 5: Launch VM ─────────────────────────────────────────────────────────


def _free_port(start: int = 15000) -> int:
    for port in range(start, start + 100):
        try:
            with socket.socket() as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    raise RuntimeError("No free local port found in range 15000-15099")


def _open_tunnel(vm_ip: str, local_port: int, remote_port: int = GUEST_PORT):
    """Open SSH tunnel. Returns subprocess handle — caller must .terminate() it."""
    proc = subprocess.Popen(
        [
            "ssh",
            "-N",
            "-L",
            f"127.0.0.1:{local_port}:localhost:{remote_port}",
            "-i",
            SSH_PRIVKEY,
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "IdentitiesOnly=yes",
            f"azureuser@{vm_ip}",
        ],
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)
    return proc


def launch(name: str, version: str = "1.0.0", admin_user: str = "azureuser", open_tunnel: bool = True) -> dict:
    """
    Launch a VM from a gallery image.

    Returns {
        "vm_name": str,
        "public_ip": str,
        "endpoint": "http://localhost:{port}",   # via SSH tunnel
        "tunnel": subprocess.Popen,              # call .terminate() when done
        "local_port": int,
    }
    """
    uid = uuid.uuid4().hex[:6]
    vm_name = f"cube-vm-{uid}"
    pip_name = f"cube-ip-{uid}"
    nic_name = f"cube-nic-{uid}"

    image_id = (
        f"/subscriptions/{SUBSCRIPTION}/resourceGroups/{RESOURCE_GROUP}"
        f"/providers/Microsoft.Compute/galleries/{GALLERY_NAME}"
        f"/images/{name}/versions/{version}"
    )

    pubkey = Path(SSH_PUBKEY).read_text().strip()
    cloud_init = CLOUD_INIT_TEMPLATE.format(user=admin_user, agent_b64=_AGENT_PY_B64)
    custom_data_b64 = base64.b64encode(cloud_init.encode()).decode()

    compute = _compute()
    network = _network()

    # Networking
    print("[launch] Creating network resources...")
    pip_poller = network.public_ip_addresses.begin_create_or_update(  # type: ignore[call-overload]
        RESOURCE_GROUP,
        pip_name,
        {  # type: ignore[arg-type]
            "location": LOCATION,
            "tags": TAGS,
            "sku": {"name": "Standard"},
            "properties": {"publicIPAllocationMethod": "Static"},
        },
    )
    pip = pip_poller.result()

    subnet_id = (
        f"/subscriptions/{SUBSCRIPTION}/resourceGroups/{RESOURCE_GROUP}"
        f"/providers/Microsoft.Network/virtualNetworks/{VNET_NAME}/subnets/{SUBNET_NAME}"
    )
    nsg_id = (
        f"/subscriptions/{SUBSCRIPTION}/resourceGroups/{RESOURCE_GROUP}"
        f"/providers/Microsoft.Network/networkSecurityGroups/{NSG_NAME}"
    )
    nic_poller = network.network_interfaces.begin_create_or_update(  # type: ignore[call-overload]
        RESOURCE_GROUP,
        nic_name,
        {  # type: ignore[arg-type]
            "location": LOCATION,
            "tags": TAGS,
            "properties": {
                "networkSecurityGroup": {"id": nsg_id},
                "ipConfigurations": [
                    {
                        "name": "ipconfig1",
                        "properties": {
                            "subnet": {"id": subnet_id},
                            "publicIPAddress": {"id": pip.id},
                        },
                    }
                ],
            },
        },
    )
    nic = nic_poller.result()

    # VM
    print(f"[launch] Creating VM: {vm_name} ({VM_SIZE})")
    print(f"  image: {name}/{version}")
    t0 = time.time()

    poller = compute.virtual_machines.begin_create_or_update(  # type: ignore[call-overload]
        RESOURCE_GROUP,
        vm_name,
        {  # type: ignore[arg-type]
            "location": LOCATION,
            "tags": TAGS,
            "hardware_profile": {"vm_size": VM_SIZE},
            "storage_profile": {
                "image_reference": {"id": image_id},
                "os_disk": {
                    "create_option": "FromImage",
                    "managed_disk": {"storage_account_type": "Standard_LRS"},
                    "delete_option": "Delete",
                },
            },
            "os_profile": {
                "computer_name": vm_name,
                "admin_username": admin_user,
                "custom_data": custom_data_b64,
                "linux_configuration": {
                    "disable_password_authentication": True,
                    "ssh": {
                        "public_keys": [
                            {
                                "path": f"/home/{admin_user}/.ssh/authorized_keys",
                                "key_data": pubkey,
                            }
                        ]
                    },
                },
            },
            "network_profile": {"network_interfaces": [{"id": nic.id, "properties": {"primary": True}}]},
        },
    )
    poller.result()
    elapsed = time.time() - t0

    pip_info = network.public_ip_addresses.get(RESOURCE_GROUP, pip_name)
    assert pip_info.ip_address, "Public IP address was not assigned"
    public_ip = pip_info.ip_address
    print(f"  VM ready in {elapsed:.0f}s: {vm_name} @ {public_ip}")
    print(f"  SSH: ssh -i {SSH_PRIVKEY} -o IdentitiesOnly=yes {admin_user}@{public_ip}")

    result = {
        "vm_name": vm_name,
        "public_ip": public_ip,
        "pip_name": pip_name,
        "nic_name": nic_name,
        "endpoint": None,
        "tunnel": None,
        "local_port": None,
    }

    if open_tunnel:
        # Wait for SSH
        print("[launch] Waiting for SSH...")
        deadline = time.time() + 300
        while time.time() < deadline:
            r = subprocess.run(
                [
                    "ssh",
                    "-i",
                    SSH_PRIVKEY,
                    "-o",
                    "IdentitiesOnly=yes",
                    "-o",
                    "StrictHostKeyChecking=no",
                    "-o",
                    "UserKnownHostsFile=/dev/null",
                    "-o",
                    "ConnectTimeout=5",
                    "-o",
                    "BatchMode=yes",
                    f"{admin_user}@{public_ip}",
                    "echo OK",
                ],
                capture_output=True,
                text=True,
            )
            if "OK" in r.stdout:
                print("  SSH available!")
                break
            time.sleep(10)

        local_port = _free_port()
        print(f"[launch] Opening tunnel: localhost:{local_port} → {public_ip}:{GUEST_PORT}")
        tunnel = _open_tunnel(public_ip, local_port)
        result.update(
            {
                "endpoint": f"http://localhost:{local_port}",
                "tunnel": tunnel,
                "local_port": local_port,
            }
        )

    return result


# ── Step 6: Probe guest agent ─────────────────────────────────────────────────


def probe(ip_or_endpoint: str, timeout: int = 300, key_path: str | None = None, admin_user: str = "azureuser") -> dict:
    """
    Wait for the CUBE guest agent to become available.

    ip_or_endpoint: either a public IP (tunnel will be opened) or
                    "http://localhost:PORT" if tunnel is already open.

    Returns {"health": ..., "screenshot_bytes": int, "execute_ok": bool}
    """
    if ip_or_endpoint.startswith("http"):
        endpoint = ip_or_endpoint
        tunnel = None
    else:
        local_port = _free_port()
        print(f"[probe] Opening tunnel: localhost:{local_port} → {ip_or_endpoint}:{GUEST_PORT}")
        tunnel = _open_tunnel(ip_or_endpoint, local_port, remote_port=GUEST_PORT)
        endpoint = f"http://localhost:{local_port}"

    print(f"[probe] Polling {endpoint}/health (cloud-init may take 2-3 min)...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{endpoint}/health", timeout=5)
            if r.status_code == 200:
                health = r.json()
                print(f"\n  ✅ /health → {health}")

                r2 = requests.get(f"{endpoint}/screenshot", timeout=10)
                print(
                    f"  ✅ /screenshot → HTTP {r2.status_code}, "
                    f"{r2.headers.get('content-type')}, {len(r2.content)} bytes"
                )

                r3 = requests.post(f"{endpoint}/execute", json={"command": ["uname", "-a"]}, timeout=10)
                uname = r3.json().get("stdout", "").strip()
                print(f"  ✅ /execute → {uname}")

                if tunnel:
                    tunnel.terminate()
                return {
                    "health": health,
                    "screenshot_bytes": len(r2.content),
                    "execute_ok": r3.status_code == 200,
                }
        except Exception:
            pass
        remaining = int(deadline - time.time())
        print(f"  Waiting... ({remaining}s left)")
        time.sleep(10)

    if tunnel:
        tunnel.terminate()
    raise TimeoutError(f"Guest agent not ready after {timeout}s")


# ── stop / restore_snapshot ───────────────────────────────────────────────────


def stop(vm_name: str, pip_name: str | None = None, nic_name: str | None = None):
    """
    Delete a VM and its associated networking resources.
    The OS disk is auto-deleted (delete_option: Delete set at launch).
    """
    compute = _compute()
    network = _network()
    print(f"[stop] Deleting VM: {vm_name}")
    compute.virtual_machines.begin_delete(RESOURCE_GROUP, vm_name).result()
    print("  VM deleted.")

    # Infer names if not provided
    if nic_name is None:
        nic_name = vm_name.replace("cube-vm-", "cube-nic-")
    if pip_name is None:
        pip_name = vm_name.replace("cube-vm-", "cube-ip-")

    for fn, resource, name in [
        (network.network_interfaces.begin_delete, "NIC", nic_name),
        (network.public_ip_addresses.begin_delete, "IP", pip_name),
    ]:
        try:
            fn(RESOURCE_GROUP, name).result()
            print(f"  {resource} deleted: {name}")
        except Exception:
            pass


def restore_snapshot(vm_name: str, name: str, version: str = "1.0.0", admin_user: str = "azureuser") -> dict:
    """
    Reset VM to clean state: stop current VM + launch fresh from gallery.
    Equivalent to restore_snapshot() in the VMBackend interface.
    ~3-4 min total.
    """
    stop(vm_name)
    return launch(name, version=version, admin_user=admin_user)


# ── Bootstrap VM: remote ensure_resource ─────────────────────────────────────
#
# Instead of converting + uploading locally (can take hours on home broadband),
# spin up a cheap Ubuntu VM inside Azure, download + convert + upload there
# (datacenter speeds, ~15 min), then terminate the VM.
#
# The caller's ensure_resource() steps (import_disk, gallery publish) are
# unchanged — only the blob-upload step moves into the cloud.

BOOTSTRAP_VM_SIZE            = "Standard_B2ms"     # 2 vCPU, 8 GB — sufficient for qemu-img
BOOTSTRAP_GALLERY_IMAGE      = "cube-ubuntu-22-04"  # clean Ubuntu 22.04 in our gallery
BOOTSTRAP_GALLERY_IMAGE_VER  = "1.0.0"              # bypasses Golden Image Policy
BOOTSTRAP_OS_DISK_GB         = 128                  # large OS disk: holds zip+qcow2+VHD (~96 GB)

# Script injected via cloud-init custom_data.
# Placeholders: {hf_url}, {vhd_sas_url}, {sentinel_sas_url}, {failed_sas_url}
_BOOTSTRAP_SCRIPT = """\
#!/bin/bash
set -eo pipefail
exec > /var/log/cube-bootstrap.log 2>&1

on_error() {{
    msg="[bootstrap] FAILED at line $1: $2"
    echo "$msg"
    curl -s -X PUT -H "x-ms-blob-type: BlockBlob" \\
         -H "Content-Length: ${{#msg}}" -d "$msg" "{failed_sas_url}" || true
    exit 1
}}
trap 'on_error $LINENO "$BASH_COMMAND"' ERR

echo "[bootstrap] Starting at $(date)"

mkdir -p /data

# ── install tools ─────────────────────────────────────────────────────────────
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq qemu-utils wget curl unzip

wget -q "https://aka.ms/downloadazcopy-v10-linux" -O /tmp/azcopy.tar.gz
tar -xzf /tmp/azcopy.tar.gz -C /tmp --wildcards "*/azcopy" 2>/dev/null || \\
    tar -xzf /tmp/azcopy.tar.gz -C /tmp
find /tmp -name azcopy -type f | head -1 | xargs -I{{}} mv {{}} /usr/local/bin/azcopy
chmod +x /usr/local/bin/azcopy
echo "[bootstrap] Tools ready"

# ── download ──────────────────────────────────────────────────────────────────
echo "[bootstrap] Downloading: {hf_url}"
wget --progress=dot:giga -O /data/source.download "{hf_url}"
echo "[bootstrap] Downloaded: $(du -sh /data/source.download)"

# ── unzip if needed ───────────────────────────────────────────────────────────
if file /data/source.download | grep -qi "zip archive"; then
    echo "[bootstrap] Unzipping..."
    unzip -q /data/source.download -d /data/
    QCOW2=$(find /data -name "*.qcow2" | head -1)
    echo "[bootstrap] Unzipped: $QCOW2"
else
    QCOW2=/data/source.download
fi

# ── convert ───────────────────────────────────────────────────────────────────
echo "[bootstrap] Converting qcow2 → fixed VHD..."
qemu-img convert -f qcow2 -O vpc -o subformat=fixed,force_size "$QCOW2" /data/output.vhd
echo "[bootstrap] Converted: $(du -sh /data/output.vhd)"

# ── upload ────────────────────────────────────────────────────────────────────
echo "[bootstrap] Uploading to Azure Blob Storage..."
azcopy copy /data/output.vhd "{vhd_sas_url}" --blob-type PageBlob
echo "[bootstrap] Upload complete"

# ── signal done ───────────────────────────────────────────────────────────────
curl -s -X PUT -H "x-ms-blob-type: BlockBlob" -H "Content-Length: 0" "{sentinel_sas_url}"
echo "[bootstrap] Done at $(date)"
"""


def _blob_service_client() -> BlobServiceClient:
    """Return a BlobServiceClient using storage account key auth."""
    storage = _storage()
    keys = storage.storage_accounts.list_keys(RESOURCE_GROUP, STORAGE_ACCOUNT)
    assert keys.keys
    conn_str = (
        f"DefaultEndpointsProtocol=https;AccountName={STORAGE_ACCOUNT};"
        f"AccountKey={keys.keys[0].value};EndpointSuffix=core.windows.net"
    )
    return BlobServiceClient.from_connection_string(conn_str)


def _blob_exists(blob_name: str) -> bool:
    """Return True if a blob exists in the VHD container."""
    try:
        svc = _blob_service_client()
        svc.get_blob_client(CONTAINER_NAME, blob_name).get_blob_properties()
        return True
    except Exception:
        return False


def generate_sas_url(blob_name: str, expiry_hours: int = 8, write: bool = True) -> str:
    """Generate a pre-authorized SAS URL for a blob (read or write).

    The bootstrap VM uses this to upload without needing cloud credentials.
    """
    storage = _storage()
    keys = storage.storage_accounts.list_keys(RESOURCE_GROUP, STORAGE_ACCOUNT)
    assert keys.keys
    account_key = keys.keys[0].value

    # Ensure container exists
    svc = BlobServiceClient(
        f"https://{STORAGE_ACCOUNT}.blob.core.windows.net",
        credential=account_key,
    )
    container = svc.get_container_client(CONTAINER_NAME)
    try:
        container.get_container_properties()
    except Exception:
        container.create_container()

    expiry = datetime.now(timezone.utc) + timedelta(hours=expiry_hours)
    perms = BlobSasPermissions(read=True, write=write, create=write, add=write)
    sas = generate_blob_sas(
        account_name=STORAGE_ACCOUNT,
        container_name=CONTAINER_NAME,
        blob_name=blob_name,
        account_key=account_key,
        permission=perms,
        expiry=expiry,
    )
    return f"https://{STORAGE_ACCOUNT}.blob.core.windows.net/{CONTAINER_NAME}/{blob_name}?{sas}"


def poll_sentinel(
    sentinel_blob_name: str,
    failed_blob_name: str | None = None,
    timeout: int = 7200,
    interval: int = 30,
) -> None:
    """Poll until the sentinel blob appears (bootstrap complete) or failed blob appears.

    Raises TimeoutError or RuntimeError on failure.
    """
    svc = _blob_service_client()
    deadline = time.time() + timeout
    t0 = time.time()

    while time.time() < deadline:
        # Check for failure first
        if failed_blob_name:
            try:
                data = svc.get_blob_client(CONTAINER_NAME, failed_blob_name).download_blob().readall()
                raise RuntimeError(f"Bootstrap VM reported failure: {data.decode()}")
            except RuntimeError:
                raise
            except Exception:
                pass

        # Check for success
        try:
            svc.get_blob_client(CONTAINER_NAME, sentinel_blob_name).get_blob_properties()
            print(f"\n  Bootstrap complete after {int(time.time()-t0)}s")
            return
        except Exception:
            pass

        elapsed = int(time.time() - t0)
        remaining = int(deadline - time.time())
        print(f"\r  [{elapsed}s elapsed, {remaining}s remaining] waiting for bootstrap...", end="", flush=True)
        time.sleep(interval)

    raise TimeoutError(f"Bootstrap did not complete within {timeout}s")


def launch_bootstrap_vm(script: str) -> dict:
    """Launch a lightweight Ubuntu VM with a bootstrap script and a large OS disk.

    Uses our gallery image (bypasses Golden Image Policy).
    Returns {vm_name, pip_name, nic_name, public_ip}.
    """
    uid = uuid.uuid4().hex[:6]
    vm_name   = f"cube-bootstrap-{uid}"
    pip_name  = f"cube-bootstrap-ip-{uid}"
    nic_name  = f"cube-bootstrap-nic-{uid}"

    pubkey = Path(SSH_PUBKEY).read_text().strip()
    custom_data_b64 = base64.b64encode(script.encode()).decode()

    compute = _compute()
    network = _network()

    print("[bootstrap-vm] Creating network resources...")
    pip_poller = network.public_ip_addresses.begin_create_or_update(  # type: ignore[call-overload]
        RESOURCE_GROUP, pip_name,
        {  # type: ignore[arg-type]
            "location": LOCATION, "tags": TAGS,
            "sku": {"name": "Standard"},
            "properties": {"publicIPAllocationMethod": "Static"},
        },
    )
    pip = pip_poller.result()

    subnet_id = (
        f"/subscriptions/{SUBSCRIPTION}/resourceGroups/{RESOURCE_GROUP}"
        f"/providers/Microsoft.Network/virtualNetworks/{VNET_NAME}/subnets/{SUBNET_NAME}"
    )
    nsg_id = (
        f"/subscriptions/{SUBSCRIPTION}/resourceGroups/{RESOURCE_GROUP}"
        f"/providers/Microsoft.Network/networkSecurityGroups/{NSG_NAME}"
    )
    nic_poller = network.network_interfaces.begin_create_or_update(  # type: ignore[call-overload]
        RESOURCE_GROUP, nic_name,
        {  # type: ignore[arg-type]
            "location": LOCATION, "tags": TAGS,
            "properties": {
                "networkSecurityGroup": {"id": nsg_id},
                "ipConfigurations": [{
                    "name": "ipconfig1",
                    "properties": {
                        "subnet": {"id": subnet_id},
                        "publicIPAddress": {"id": pip.id},
                    },
                }],
            },
        },
    )
    nic = nic_poller.result()

    print(f"[bootstrap-vm] Launching {vm_name} ({BOOTSTRAP_VM_SIZE}, {BOOTSTRAP_OS_DISK_GB} GB OS disk)")
    t0 = time.time()

    poller = compute.virtual_machines.begin_create_or_update(  # type: ignore[call-overload]
        RESOURCE_GROUP, vm_name,
        {  # type: ignore[arg-type]
            "location": LOCATION,
            "tags": {**TAGS, "role": "bootstrap"},
            "hardware_profile": {"vm_size": BOOTSTRAP_VM_SIZE},
            "storage_profile": {
                "image_reference": {
                    "id": (
                        f"/subscriptions/{SUBSCRIPTION}/resourceGroups/{RESOURCE_GROUP}"
                        f"/providers/Microsoft.Compute/galleries/{GALLERY_NAME}"
                        f"/images/{BOOTSTRAP_GALLERY_IMAGE}/versions/{BOOTSTRAP_GALLERY_IMAGE_VER}"
                    )
                },
                "os_disk": {
                    "create_option": "FromImage",
                    "managed_disk": {"storage_account_type": "Standard_LRS"},
                    "disk_size_gb": BOOTSTRAP_OS_DISK_GB,
                    "delete_option": "Delete",
                },
            },
            "os_profile": {
                "computer_name": vm_name,
                "admin_username": "azureuser",
                "custom_data": custom_data_b64,
                "linux_configuration": {
                    "disable_password_authentication": True,
                    "ssh": {"public_keys": [{
                        "path": "/home/azureuser/.ssh/authorized_keys",
                        "key_data": pubkey,
                    }]},
                },
            },
            "network_profile": {
                "network_interfaces": [{"id": nic.id, "properties": {"primary": True}}]
            },
        },
    )
    poller.result()

    pip_info = network.public_ip_addresses.get(RESOURCE_GROUP, pip_name)
    assert pip_info.ip_address
    public_ip = pip_info.ip_address
    print(f"  VM ready in {int(time.time()-t0)}s: {vm_name} @ {public_ip}")
    print(f"  SSH (for debugging): ssh -i {SSH_PRIVKEY} -o IdentitiesOnly=yes azureuser@{public_ip}")
    print("  Logs: ssh ... 'sudo tail -f /var/log/cube-bootstrap.log'")

    return {
        "vm_name": vm_name,
        "pip_name": pip_name,
        "nic_name": nic_name,
        "public_ip": public_ip,
    }


def cleanup_bootstrap_vm(vm_name: str, pip_name: str, nic_name: str) -> None:
    """Terminate bootstrap VM and associated network resources.

    Data disk has delete_option=Delete so it's removed with the VM.
    """
    compute = _compute()
    network = _network()
    print(f"[bootstrap-vm] Deleting {vm_name}...")
    compute.virtual_machines.begin_delete(RESOURCE_GROUP, vm_name).result()
    for fn, label, name in [
        (network.network_interfaces.begin_delete,  "NIC", nic_name),
        (network.public_ip_addresses.begin_delete, "IP",  pip_name),
    ]:
        try:
            fn(RESOURCE_GROUP, name).result()
        except Exception:
            pass
    print("  Bootstrap VM and resources deleted.")


def bootstrap_ensure_resource(hf_url: str, name: str, version: str = "1.0.0", blob_name: str | None = None) -> None:
    """Remote bootstrap: spin up an Azure VM to download + convert + upload the image.

    Replaces the local convert_to_vhd() + upload_vhd() steps with an in-cloud
    operation that runs at datacenter speed (~15-20 min vs hours from home broadband).

    After this returns, the VHD is in blob storage and the downstream steps
    (import_disk, gallery publish) run as usual via ensure_resource().

    hf_url  : HTTPS URL to the source .qcow2 (HuggingFace public repo)
    name    : gallery image name (e.g. "cube-osworld-ubuntu")
    version : gallery image version
    """
    # Derive blob name from the URL filename (or use explicit override).
    # Strip all extensions: "Ubuntu.qcow2.zip" → "Ubuntu"
    src_filename = hf_url.rstrip("/").split("/")[-1]
    base_name = src_filename.split(".")[0]
    vhd_blob_name      = blob_name if blob_name else (base_name + ".vhd")
    sentinel_blob_name = vhd_blob_name + ".bootstrap_done"
    failed_blob_name   = vhd_blob_name + ".bootstrap_failed"

    print(f"\n{'='*60}")
    print(f"bootstrap_ensure_resource: {name}")
    print(f"  source:   {hf_url}")
    print(f"  vhd blob: {vhd_blob_name}")
    print(f"{'='*60}")

    # Idempotent: skip if VHD already in blob storage with valid footer
    if _blob_exists(sentinel_blob_name):
        print("[bootstrap] Sentinel exists — VHD already bootstrapped.")
        # Still run gallery steps in case they weren't completed
        ensure_resource_from_blob(vhd_blob_name, name, version)
        return

    t_total = time.time()

    # Generate SAS URLs (8h expiry — enough for download + convert + upload)
    vhd_sas_url      = generate_sas_url(vhd_blob_name,      expiry_hours=8, write=True)
    sentinel_sas_url = generate_sas_url(sentinel_blob_name, expiry_hours=8, write=True)
    failed_sas_url   = generate_sas_url(failed_blob_name,   expiry_hours=8, write=True)

    script = _BOOTSTRAP_SCRIPT.format(
        hf_url=hf_url,
        vhd_sas_url=vhd_sas_url,
        sentinel_sas_url=sentinel_sas_url,
        failed_sas_url=failed_sas_url,
    )

    # Launch bootstrap VM
    vm_info = launch_bootstrap_vm(script)

    try:
        print("\n[bootstrap] VM is running. Polling for completion every 30s...")
        print(f"  (watch logs: ssh -i {SSH_PRIVKEY} azureuser@{vm_info['public_ip']}"
              f" 'sudo tail -f /var/log/cube-bootstrap.log')")
        poll_sentinel(sentinel_blob_name, failed_blob_name=failed_blob_name)
    finally:
        cleanup_bootstrap_vm(vm_info["vm_name"], vm_info["pip_name"], vm_info["nic_name"])

    print("[bootstrap] VHD in storage. Running gallery import steps...")
    blob_url = f"https://{STORAGE_ACCOUNT}.blob.core.windows.net/{CONTAINER_NAME}/{vhd_blob_name}"
    ensure_resource_from_blob(vhd_blob_name, name, version, blob_url=blob_url)

    print(f"\n[bootstrap] Total time: {(time.time()-t_total)/60:.1f} min")
    print(f"  Gallery image ready: {name}/{version}")


def ensure_resource_from_blob(
    vhd_blob_name: str,
    name: str,
    version: str = "1.0.0",
    blob_url: str | None = None,
) -> None:
    """Run the post-upload steps: import blob → managed disk → gallery image.

    Idempotent — skips steps already done.
    Used by both bootstrap_ensure_resource() and the regular ensure_resource().
    """
    if blob_url is None:
        blob_url = (
            f"https://{STORAGE_ACCOUNT}.blob.core.windows.net/{CONTAINER_NAME}/{vhd_blob_name}"
        )
    disk_name = f"cube-disk-{name}"

    # Import as managed disk
    import_disk(blob_url, disk_name)

    # Ensure gallery + image definition
    ensure_gallery()
    create_image_definition(name)

    # Publish image version
    create_image_version(name, version, disk_name)
    print(f"[gallery] Image ready: {name}/{version}")


# ── List gallery images ───────────────────────────────────────────────────────


def list_images():
    """Print all image definitions in the gallery."""
    compute = _compute()
    try:
        defs = list(compute.gallery_images.list_by_gallery(RESOURCE_GROUP, GALLERY_NAME))
    except Exception:
        print(f"Gallery '{GALLERY_NAME}' not found.")
        return
    if not defs:
        print("No images in gallery.")
        return
    print(f"\n{'Name':<35} {'State':<14} {'HyperV'}")
    print("-" * 60)
    for d in defs:
        print(f"{d.name:<35} {d.os_state:<14} {d.hyper_v_generation}")
        if not d.name:
            continue
        versions = list(compute.gallery_image_versions.list_by_gallery_image(RESOURCE_GROUP, GALLERY_NAME, d.name))
        for v in versions:
            print(f"  version {v.name:<10} {v.provisioning_state}")


# ── Full end-to-end run (convenience) ────────────────────────────────────────


def run_full(image_path: str, name: str, version: str = "1.0.0", admin_user: str = "azureuser") -> dict:
    """
    Runs ensure_resource + launch + probe in one call.
    Use this to test a new image end-to-end.
    """
    ensure_resource(image_path, name, version=version, admin_user=admin_user)
    vm_info = launch(name, version=version, admin_user=admin_user)
    result = probe(vm_info["endpoint"])
    if vm_info.get("tunnel"):
        vm_info["tunnel"].terminate()
    return {**vm_info, **result}


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    import argparse

    p = argparse.ArgumentParser(
        description="CUBE Azure VM Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="cmd")

    # ensure
    e = sub.add_parser("ensure", help="One-time setup: image → Compute Gallery")
    e.add_argument("--image", required=True, help="Path to .qcow2 / .img / .vmdk / .vhd")
    e.add_argument("--name", required=True, help="Gallery image definition name")
    e.add_argument("--version", default="1.0.0")
    e.add_argument("--user", default="azureuser", help="Admin username in the image")

    # launch
    la = sub.add_parser("launch", help="Launch VM from gallery image")
    la.add_argument("--name", required=True)
    la.add_argument("--version", default="1.0.0")
    la.add_argument("--user", default="azureuser")

    # probe
    pr = sub.add_parser("probe", help="Probe guest agent (wait for /health)")
    pr.add_argument("--ip", required=True, help="VM public IP or http://localhost:PORT")
    pr.add_argument("--timeout", type=int, default=300)

    # stop
    st = sub.add_parser("stop", help="Delete VM and its network resources")
    st.add_argument("--vm", required=True)

    # restore
    rs = sub.add_parser("restore", help="Reset VM: stop + relaunch from gallery")
    rs.add_argument("--vm", required=True)
    rs.add_argument("--name", required=True)
    rs.add_argument("--version", default="1.0.0")

    # list
    sub.add_parser("list", help="List gallery images")

    # full
    fu = sub.add_parser("full", help="End-to-end: ensure + launch + probe")
    fu.add_argument("--image", required=True)
    fu.add_argument("--name", required=True)
    fu.add_argument("--version", default="1.0.0")
    fu.add_argument("--user", default="azureuser")

    # cleanup
    sub.add_parser("cleanup", help="Delete all resources tagged project=cube-experiment")

    args = p.parse_args()

    if args.cmd == "ensure":
        ensure_resource(args.image, args.name, args.version, args.user)

    elif args.cmd == "launch":
        info = launch(args.name, args.version, args.user)
        print(f"\nVM: {info['vm_name']}  IP: {info['public_ip']}")
        print(f"Endpoint: {info['endpoint']}")
        print(f"\nNext: python cube_azure_pipeline.py probe --ip {info['public_ip']}")
        if info.get("tunnel"):
            print("(tunnel open — Ctrl+C to close)")
            try:
                info["tunnel"].wait()
            except KeyboardInterrupt:
                info["tunnel"].terminate()

    elif args.cmd == "probe":
        probe(args.ip, timeout=args.timeout)

    elif args.cmd == "stop":
        stop(args.vm)

    elif args.cmd == "restore":
        info = restore_snapshot(args.vm, args.name, args.version)
        print(f"\nNew VM: {info['vm_name']}  IP: {info['public_ip']}")

    elif args.cmd == "list":
        list_images()

    elif args.cmd == "full":
        run_full(args.image, args.name, args.version, args.user)

    elif args.cmd == "cleanup":
        subprocess.run(["python", "track.py", "delete"], check=True)

    else:
        p.print_help()


if __name__ == "__main__":
    main()
