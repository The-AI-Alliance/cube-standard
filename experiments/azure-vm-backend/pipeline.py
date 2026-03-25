"""
Full pipeline: qcow2 → Azure VM

Steps:
  1. convert   — qcow2 → fixed VHD (local, qemu-img)
  2. upload    — VHD → Azure Blob Storage (westus2)
  3. import    — Blob → Azure Managed Disk
  4. snapshot  — Managed Disk → Snapshot (reusable base)
  5. launch    — Snapshot → Disk → VM (repeatable, isolated)
  6. probe     — wait for /screenshot on port 5000

Run individual steps:
  python pipeline.py convert  --qcow2 tiny.qcow2
  python pipeline.py upload   --vhd   tiny.vhd
  python pipeline.py import   --blob-url https://...
  python pipeline.py snapshot --disk  cube-exp-disk-abc
  python pipeline.py launch   --snapshot cube-exp-snap-abc
  python pipeline.py run      --qcow2 tiny.qcow2   # full pipeline
  python pipeline.py probe    --ip 1.2.3.4
"""

import argparse
import os
import subprocess
import time
import uuid
from pathlib import Path

import requests
from azure.identity import DefaultAzureCredential, AzureCliCredential
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.network import NetworkManagementClient
from azure.mgmt.storage import StorageManagementClient
from azure.storage.blob import BlobServiceClient, BlobClient

from track import record

# ── Constants ─────────────────────────────────────────────────────────────────
SUBSCRIPTION   = "aeb958d3-a614-450e-94bc-88f284dc0664"
RESOURCE_GROUP = "ui_assist"
LOCATION       = "westus2"
STORAGE_ACCOUNT = "cubeexpvhd"     # will be created in westus2
CONTAINER_NAME  = "vhds"
VNET_NAME       = "vnet-westus2"
SUBNET_NAME     = "snet-westus2-1"
NSG_NAME        = "osworld-nsg"    # reuse existing NSG (has port 22,80,443,3389)
SSH_KEY_NAME    = "os_world_key"   # reuse existing SSH key in resource group

TAGS           = {"project": "cube-experiment"}
GUEST_PORT     = 5000
READY_TIMEOUT  = 300
READY_POLL     = 5

# VM sizes to try (cheapest first for experiments)
VM_SIZE        = "Standard_D4s_v3"   # 4 vCPU, 16 GB RAM


# ── Credentials ───────────────────────────────────────────────────────────────

def _cred():
    # AzureCliCredential reuses existing `az login` session — no env vars needed
    return AzureCliCredential()

def _compute() -> ComputeManagementClient:
    return ComputeManagementClient(_cred(), SUBSCRIPTION)

def _network() -> NetworkManagementClient:
    return NetworkManagementClient(_cred(), SUBSCRIPTION)

def _storage_mgmt() -> StorageManagementClient:
    return StorageManagementClient(_cred(), SUBSCRIPTION)


# ── Step 1: Convert ───────────────────────────────────────────────────────────

def step_convert(qcow2_path: str) -> str:
    """
    Convert qcow2 → fixed-size VHD.

    Azure requires:
      - VHD format (not VHDX, not raw)
      - Fixed subformat (not dynamic)
      - Size aligned to 1 MB (force_size ensures this)

    Returns path to .vhd file.
    """
    src = Path(qcow2_path).resolve()
    dst = src.with_suffix(".vhd")

    print(f"[convert] {src.name} → {dst.name}")

    # Get source size for info
    info = subprocess.run(
        ["qemu-img", "info", "--output=json", str(src)],
        capture_output=True, text=True, check=True,
    )
    import json
    info_data = json.loads(info.stdout)
    size_gb = info_data["virtual-size"] / (1024**3)
    print(f"  Virtual size: {size_gb:.1f} GB")
    print(f"  Format: {info_data['format']}")

    # Convert — this can take a while for large images
    print("  Converting (this may take a few minutes for large images)...")
    subprocess.run(
        [
            "qemu-img", "convert",
            "-f", "qcow2",
            "-O", "vpc",
            "-o", "subformat=fixed,force_size",
            str(src), str(dst),
        ],
        check=True,
    )
    vhd_size_gb = dst.stat().st_size / (1024**3)
    print(f"  Done → {dst.name} ({vhd_size_gb:.1f} GB on disk)")
    return str(dst)


# ── Step 2: Upload ────────────────────────────────────────────────────────────

def _ensure_storage_account() -> str:
    """Create storage account in westus2 if it doesn't exist. Returns connection string."""
    storage = _storage_mgmt()
    try:
        acct = storage.storage_accounts.get_properties(RESOURCE_GROUP, STORAGE_ACCOUNT)
        print(f"  [upload] Storage account exists: {STORAGE_ACCOUNT}")
    except Exception:
        print(f"  [upload] Creating storage account: {STORAGE_ACCOUNT} in {LOCATION}...")
        poller = storage.storage_accounts.begin_create(
            RESOURCE_GROUP, STORAGE_ACCOUNT,
            {
                "location": LOCATION,
                "tags": TAGS,
                "sku": {"name": "Standard_LRS"},
                "kind": "StorageV2",
            }
        )
        acct = poller.result()
        record(STORAGE_ACCOUNT, "Microsoft.Storage/storageAccounts", acct.id)
        print(f"  Storage account created.")

    # Get key
    keys = storage.storage_accounts.list_keys(RESOURCE_GROUP, STORAGE_ACCOUNT)
    key = keys.keys[0].value
    return f"DefaultEndpointsProtocol=https;AccountName={STORAGE_ACCOUNT};AccountKey={key};EndpointSuffix=core.windows.net"


def step_upload(vhd_path: str) -> str:
    """
    Upload VHD to Azure Blob Storage.
    Returns the blob URL.
    """
    vhd = Path(vhd_path).resolve()
    blob_name = vhd.name
    size = vhd.stat().st_size

    print(f"[upload] {vhd.name} ({size / (1024**3):.1f} GB)")

    conn_str = _ensure_storage_account()
    svc = BlobServiceClient.from_connection_string(conn_str)

    # Create container if needed
    container = svc.get_container_client(CONTAINER_NAME)
    try:
        container.get_container_properties()
    except Exception:
        print(f"  Creating container: {CONTAINER_NAME}")
        container.create_container()

    blob_client = svc.get_blob_client(CONTAINER_NAME, blob_name)
    print(f"  Uploading to {STORAGE_ACCOUNT}/{CONTAINER_NAME}/{blob_name} ...")

    with open(vhd, "rb") as f:
        blob_client.upload_blob(
            f,
            blob_type="PageBlob",   # VHDs must be page blobs
            overwrite=True,
            max_concurrency=4,
        )

    blob_url = f"https://{STORAGE_ACCOUNT}.blob.core.windows.net/{CONTAINER_NAME}/{blob_name}"
    print(f"  Uploaded: {blob_url}")
    return blob_url


# ── Step 3: Import as Managed Disk ───────────────────────────────────────────

def step_import(blob_url: str) -> str:
    """
    Create a Managed Disk from the uploaded VHD blob.
    Returns the disk name.
    """
    uid = uuid.uuid4().hex[:6]
    disk_name = f"cube-exp-disk-{uid}"

    print(f"[import] Blob → Managed Disk: {disk_name}")

    compute = _compute()
    poller = compute.disks.begin_create_or_update(
        RESOURCE_GROUP, disk_name,
        {
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
        }
    )
    disk = poller.result()
    record(disk_name, "Microsoft.Compute/disks", disk.id)
    print(f"  Disk created: {disk_name}")
    return disk_name


# ── Step 4: Snapshot ──────────────────────────────────────────────────────────

def step_snapshot(disk_name: str) -> str:
    """
    Create a snapshot from the managed disk.
    This is the reusable base — each VM gets a fresh disk from this snapshot.
    Returns snapshot name.
    """
    uid = uuid.uuid4().hex[:6]
    snap_name = f"cube-exp-snap-{uid}"

    compute = _compute()
    disk = compute.disks.get(RESOURCE_GROUP, disk_name)

    print(f"[snapshot] Disk → Snapshot: {snap_name}")
    poller = compute.snapshots.begin_create_or_update(
        RESOURCE_GROUP, snap_name,
        {
            "location": LOCATION,
            "tags": TAGS,
            "properties": {
                "creationData": {
                    "createOption": "Copy",
                    "sourceResourceId": disk.id,
                },
                "incremental": False,
            },
        }
    )
    snap = poller.result()
    record(snap_name, "Microsoft.Compute/snapshots", snap.id)
    print(f"  Snapshot created: {snap_name}")
    return snap_name


# ── Step 5: Launch VM ─────────────────────────────────────────────────────────

def _get_ssh_pubkey() -> str:
    result = subprocess.run(
        ["az", "sshkey", "show", "-g", RESOURCE_GROUP, "-n", SSH_KEY_NAME,
         "--query", "publicKey", "-o", "tsv"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def _create_pip(network: NetworkManagementClient, name: str) -> str:
    print(f"  Creating public IP: {name}")
    poller = network.public_ip_addresses.begin_create_or_update(
        RESOURCE_GROUP, name,
        {"location": LOCATION, "tags": TAGS,
         "sku": {"name": "Standard"},
         "properties": {"publicIPAllocationMethod": "Static"}},
    )
    pip = poller.result()
    record(name, "Microsoft.Network/publicIPAddresses", pip.id)
    return pip.id


def _create_nic(network: NetworkManagementClient, name: str, pip_id: str) -> str:
    print(f"  Creating NIC: {name}")
    subnet_id = (
        f"/subscriptions/{SUBSCRIPTION}/resourceGroups/{RESOURCE_GROUP}"
        f"/providers/Microsoft.Network/virtualNetworks/{VNET_NAME}/subnets/{SUBNET_NAME}"
    )
    nsg_id = (
        f"/subscriptions/{SUBSCRIPTION}/resourceGroups/{RESOURCE_GROUP}"
        f"/providers/Microsoft.Network/networkSecurityGroups/{NSG_NAME}"
    )
    poller = network.network_interfaces.begin_create_or_update(
        RESOURCE_GROUP, name,
        {
            "location": LOCATION,
            "tags": TAGS,
            "properties": {
                "networkSecurityGroup": {"id": nsg_id},
                "ipConfigurations": [{
                    "name": "ipconfig1",
                    "properties": {
                        "subnet": {"id": subnet_id},
                        "publicIPAddress": {"id": pip_id},
                    },
                }],
            },
        },
    )
    nic = poller.result()
    record(name, "Microsoft.Network/networkInterfaces", nic.id)
    return nic.id


def step_launch(snapshot_name: str) -> dict:
    """
    Launch a VM from snapshot:
      1. Create a fresh disk from the snapshot (Specialized — no cloud-init)
      2. Attach disk to new VM
      3. Wait for guest agent on port 5000

    Returns {"vm_name": ..., "public_ip": ..., "endpoint": ...}
    """
    uid = uuid.uuid4().hex[:6]
    vm_name   = f"cube-exp-vm-{uid}"
    disk_name = f"cube-exp-osdisk-{uid}"
    pip_name  = f"cube-exp-ip-{uid}"
    nic_name  = f"cube-exp-nic-{uid}"

    compute = _compute()
    network = _network()

    # Get snapshot
    snap = compute.snapshots.get(RESOURCE_GROUP, snapshot_name)

    # 1. Create OS disk from snapshot
    print(f"[launch] Creating OS disk from snapshot: {disk_name}")
    poller = compute.disks.begin_create_or_update(
        RESOURCE_GROUP, disk_name,
        {
            "location": LOCATION,
            "tags": TAGS,
            "sku": {"name": "Standard_LRS"},
            "properties": {
                "creationData": {
                    "createOption": "Copy",
                    "sourceResourceId": snap.id,
                },
                "osType": "Linux",
            },
        }
    )
    disk = poller.result()
    record(disk_name, "Microsoft.Compute/disks", disk.id)

    # 2. Create networking
    pip_id = _create_pip(network, pip_name)
    nic_id = _create_nic(network, nic_name, pip_id)

    # 3. Create VM — Specialized (no os_profile, no cloud-init)
    print(f"[launch] Creating VM: {vm_name} ({VM_SIZE})")
    poller = compute.virtual_machines.begin_create_or_update(
        RESOURCE_GROUP, vm_name,
        {
            "location": LOCATION,
            "tags": TAGS,
            "hardware_profile": {"vm_size": VM_SIZE},
            "storage_profile": {
                "os_disk": {
                    "create_option": "Attach",           # attach existing disk
                    "managed_disk": {"id": disk.id},
                    "os_type": "Linux",
                    "delete_option": "Delete",           # auto-delete when VM deleted
                },
            },
            # No os_profile for Specialized images
            "network_profile": {
                "network_interfaces": [{"id": nic_id, "properties": {"primary": True}}]
            },
        }
    )
    vm = poller.result()
    record(vm_name, "Microsoft.Compute/virtualMachines", vm.id)

    # 4. Get public IP
    pip = network.public_ip_addresses.get(RESOURCE_GROUP, pip_name)
    public_ip = pip.ip_address
    endpoint = f"http://{public_ip}:{GUEST_PORT}"

    print(f"[launch] VM created: {vm_name} @ {public_ip}")
    return {"vm_name": vm_name, "public_ip": public_ip, "endpoint": endpoint}


def _find_free_port(start: int = 15000) -> int:
    import socket
    for port in range(start, start + 100):
        try:
            with socket.socket() as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    raise RuntimeError("No free port found")


def _open_ssh_tunnel(vm_ip: str, key_path: str, local_port: int, remote_port: int):
    """Open SSH tunnel. Returns subprocess handle — caller must .terminate() it."""
    proc = subprocess.Popen([
        "ssh", "-N",
        "-L", f"127.0.0.1:{local_port}:localhost:{remote_port}",
        "-i", key_path,
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=30",
        f"azureuser@{vm_ip}",
    ], stderr=subprocess.DEVNULL)
    time.sleep(2)  # let tunnel establish
    return proc


# ── Step 6: Probe guest agent ─────────────────────────────────────────────────

def step_probe(ip: str, timeout: int = READY_TIMEOUT) -> bool:
    """
    Wait for the CUBE guest agent HTTP server on port 5000.
    Returns True if ready, raises TimeoutError otherwise.
    """
    endpoint = f"http://{ip}:{GUEST_PORT}/screenshot"
    print(f"[probe] Waiting for guest agent at {endpoint} ...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(endpoint, timeout=5)
            if r.status_code == 200:
                print(f"  Ready! (content-type: {r.headers.get('content-type')})")
                return True
            else:
                print(f"  HTTP {r.status_code} — VM booted but guest agent not ready yet")
        except requests.exceptions.ConnectionError:
            pass  # VM still booting
        except requests.exceptions.RequestException as e:
            print(f"  {e}")
        print(f"  Waiting... ({int(deadline - time.time())}s left)")
        time.sleep(READY_POLL)
    raise TimeoutError(f"Guest agent not ready after {timeout}s")


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_all(qcow2_path: str):
    """Run the complete pipeline end-to-end."""
    print("=" * 60)
    print("CUBE Azure VM Pipeline")
    print("=" * 60)

    vhd_path    = step_convert(qcow2_path)
    blob_url    = step_upload(vhd_path)
    disk_name   = step_import(blob_url)
    snap_name   = step_snapshot(disk_name)
    vm_info     = step_launch(snap_name)
    step_probe(vm_info["public_ip"])

    print("\n" + "=" * 60)
    print(f"Pipeline complete!")
    print(f"  VM:       {vm_info['vm_name']}")
    print(f"  IP:       {vm_info['public_ip']}")
    print(f"  Endpoint: {vm_info['endpoint']}")
    print(f"\nTo clean up: python track.py delete")
    print("=" * 60)
    return vm_info


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CUBE Azure VM pipeline")
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("convert",  help="Step 1: qcow2 → VHD")
    p.add_argument("--qcow2", required=True)

    p = sub.add_parser("upload",   help="Step 2: VHD → Blob Storage")
    p.add_argument("--vhd", required=True)

    p = sub.add_parser("import",   help="Step 3: Blob → Managed Disk")
    p.add_argument("--blob-url", required=True)

    p = sub.add_parser("snapshot", help="Step 4: Disk → Snapshot")
    p.add_argument("--disk", required=True)

    p = sub.add_parser("launch",   help="Step 5: Snapshot → VM")
    p.add_argument("--snapshot", required=True)

    p = sub.add_parser("probe",    help="Step 6: Wait for guest agent")
    p.add_argument("--ip", required=True)
    p.add_argument("--timeout", type=int, default=READY_TIMEOUT)

    p = sub.add_parser("run",      help="Full pipeline end-to-end")
    p.add_argument("--qcow2", required=True)

    args = parser.parse_args()

    if args.cmd == "convert":
        step_convert(args.qcow2)
    elif args.cmd == "upload":
        step_upload(args.vhd)
    elif args.cmd == "import":
        step_import(args.blob_url)
    elif args.cmd == "snapshot":
        step_snapshot(args.disk)
    elif args.cmd == "launch":
        info = step_launch(args.snapshot)
        print(f"\n{info}")
        print(f"\nNow run: python pipeline.py probe --ip {info['public_ip']}")
    elif args.cmd == "probe":
        step_probe(args.ip, args.timeout)
    elif args.cmd == "run":
        run_all(args.qcow2)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
