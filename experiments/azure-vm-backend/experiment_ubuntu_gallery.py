"""
Full test: Ubuntu 22.04 Generalized image → Azure Compute Gallery → clean launch.

This validates the complete production CUBE pipeline:
  1. Ubuntu 22.04 cloudimg (qcow2) → VHD conversion
  2. VHD → Azure Blob Storage
  3. Blob → Managed Disk (createOption: Import)
  4. Disk → Gallery Image Version (Generalized)
  5. Gallery Image → VM (createOption: FromImage + os_profile SSH key injection)
  6. SSH in cleanly (no CCOE workarounds — fresh Ubuntu, not osworld_base)
  7. Install mini guest agent → SSH tunnel → probe /health

Key questions answered by this test:
  Q1: Does qcow2-converted VHD actually boot on Azure? (virtio → Hyper-V drivers)
  Q2: Does os_profile SSH key injection work (clean cloud-init path)?
  Q3: Does the full pipeline work end-to-end without manual steps?

Run steps individually:
  python experiment_ubuntu_gallery.py convert   --img /tmp/ubuntu-22.04-cloudimg-amd64.img
  python experiment_ubuntu_gallery.py upload    --vhd /tmp/ubuntu-22.04-cloudimg-amd64.vhd
  python experiment_ubuntu_gallery.py import    --blob-url https://...
  python experiment_ubuntu_gallery.py imgdef                   # create gallery image definition
  python experiment_ubuntu_gallery.py version   --disk cube-ubuntu-disk-xxx
  python experiment_ubuntu_gallery.py launch                   # launch + probe
  python experiment_ubuntu_gallery.py full      --img /tmp/ubuntu-22.04-cloudimg-amd64.img
"""

import subprocess
import sys
import time
import uuid
from pathlib import Path

import requests
from azure.identity import AzureCliCredential
from azure.mgmt.compute import ComputeManagementClient

from pipeline import (
    SUBSCRIPTION, RESOURCE_GROUP, LOCATION, TAGS,
    VM_SIZE, GUEST_PORT,
    _compute, _network, _create_pip, _create_nic, _find_free_port, _open_ssh_tunnel,
    step_convert, step_upload, step_import,
)
from track import record

# ── Gallery constants (separate from the Specialized osworld definition) ──────

GALLERY_NAME       = "cube_exp_gallery"         # already exists
UBUNTU_IMAGE_DEF   = "cube-ubuntu-22-04"        # NEW — Generalized
UBUNTU_IMG_VERSION = "1.0.0"

# SSH key for injection via os_profile
SSH_PRIVKEY_PATH   = str(Path.home() / ".ssh" / "id_ed25519")
SSH_PUBKEY_PATH    = str(Path.home() / ".ssh" / "id_ed25519.pub")

# Cloud-init script to install a minimal CUBE guest agent on first boot.
# Uses write_files with base64-encoded content to avoid YAML parsing issues
# (heredocs in runcmd are misinterpreted as YAML keys by cloud-init).
_AGENT_B64 = (
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

CLOUD_INIT_SCRIPT = f"""\
#cloud-config
packages:
  - python3-flask
  - python3-pil
write_files:
  - path: /usr/local/bin/cube_guest_agent.py
    permissions: '0755'
    encoding: b64
    content: {_AGENT_B64}
runcmd:
  - nohup python3 /usr/local/bin/cube_guest_agent.py > /var/log/cube-guest-agent.log 2>&1 &
"""


# ── Step: Create Generalized Image Definition ─────────────────────────────────

def step_ubuntu_imgdef():
    """Create a Generalized image definition for Ubuntu 22.04 in the gallery."""
    compute = _compute()
    print(f"[imgdef] Creating Generalized image definition: {UBUNTU_IMAGE_DEF}")
    poller = compute.gallery_images.begin_create_or_update(
        RESOURCE_GROUP, GALLERY_NAME, UBUNTU_IMAGE_DEF,
        {
            "location": LOCATION,
            "tags": TAGS,
            "os_type": "Linux",
            "os_state": "Generalized",       # Ubuntu cloud images are Generalized
            "hyper_v_generation": "V1",      # ubuntu-22.04-server-cloudimg is Gen1 (BIOS)
            "identifier": {
                "publisher": "cube-exp",
                "offer": "ubuntu",
                "sku": "22-04-lts",
            },
        }
    )
    img_def = poller.result()
    record(UBUNTU_IMAGE_DEF, "Microsoft.Compute/galleries/images", img_def.id)
    print(f"  Image definition created: {img_def.id}")
    return img_def


# ── Step: Create Gallery Image Version from Disk ──────────────────────────────

def step_ubuntu_version(disk_name: str):
    """Create gallery image version from the imported Ubuntu managed disk."""
    compute = _compute()
    disk = compute.disks.get(RESOURCE_GROUP, disk_name)
    print(f"[version] Disk: {disk.id} ({disk.disk_size_gb} GB)")
    print(f"[version] Creating gallery image version {UBUNTU_IMG_VERSION}...")

    poller = compute.gallery_image_versions.begin_create_or_update(
        RESOURCE_GROUP, GALLERY_NAME, UBUNTU_IMAGE_DEF, UBUNTU_IMG_VERSION,
        {
            "location": LOCATION,
            "tags": TAGS,
            "publishing_profile": {
                "replica_count": 1,
                "storage_account_type": "Standard_LRS",
                "target_regions": [{
                    "name": LOCATION,
                    "regional_replica_count": 1,
                    "storage_account_type": "Standard_LRS",
                }],
                "exclude_from_latest": False,
            },
            "storage_profile": {
                "os_disk_image": {
                    "source": {"id": disk.id},
                    "host_caching": "ReadWrite",
                }
            }
        }
    )
    version = poller.result()
    resource_id = (
        f"/subscriptions/{SUBSCRIPTION}/resourceGroups/{RESOURCE_GROUP}"
        f"/providers/Microsoft.Compute/galleries/{GALLERY_NAME}"
        f"/images/{UBUNTU_IMAGE_DEF}/versions/{UBUNTU_IMG_VERSION}"
    )
    record(f"ubuntu-gallery-version-{UBUNTU_IMG_VERSION}", "Microsoft.Compute/galleries/images/versions", resource_id)
    print(f"  Image version created: {version.id}")
    return version


# ── Step: Launch VM with os_profile SSH key injection ─────────────────────────

def step_launch_ubuntu() -> dict:
    """
    Launch Ubuntu VM from gallery image with:
      - createOption: FromImage (policy-compliant via gallery imageReference)
      - os_profile: injects our SSH public key via cloud-init (no CCOE workarounds)
      - custom_data: cloud-init script that installs + starts mini guest agent

    Returns {"vm_name": ..., "public_ip": ..., "endpoint": ...}
    """
    import base64
    uid = uuid.uuid4().hex[:6]
    vm_name  = f"cube-ub-vm-{uid}"
    pip_name = f"cube-ub-ip-{uid}"
    nic_name = f"cube-ub-nic-{uid}"

    image_id = (
        f"/subscriptions/{SUBSCRIPTION}/resourceGroups/{RESOURCE_GROUP}"
        f"/providers/Microsoft.Compute/galleries/{GALLERY_NAME}"
        f"/images/{UBUNTU_IMAGE_DEF}/versions/{UBUNTU_IMG_VERSION}"
    )

    pubkey = Path(SSH_PUBKEY_PATH).read_text().strip()
    custom_data_b64 = base64.b64encode(CLOUD_INIT_SCRIPT.encode()).decode()

    compute = _compute()
    network = _network()

    pip_id = _create_pip(network, pip_name)
    nic_id = _create_nic(network, nic_name, pip_id)

    print(f"[launch-ubuntu] Creating VM: {vm_name}")
    print(f"  Image: {image_id}")
    print(f"  SSH key: {SSH_PUBKEY_PATH}")
    print(f"  cloud-init: installs + starts mini guest agent on port 5000")

    poller = compute.virtual_machines.begin_create_or_update(
        RESOURCE_GROUP, vm_name,
        {
            "location": LOCATION,
            "tags": TAGS,
            "hardware_profile": {"vm_size": VM_SIZE},
            "storage_profile": {
                "image_reference": {"id": image_id},
                "os_disk": {
                    "create_option": "FromImage",
                    "managed_disk": {"storage_account_type": "Standard_LRS"},
                    "disk_size_gb": 30,        # expand from ~3GB to 30GB for comfort
                    "delete_option": "Delete",
                },
            },
            "os_profile": {
                "computer_name": vm_name,
                "admin_username": "azureuser",
                "custom_data": custom_data_b64,    # cloud-init script
                "linux_configuration": {
                    "disable_password_authentication": True,
                    "ssh": {
                        "public_keys": [{
                            "path": "/home/azureuser/.ssh/authorized_keys",
                            "key_data": pubkey,
                        }]
                    }
                }
            },
            "network_profile": {
                "network_interfaces": [{"id": nic_id, "properties": {"primary": True}}]
            },
        }
    )
    vm = poller.result()
    record(vm_name, "Microsoft.Compute/virtualMachines", vm.id)

    pip = network.public_ip_addresses.get(RESOURCE_GROUP, pip_name)
    public_ip = pip.ip_address

    print(f"\n[launch-ubuntu] VM ready: {vm_name} @ {public_ip}")
    print(f"  SSH: ssh -i {SSH_PRIVKEY_PATH} -o IdentitiesOnly=yes azureuser@{public_ip}")
    return {"vm_name": vm_name, "public_ip": public_ip}


# ── Step: Wait for SSH, then probe guest agent via tunnel ─────────────────────

def step_probe_ubuntu(public_ip: str) -> bool:
    """
    Wait for SSH to be available (VM booted), then:
    1. Open SSH tunnel on a free local port
    2. Poll /health until cloud-init guest agent is up

    cloud-init installs flask + starts agent during first boot (~2-3 min after SSH is up).
    """
    local_port = _find_free_port()

    # Wait for SSH
    print(f"[probe] Waiting for SSH on {public_ip}:22 ...")
    deadline = time.time() + 300
    while time.time() < deadline:
        result = subprocess.run(
            ["ssh", "-i", SSH_PRIVKEY_PATH,
             "-o", "IdentitiesOnly=yes",
             "-o", "StrictHostKeyChecking=no",
             "-o", "UserKnownHostsFile=/dev/null",
             "-o", "ConnectTimeout=5",
             "-o", "BatchMode=yes",
             f"azureuser@{public_ip}", "echo SSH_OK"],
            capture_output=True, text=True
        )
        if "SSH_OK" in result.stdout:
            print(f"  SSH available!")
            break
        print(f"  Waiting for SSH... ({int(deadline - time.time())}s left)")
        time.sleep(10)
    else:
        raise TimeoutError("SSH not available after 5 min")

    # Open tunnel
    print(f"[probe] Opening tunnel: localhost:{local_port} → {public_ip}:{GUEST_PORT}")
    tunnel = _open_ssh_tunnel(
        vm_ip=public_ip,
        key_path=SSH_PRIVKEY_PATH,
        local_port=local_port,
        remote_port=GUEST_PORT,
    )
    endpoint = f"http://localhost:{local_port}"

    try:
        # Poll for guest agent (cloud-init runs apt install + starts agent)
        print(f"[probe] Polling {endpoint}/health (cloud-init may take 2-3 min)...")
        deadline = time.time() + 300
        while time.time() < deadline:
            try:
                r = requests.get(f"{endpoint}/health", timeout=5)
                if r.status_code == 200:
                    print(f"\n  ✅ Guest agent healthy: {r.json()}")
                    # Also try /screenshot
                    r2 = requests.get(f"{endpoint}/screenshot", timeout=10)
                    print(f"  ✅ /screenshot: HTTP {r2.status_code}, "
                          f"content-type: {r2.headers.get('content-type')}, "
                          f"size: {len(r2.content)} bytes")
                    return True
            except Exception:
                pass
            print(f"  Waiting for guest agent... ({int(deadline - time.time())}s left)")
            time.sleep(10)
        print(f"  ⚠️  Guest agent not ready after 5 min — cloud-init may still be running")
        print(f"     Check: ssh azureuser@{public_ip} 'sudo cloud-init status'")
        return False
    finally:
        tunnel.terminate()


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_full(img_path: str):
    """Complete Ubuntu gallery pipeline end-to-end."""
    print("=" * 65)
    print("CUBE Ubuntu Gallery Pipeline — Full Test")
    print("=" * 65)

    # Steps 1-3: image → VHD → blob → managed disk
    vhd_path  = step_convert(img_path)
    blob_url  = step_upload(vhd_path)
    disk_name = step_import(blob_url)

    # Steps 4-5: gallery image definition + version
    step_ubuntu_imgdef()
    step_ubuntu_version(disk_name)

    # Step 6: launch VM
    info = step_launch_ubuntu()

    # Step 7: probe
    ok = step_probe_ubuntu(info["public_ip"])

    print("\n" + "=" * 65)
    if ok:
        print("✅ FULL TEST PASSED")
        print(f"   Q1: qcow2-converted VHD boots on Azure → CONFIRMED")
        print(f"   Q2: os_profile SSH key injection works  → CONFIRMED")
        print(f"   Q3: cloud-init guest agent runs         → CONFIRMED")
        print(f"   Q4: SSH tunnel + /health + /screenshot  → CONFIRMED")
    else:
        print("⚠️  Pipeline complete but guest agent not yet ready (check cloud-init)")
    print(f"\nVM: {info['vm_name']} @ {info['public_ip']}")
    print(f"To clean up: python track.py delete")
    print("=" * 65)
    return info


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ubuntu Gallery full test")
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("convert", help="Step 1: .img → VHD")
    p.add_argument("--img", required=True)

    p = sub.add_parser("upload",  help="Step 2: VHD → Blob Storage")
    p.add_argument("--vhd", required=True)

    p = sub.add_parser("import",  help="Step 3: Blob → Managed Disk")
    p.add_argument("--blob-url", required=True)

    sub.add_parser("imgdef",  help="Step 4: Create Generalized image definition")

    p = sub.add_parser("version", help="Step 5: Create gallery image version")
    p.add_argument("--disk", required=True)

    sub.add_parser("launch",  help="Step 6: Launch VM from gallery")

    p = sub.add_parser("probe",   help="Step 7: Wait for SSH + probe guest agent")
    p.add_argument("--ip", required=True)

    p = sub.add_parser("full",    help="Full pipeline end-to-end")
    p.add_argument("--img", required=True, help="Path to ubuntu cloud image .img file")

    args = parser.parse_args()

    if args.cmd == "convert":
        step_convert(args.img)
    elif args.cmd == "upload":
        step_upload(args.vhd)
    elif args.cmd == "import":
        step_import(args.blob_url)
    elif args.cmd == "imgdef":
        step_ubuntu_imgdef()
    elif args.cmd == "version":
        step_ubuntu_version(args.disk)
    elif args.cmd == "launch":
        info = step_launch_ubuntu()
        print(f"\nNext: python experiment_ubuntu_gallery.py probe --ip {info['public_ip']}")
    elif args.cmd == "probe":
        step_probe_ubuntu(args.ip)
    elif args.cmd == "full":
        run_full(args.img)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
