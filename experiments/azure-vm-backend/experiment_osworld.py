"""
OSWorld real image pipeline.

Source: xlangai/ubuntu_osworld on HuggingFace (Ubuntu-x86.zip, VMware VMDK)
Converts: VMDK → fixed VHD → Azure Blob → Managed Disk → Gallery → VM

Key difference from Ubuntu 22.04 test:
  - Source is VMDK not qcow2 (use -f vmdk)
  - Image is likely Specialized (VMware-based, may not have cloud-init)
  - Access: try os_profile first; fall back to Run Command for SSH key injection
  - Goal: /screenshot should return actual Ubuntu desktop (GNOME/XFCE), not black rectangle
"""

import base64
import subprocess
import time
import uuid
from pathlib import Path

import requests

from pipeline import (
    SUBSCRIPTION, RESOURCE_GROUP, LOCATION, TAGS, VM_SIZE, GUEST_PORT,
    _compute, _network, _create_pip, _create_nic, _find_free_port, _open_ssh_tunnel,
    step_upload, step_import,
)
from experiment_ubuntu_gallery import (
    GALLERY_NAME, SSH_PRIVKEY_PATH, SSH_PUBKEY_PATH,
    step_probe_ubuntu,
)
from track import record

OSWORLD_IMAGE_DEF   = "cube-osworld-ubuntu-x86"
OSWORLD_IMG_VERSION = "1.0.0"

# Same cloud-init script as Ubuntu test (fixed YAML version)
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


def step_convert_vmdk(vmdk_path: str) -> str:
    """Convert VMDK (multi-extent) to fixed VHD. Returns VHD path."""
    import subprocess
    src = Path(vmdk_path).resolve()
    dst = src.parent / "osworld.vhd"
    if dst.exists():
        print(f"[convert] VHD already exists: {dst} ({dst.stat().st_size/1024**3:.1f} GB), skipping.")
        return str(dst)
    print(f"[convert] VMDK → VHD (50 GiB fixed, may take ~2 min)...")
    t0 = time.time()
    subprocess.run([
        "qemu-img", "convert", "-f", "vmdk", "-O", "vpc",
        "-o", "subformat=fixed,force_size",
        str(src), str(dst),
    ], check=True)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s → {dst} ({dst.stat().st_size/1024**3:.1f} GB)")
    return str(dst)


def step_osworld_imgdef(os_state: str = "Generalized"):
    """Create image definition for OSWorld Ubuntu image."""
    compute = _compute()
    print(f"[imgdef] Creating {os_state} image definition: {OSWORLD_IMAGE_DEF}")
    poller = compute.gallery_images.begin_create_or_update(
        RESOURCE_GROUP, GALLERY_NAME, OSWORLD_IMAGE_DEF,
        {
            "location": LOCATION,
            "tags": TAGS,
            "os_type": "Linux",
            "os_state": os_state,
            "hyper_v_generation": "V1",
            "identifier": {
                "publisher": "cube-exp",
                "offer": "osworld",
                "sku": "ubuntu-x86",
            },
        }
    )
    img_def = poller.result()
    record(OSWORLD_IMAGE_DEF, "Microsoft.Compute/galleries/images", img_def.id)
    print(f"  Created: {img_def.id}")
    return img_def


def step_osworld_version(disk_name: str):
    """Create gallery image version from OSWorld managed disk."""
    compute = _compute()
    disk = compute.disks.get(RESOURCE_GROUP, disk_name)
    print(f"[version] Creating gallery version {OSWORLD_IMG_VERSION} from {disk_name} ({disk.disk_size_gb} GB)...")
    t0 = time.time()
    poller = compute.gallery_image_versions.begin_create_or_update(
        RESOURCE_GROUP, GALLERY_NAME, OSWORLD_IMAGE_DEF, OSWORLD_IMG_VERSION,
        {
            "location": LOCATION,
            "tags": TAGS,
            "publishing_profile": {
                "replica_count": 1,
                "storage_account_type": "Standard_LRS",
                "target_regions": [{"name": LOCATION, "regional_replica_count": 1,
                                    "storage_account_type": "Standard_LRS"}],
                "exclude_from_latest": False,
            },
            "storage_profile": {
                "os_disk_image": {"source": {"id": disk.id}, "host_caching": "ReadWrite"}
            }
        }
    )
    version = poller.result()
    elapsed = time.time() - t0
    record(f"osworld-version-{OSWORLD_IMG_VERSION}", "Microsoft.Compute/galleries/images/versions",
           f"/subscriptions/{SUBSCRIPTION}/resourceGroups/{RESOURCE_GROUP}/providers/"
           f"Microsoft.Compute/galleries/{GALLERY_NAME}/images/{OSWORLD_IMAGE_DEF}/versions/{OSWORLD_IMG_VERSION}")
    print(f"  Version created in {elapsed:.0f}s: {version.id}")
    return version


def step_launch_osworld(generalized: bool = True) -> dict:
    """Launch OSWorld VM from gallery. Try Generalized (cloud-init) first."""
    import base64
    uid = uuid.uuid4().hex[:6]
    vm_name  = f"cube-osw-vm-{uid}"
    pip_name = f"cube-osw-ip-{uid}"
    nic_name = f"cube-osw-nic-{uid}"

    image_id = (
        f"/subscriptions/{SUBSCRIPTION}/resourceGroups/{RESOURCE_GROUP}"
        f"/providers/Microsoft.Compute/galleries/{GALLERY_NAME}"
        f"/images/{OSWORLD_IMAGE_DEF}/versions/{OSWORLD_IMG_VERSION}"
    )
    pubkey = Path(SSH_PUBKEY_PATH).read_text().strip()
    custom_data_b64 = base64.b64encode(CLOUD_INIT_SCRIPT.encode()).decode()

    compute = _compute()
    network = _network()
    pip_id = _create_pip(network, pip_name)
    nic_id = _create_nic(network, nic_name, pip_id)

    print(f"[launch] VM: {vm_name}  image: {OSWORLD_IMAGE_DEF}/{OSWORLD_IMG_VERSION}")
    t0 = time.time()

    vm_params = {
        "location": LOCATION,
        "tags": TAGS,
        "hardware_profile": {"vm_size": VM_SIZE},
        "storage_profile": {
            "image_reference": {"id": image_id},
            "os_disk": {
                "create_option": "FromImage",
                "managed_disk": {"storage_account_type": "Standard_LRS"},
                "disk_size_gb": 64,
                "delete_option": "Delete",
            },
        },
        "network_profile": {
            "network_interfaces": [{"id": nic_id, "properties": {"primary": True}}]
        },
    }

    if generalized:
        vm_params["os_profile"] = {
            "computer_name": vm_name,
            "admin_username": "azureuser",
            "custom_data": custom_data_b64,
            "linux_configuration": {
                "disable_password_authentication": True,
                "ssh": {"public_keys": [{"path": "/home/azureuser/.ssh/authorized_keys",
                                          "key_data": pubkey}]}
            }
        }

    try:
        poller = compute.virtual_machines.begin_create_or_update(RESOURCE_GROUP, vm_name, vm_params)
        vm = poller.result()
    except Exception as e:
        if "OSProvisioningInternalError" in str(e) or "GeneralizationError" in str(e):
            print(f"  Generalized launch failed (image is Specialized). Retrying without os_profile...")
            vm_params.pop("os_profile", None)
            vm_params["storage_profile"]["os_disk"]["create_option"] = "Attach"
            # For Specialized, need to pre-create disk from snapshot
            raise NotImplementedError("Specialized path: use createOption=Attach with pre-created disk")
        raise

    record(vm_name, "Microsoft.Compute/virtualMachines", vm.id)
    pip = network.public_ip_addresses.get(RESOURCE_GROUP, pip_name)
    elapsed = time.time() - t0
    print(f"  VM ready in {elapsed:.0f}s: {vm_name} @ {pip.ip_address}")
    return {"vm_name": vm_name, "public_ip": pip.ip_address}


def run_osworld_full(vhd_path: str):
    """Full OSWorld pipeline from pre-converted VHD."""
    timings = {}
    print("=" * 65)
    print("OSWorld Real Image Pipeline")
    print("=" * 65)

    t0 = time.time()
    blob_url = step_upload(vhd_path)
    timings["upload"] = time.time() - t0

    t0 = time.time()
    disk_name = step_import(blob_url)
    timings["import"] = time.time() - t0

    t0 = time.time()
    step_osworld_imgdef("Generalized")
    step_osworld_version(disk_name)
    timings["gallery"] = time.time() - t0

    t0 = time.time()
    info = step_launch_osworld(generalized=True)
    timings["launch"] = time.time() - t0

    print("\n--- Timings ---")
    for step, secs in timings.items():
        print(f"  {step:10s}: {secs/60:.1f} min")
    print(f"  {'total':10s}: {sum(timings.values())/60:.1f} min")

    print(f"\nNext: python experiment_osworld.py probe --ip {info['public_ip']}")
    return info


def main():
    import argparse
    parser = argparse.ArgumentParser(description="OSWorld real image pipeline")
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("convert", help="VMDK → VHD")
    p.add_argument("--vmdk", required=True)

    p = sub.add_parser("upload", help="VHD → Blob Storage")
    p.add_argument("--vhd", required=True)

    p = sub.add_parser("import", help="Blob → Managed Disk")
    p.add_argument("--blob-url", required=True)

    sub.add_parser("imgdef", help="Create gallery image definition")

    p = sub.add_parser("version", help="Create gallery image version")
    p.add_argument("--disk", required=True)

    sub.add_parser("launch", help="Launch VM from gallery")

    p = sub.add_parser("probe", help="Probe guest agent via SSH tunnel")
    p.add_argument("--ip", required=True)

    p = sub.add_parser("full", help="Upload→import→gallery→launch (VHD already converted)")
    p.add_argument("--vhd", required=True)

    args = parser.parse_args()

    if args.cmd == "convert":
        step_convert_vmdk(args.vmdk)
    elif args.cmd == "upload":
        step_upload(args.vhd)
    elif args.cmd == "import":
        step_import(args.blob_url)
    elif args.cmd == "imgdef":
        step_osworld_imgdef()
    elif args.cmd == "version":
        step_osworld_version(args.disk)
    elif args.cmd == "launch":
        info = step_launch_osworld()
        print(f"\nNext: python experiment_osworld.py probe --ip {info['public_ip']}")
    elif args.cmd == "probe":
        step_probe_ubuntu(args.ip)
    elif args.cmd == "full":
        run_osworld_full(args.vhd)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
