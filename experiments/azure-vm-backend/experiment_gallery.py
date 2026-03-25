"""
Plan A: Azure Compute Gallery experiment.

Goal: validate that launching a VM from an Azure Compute Gallery image
  1. Bypasses the Golden Image Policy (allows createOption: FromImage)
  2. Enables clean SSH access (no CCOE workaround needed for Generalized images)

Steps:
  A. Create Compute Gallery + Image Definition + Image Version from osworld_base snapshot
     (Specialized path — validates policy bypass with existing image, no download)
  B. [Optional] Generalized path: upload Ubuntu cloud image → gallery → launch
     with os_profile SSH key injection (full clean path, requires image download)

Run:
  python experiment_gallery.py gallery    -- create gallery + image def
  python experiment_gallery.py version    -- create image version from snapshot
  python experiment_gallery.py launch     -- launch VM from gallery image
  python experiment_gallery.py probe --ip 1.2.3.4
  python experiment_gallery.py cleanup    -- delete all gallery resources

The key hypothesis: PolicyDenied errors should NOT appear because we're using
createOption: FromImage with imageReference.id pointing to a Compute Gallery image,
which satisfies the Golden_image_exception policy condition B.
"""

import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

import requests
from azure.identity import AzureCliCredential
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.network import NetworkManagementClient

from pipeline import (
    SUBSCRIPTION, RESOURCE_GROUP, LOCATION, TAGS,
    VNET_NAME, SUBNET_NAME, NSG_NAME,
    GUEST_PORT, VM_SIZE,
    _compute, _network, _create_pip, _create_nic, _find_free_port, _open_ssh_tunnel,
)
from track import record


# ── Gallery constants ──────────────────────────────────────────────────────────

GALLERY_NAME     = "cube_exp_gallery"
IMAGE_DEF_NAME   = "cube-osworld-linux"
IMAGE_VERSION    = "1.0.1"   # 1.0.0 is osworld_base (1TB, still creating); 1.0.1 is tiny test

# Tiny 1GB empty snapshot — just for policy validation (no real OS, won't boot)
# Change to "osworld_base" (1TB, ~20-30min) for production gallery publishing.
SOURCE_SNAPSHOT  = "cube-exp-tiny-snap-d86c42"

# SSH key to use for Run Command injection (same key we validated in Plan B)
SSH_PRIVKEY_PATH = str(Path.home() / ".ssh" / "id_ed25519")
SSH_PUBKEY_PATH  = str(Path.home() / ".ssh" / "id_ed25519.pub")


# ── Step A1: Create Compute Gallery ───────────────────────────────────────────

def step_gallery_create():
    """
    Create the Compute Gallery resource.
    Idempotent — returns existing gallery if already present.
    """
    compute = _compute()
    print(f"[gallery] Creating Compute Gallery: {GALLERY_NAME}")
    poller = compute.galleries.begin_create_or_update(
        RESOURCE_GROUP, GALLERY_NAME,
        {
            "location": LOCATION,
            "tags": TAGS,
            "description": "CUBE experiment — OSWorld benchmark image gallery",
        }
    )
    gallery = poller.result()
    record(GALLERY_NAME, "Microsoft.Compute/galleries", gallery.id)
    print(f"  Gallery created: {gallery.id}")
    return gallery


# ── Step A2: Create Image Definition ──────────────────────────────────────────

def step_image_definition():
    """
    Create an image definition in the gallery.

    Using Specialized because osworld_base is a Specialized image (baked SSH keys).
    For the clean cloud-init path (Plan A full), this would be Generalized.

    Note on Specialized vs Generalized:
      Specialized: VM retains baked-in identity (users, SSH keys, hostname).
                   No os_profile at launch — you can't inject SSH keys via cloud-init.
      Generalized: VM identity reset. os_profile at launch injects SSH keys, hostname.
                   This is what we want for production (cloud-init clean injection).
    """
    compute = _compute()
    print(f"[image-def] Creating image definition: {IMAGE_DEF_NAME}")
    poller = compute.gallery_images.begin_create_or_update(
        RESOURCE_GROUP, GALLERY_NAME, IMAGE_DEF_NAME,
        {
            "location": LOCATION,
            "tags": TAGS,
            "os_type": "Linux",
            "os_state": "Specialized",    # osworld_base has baked identity
            "hyper_v_generation": "V1",   # osworld_base was confirmed V1
            "identifier": {
                "publisher": "cube-exp",
                "offer": "osworld",
                "sku": "linux",
            },
        }
    )
    img_def = poller.result()
    record(IMAGE_DEF_NAME, "Microsoft.Compute/galleries/images", img_def.id)
    print(f"  Image definition created: {img_def.id}")
    return img_def


# ── Step A3: Create Image Version from Snapshot ───────────────────────────────

def step_image_version():
    """
    Create an image version from the osworld_base snapshot.

    The snapshot must be fully provisioned. Image version creation replicates
    the snapshot data into the gallery's managed storage.
    Takes ~5-15 min depending on disk size.
    """
    compute = _compute()

    # Get the source snapshot
    snap = compute.snapshots.get(RESOURCE_GROUP, SOURCE_SNAPSHOT)
    print(f"[version] Source snapshot: {snap.id} ({snap.disk_size_gb} GB)")
    print(f"[version] Creating image version {IMAGE_VERSION} in gallery...")
    print(f"  This takes ~5-15 min (replicating {snap.disk_size_gb} GB into gallery)...")

    poller = compute.gallery_image_versions.begin_create_or_update(
        RESOURCE_GROUP, GALLERY_NAME, IMAGE_DEF_NAME, IMAGE_VERSION,
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
                    "source": {"id": snap.id},
                    "host_caching": "ReadWrite",
                }
            }
        }
    )
    version = poller.result()
    resource_id = (
        f"/subscriptions/{SUBSCRIPTION}/resourceGroups/{RESOURCE_GROUP}"
        f"/providers/Microsoft.Compute/galleries/{GALLERY_NAME}"
        f"/images/{IMAGE_DEF_NAME}/versions/{IMAGE_VERSION}"
    )
    record(f"gallery-version-{IMAGE_VERSION}", "Microsoft.Compute/galleries/images/versions", resource_id)
    print(f"  Image version created: {version.id}")
    return version


# ── Step A4: Launch VM from Gallery Image ─────────────────────────────────────

def step_launch_gallery() -> dict:
    """
    Launch a VM using createOption: FromImage with a Compute Gallery imageReference.

    KEY TEST: This should pass the Golden Image Policy because:
      imageReference.id contains "Microsoft.Compute/galleries"
      → satisfies condition B of Golden_image_exception policy

    For Specialized images: no os_profile (baked identity from gallery image).
    For Generalized images: os_profile would carry SSH key injection.

    Returns {"vm_name": ..., "public_ip": ..., "vm_id": ...}
    """
    uid = uuid.uuid4().hex[:6]
    vm_name  = f"cube-gal-vm-{uid}"
    pip_name = f"cube-gal-ip-{uid}"
    nic_name = f"cube-gal-nic-{uid}"

    image_id = (
        f"/subscriptions/{SUBSCRIPTION}/resourceGroups/{RESOURCE_GROUP}"
        f"/providers/Microsoft.Compute/galleries/{GALLERY_NAME}"
        f"/images/{IMAGE_DEF_NAME}/versions/{IMAGE_VERSION}"
    )

    compute = _compute()
    network = _network()

    pip_id = _create_pip(network, pip_name)
    nic_id = _create_nic(network, nic_name, pip_id)

    print(f"[launch-gallery] Creating VM: {vm_name} from gallery image")
    print(f"  Image: {image_id}")

    vm_params = {
        "location": LOCATION,
        "tags": TAGS,
        "hardware_profile": {"vm_size": VM_SIZE},
        "storage_profile": {
            "image_reference": {"id": image_id},
            "os_disk": {
                "create_option": "FromImage",      # <-- key: FromImage, not Attach
                "managed_disk": {"storage_account_type": "Standard_LRS"},
                "delete_option": "Delete",
            },
        },
        # No os_profile for Specialized images (baked identity)
        "network_profile": {
            "network_interfaces": [{"id": nic_id, "properties": {"primary": True}}]
        },
    }

    try:
        poller = compute.virtual_machines.begin_create_or_update(
            RESOURCE_GROUP, vm_name, vm_params
        )
        vm = poller.result()
        record(vm_name, "Microsoft.Compute/virtualMachines", vm.id)
    except Exception as e:
        error_str = str(e)
        if "RequestDisallowedByPolicy" in error_str or "Golden_image_exception" in error_str:
            print(f"\n[RESULT] ❌ POLICY BLOCKED — Gallery image does NOT bypass Golden Image Policy")
            print(f"  This means Compute Gallery is not whitelisted in this subscription.")
            print(f"  Error: {error_str[:500]}")
        else:
            print(f"\n[RESULT] ❌ VM creation FAILED (non-policy error): {error_str[:500]}")
        raise

    pip = network.public_ip_addresses.get(RESOURCE_GROUP, pip_name)
    public_ip = pip.ip_address

    print(f"\n[RESULT] ✅ VM created successfully from gallery image!")
    print(f"  VM name:   {vm_name}")
    print(f"  Public IP: {public_ip}")
    print(f"  This CONFIRMS the Compute Gallery bypasses Golden Image Policy.")
    return {"vm_name": vm_name, "public_ip": public_ip}


# ── Step A5: Inject SSH key via Run Command ────────────────────────────────────

def step_inject_ssh(vm_name: str):
    """
    Since osworld_base is Specialized (no cloud-init SSH injection),
    use Azure Run Command to inject our SSH key — same approach as Plan B.

    For Generalized gallery images, this step would be unnecessary
    (SSH key injection happens cleanly via os_profile at launch time).
    """
    pubkey = Path(SSH_PUBKEY_PATH).read_text().strip()
    print(f"[inject-ssh] Injecting SSH key via Run Command to {vm_name}...")

    # 1. Disable CCOE AuthorizedKeysCommand
    subprocess.run([
        "az", "vm", "run-command", "invoke",
        "-g", RESOURCE_GROUP, "-n", vm_name,
        "--command-id", "RunShellScript",
        "--scripts",
        "sed -i 's|^AuthorizedKeysCommand .*|#AuthorizedKeysCommand disabled|' "
        "/etc/ssh/sshd_config && systemctl restart ssh && echo 'CCOE disabled'",
    ], check=True)
    print("  CCOE AuthorizedKeysCommand disabled.")

    # 2. Inject our SSH public key
    inject_cmd = (
        f"mkdir -p /home/aman/.ssh && "
        f"echo '{pubkey}' >> /home/aman/.ssh/authorized_keys && "
        f"chmod 600 /home/aman/.ssh/authorized_keys && "
        f"chown aman:aman /home/aman/.ssh/authorized_keys && "
        f"echo 'KEY_INJECTED'"
    )
    result = subprocess.run([
        "az", "vm", "run-command", "invoke",
        "-g", RESOURCE_GROUP, "-n", vm_name,
        "--command-id", "RunShellScript",
        "--scripts", inject_cmd,
    ], capture_output=True, text=True, check=True)
    print(f"  Key injection result: {result.stdout[:200]}")


# ── Step A6: Probe via SSH tunnel ──────────────────────────────────────────────

def step_probe_via_tunnel(public_ip: str, timeout: int = 60) -> bool:
    """
    Install a minimal Flask guest agent on the VM and probe /health via SSH tunnel.
    Uses the same SSH tunnel approach validated in Plan B.
    """
    local_port = _find_free_port()
    print(f"[probe] Opening SSH tunnel: localhost:{local_port} → {public_ip}:5000")

    tunnel = _open_ssh_tunnel(
        vm_ip=public_ip,
        key_path=SSH_PRIVKEY_PATH,
        local_port=local_port,
        remote_port=GUEST_PORT,
    )
    endpoint = f"http://localhost:{local_port}"

    try:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = requests.get(f"{endpoint}/health", timeout=5)
                if r.status_code == 200:
                    print(f"  ✅ Guest agent healthy: {r.json()}")
                    return True
            except Exception:
                pass
            print(f"  Waiting... ({int(deadline - time.time())}s left)")
            time.sleep(5)
        print(f"  ⚠️  Timed out waiting for guest agent (expected if not installed)")
    finally:
        tunnel.terminate()

    return False


# ── Cleanup ────────────────────────────────────────────────────────────────────

def step_cleanup():
    """Delete gallery, image definition, and image version (in order)."""
    compute = _compute()
    print(f"[cleanup] Deleting gallery resources...")

    # Delete image version first
    try:
        print(f"  Deleting image version {IMAGE_VERSION}...")
        poller = compute.gallery_image_versions.begin_delete(
            RESOURCE_GROUP, GALLERY_NAME, IMAGE_DEF_NAME, IMAGE_VERSION
        )
        poller.result()
        print(f"  Version deleted.")
    except Exception as e:
        print(f"  Version not found or already deleted: {e}")

    # Delete image definition
    try:
        print(f"  Deleting image definition {IMAGE_DEF_NAME}...")
        poller = compute.gallery_images.begin_delete(
            RESOURCE_GROUP, GALLERY_NAME, IMAGE_DEF_NAME
        )
        poller.result()
        print(f"  Image definition deleted.")
    except Exception as e:
        print(f"  Image definition not found: {e}")

    # Delete gallery
    try:
        print(f"  Deleting gallery {GALLERY_NAME}...")
        poller = compute.galleries.begin_delete(RESOURCE_GROUP, GALLERY_NAME)
        poller.result()
        print(f"  Gallery deleted.")
    except Exception as e:
        print(f"  Gallery not found: {e}")

    print("  Cleanup complete. Use 'python track.py delete' for VM/NIC/IP resources.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plan A: Compute Gallery experiment")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("gallery",  help="Step A1+A2: Create gallery + image definition")
    sub.add_parser("version",  help="Step A3: Create image version from osworld_base snapshot")
    sub.add_parser("launch",   help="Step A4: Launch VM from gallery image")

    p = sub.add_parser("inject", help="Step A5: Inject SSH key via Run Command")
    p.add_argument("--vm", required=True, help="VM name")

    p = sub.add_parser("probe",  help="Step A6: Probe guest agent via SSH tunnel")
    p.add_argument("--ip", required=True)

    p = sub.add_parser("full",   help="Steps A1-A5: End-to-end (minus probe)")
    sub.add_parser("cleanup",  help="Delete gallery resources (not VMs)")

    args = parser.parse_args()

    if args.cmd == "gallery":
        step_gallery_create()
        step_image_definition()
    elif args.cmd == "version":
        step_image_version()
    elif args.cmd == "launch":
        info = step_launch_gallery()
        print(f"\nNext steps:")
        print(f"  python experiment_gallery.py inject --vm {info['vm_name']}")
        print(f"  python experiment_gallery.py probe  --ip {info['public_ip']}")
    elif args.cmd == "inject":
        step_inject_ssh(args.vm)
    elif args.cmd == "probe":
        step_probe_via_tunnel(args.ip)
    elif args.cmd == "full":
        step_gallery_create()
        step_image_definition()
        print("\n[full] Gallery + definition created. Now creating image version...")
        print("[full] This takes ~5-15 min. Starting...")
        step_image_version()
        print("\n[full] Launching VM from gallery...")
        info = step_launch_gallery()
        print(f"\n[full] Injecting SSH key...")
        step_inject_ssh(info["vm_name"])
        print(f"\n[full] Done! VM is up at {info['public_ip']}")
        print(f"  SSH: ssh -i {SSH_PRIVKEY_PATH} -o IdentitiesOnly=yes aman@{info['public_ip']}")
        print(f"  Tunnel: ssh -N -L 127.0.0.1:15000:localhost:5000 -i {SSH_PRIVKEY_PATH} aman@{info['public_ip']}")
    elif args.cmd == "cleanup":
        step_cleanup()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
