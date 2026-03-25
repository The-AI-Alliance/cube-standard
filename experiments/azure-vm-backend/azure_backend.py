"""
AzureVMBackend experiment.

Explores the full pipeline:
  1. ensure_resource()  — upload qcow2 → Azure Managed Image (once per region)
  2. launch()           — spin up VM from image, wait for /screenshot
  3. restore_snapshot() — stop + relaunch from image (NEW_INSTANCE isolation)
  4. stop()             — deallocate + delete VM

Run stages independently:
  python azure_backend.py list-images
  python azure_backend.py launch --image <image-name>
  python azure_backend.py stop --vm <vm-name>
  python azure_backend.py ensure --qcow2 <path>
"""

import argparse
import subprocess
import sys
import time
import uuid
from pathlib import Path

import requests
from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.compute.models import (
    HardwareProfile,
    ImageReference,
    NetworkProfile,
    NetworkInterfaceReference,
    OSDisk,
    OSProfile,
    StorageProfile,
    VirtualMachine,
    VirtualMachineSizeTypes,
)
from azure.mgmt.network import NetworkManagementClient
from azure.mgmt.network.models import (
    NetworkInterface,
    NetworkInterfaceIPConfiguration,
    PublicIPAddress,
    SecurityRule,
)

from track import TAG, record

# ── Config ────────────────────────────────────────────────────────────────────
SUBSCRIPTION = "aeb958d3-a614-450e-94bc-88f284dc0664"
RESOURCE_GROUP = "ui_assist"
LOCATION = "westus2"
TAGS = {"project": "cube-experiment"}

GUEST_PORT = 5000
READY_TIMEOUT = 300  # seconds
READY_POLL = 5       # seconds

# ── Azure clients ─────────────────────────────────────────────────────────────

def _credential():
    return DefaultAzureCredential()

def _compute():
    return ComputeManagementClient(_credential(), SUBSCRIPTION)

def _network():
    return NetworkManagementClient(_credential(), SUBSCRIPTION)


# ── Image listing ─────────────────────────────────────────────────────────────

def list_images():
    """List all managed images in the resource group."""
    compute = _compute()
    images = list(compute.images.list_by_resource_group(RESOURCE_GROUP))
    if not images:
        print("No managed images found.")
        return
    print(f"\n{'Name':<45} {'Location':<12} {'OS'}")
    print("-" * 70)
    for img in images:
        os_type = img.storage_profile.os_disk.os_type if img.storage_profile else "?"
        print(f"{img.name:<45} {img.location:<12} {os_type}")


# ── VM launch ─────────────────────────────────────────────────────────────────

def _create_public_ip(network: NetworkManagementClient, name: str) -> PublicIPAddress:
    print(f"  Creating public IP: {name}")
    poller = network.public_ip_addresses.begin_create_or_update(
        RESOURCE_GROUP, name,
        {"location": LOCATION, "tags": TAGS,
         "sku": {"name": "Standard"},
         "properties": {"publicIPAllocationMethod": "Static"}},
    )
    ip = poller.result()
    record(name, "Microsoft.Network/publicIPAddresses", ip.id)
    return ip


def _create_nic(network: NetworkManagementClient, name: str, pip_id: str, vnet: str, subnet: str) -> NetworkInterface:
    print(f"  Creating NIC: {name}")
    subnet_id = (
        f"/subscriptions/{SUBSCRIPTION}/resourceGroups/{RESOURCE_GROUP}"
        f"/providers/Microsoft.Network/virtualNetworks/{vnet}/subnets/{subnet}"
    )
    poller = network.network_interfaces.begin_create_or_update(
        RESOURCE_GROUP, name,
        {
            "location": LOCATION,
            "tags": TAGS,
            "properties": {
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
    return nic


def launch_vm(image_name: str, vm_size: str = "Standard_D4s_v3") -> dict:
    """
    Launch a VM from a managed image.
    Returns {"vm_name": ..., "public_ip": ..., "endpoint": ...}
    """
    uid = uuid.uuid4().hex[:6]
    vm_name = f"cube-exp-{uid}"
    pip_name = f"{vm_name}-ip"
    nic_name = f"{vm_name}-nic"

    compute = _compute()
    network = _network()

    # Find image
    image = compute.images.get(RESOURCE_GROUP, image_name)
    print(f"Launching VM from image: {image_name} ({image.location})")

    # Create public IP + NIC
    # Reuse existing VNet/subnet from the osworld setup
    pip = _create_public_ip(network, pip_name)
    nic = _create_nic(network, nic_name, pip.id, "vnet-westus2", "default")

    print(f"  Creating VM: {vm_name} ({vm_size})")
    vm_params = {
        "location": LOCATION,
        "tags": TAGS,
        "hardware_profile": {"vm_size": vm_size},
        "storage_profile": {
            "image_reference": {"id": image.id},
            "os_disk": {
                "create_option": "FromImage",
                "delete_option": "Delete",  # auto-delete disk when VM deleted
            },
        },
        "os_profile": {
            "computer_name": vm_name,
            "admin_username": "azureuser",
            "linux_configuration": {
                "disable_password_authentication": True,
                "ssh": {
                    "public_keys": [{
                        "path": "/home/azureuser/.ssh/authorized_keys",
                        # Reuse existing key from resource group
                        "key_data": _get_ssh_pubkey("os_world_key"),
                    }]
                },
            },
        },
        "network_profile": {
            "network_interfaces": [{"id": nic.id, "primary": True}]
        },
    }

    poller = compute.virtual_machines.begin_create_or_update(
        RESOURCE_GROUP, vm_name, vm_params
    )
    vm = poller.result()
    record(vm_name, "Microsoft.Compute/virtualMachines", vm.id)

    # Get public IP
    pip_refreshed = network.public_ip_addresses.get(RESOURCE_GROUP, pip_name)
    public_ip = pip_refreshed.ip_address
    endpoint = f"http://{public_ip}:{GUEST_PORT}"

    print(f"  VM created: {vm_name} @ {public_ip}")
    print(f"  Waiting for guest agent at {endpoint} ...")

    _wait_for_endpoint(endpoint)
    print(f"  Ready! endpoint={endpoint}")

    return {"vm_name": vm_name, "public_ip": public_ip, "endpoint": endpoint}


def _get_ssh_pubkey(key_name: str) -> str:
    result = subprocess.run(
        ["az", "sshkey", "show", "-g", RESOURCE_GROUP, "-n", key_name,
         "--query", "publicKey", "-o", "tsv"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def _wait_for_endpoint(endpoint: str, timeout: int = READY_TIMEOUT):
    deadline = time.time() + timeout
    url = f"{endpoint}/screenshot"
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(READY_POLL)
    raise TimeoutError(f"VM guest agent not ready after {timeout}s at {endpoint}")


# ── VM stop ───────────────────────────────────────────────────────────────────

def stop_vm(vm_name: str):
    """Delete VM (disk auto-deleted via delete_option=Delete)."""
    compute = _compute()
    print(f"Deleting VM: {vm_name}")
    poller = compute.virtual_machines.begin_delete(RESOURCE_GROUP, vm_name)
    poller.result()
    print(f"  VM deleted: {vm_name}")

    # Clean up NIC and IP
    network = _network()
    nic_name = f"{vm_name}-nic"
    pip_name = f"{vm_name}-ip"
    try:
        network.network_interfaces.begin_delete(RESOURCE_GROUP, nic_name).result()
        print(f"  NIC deleted: {nic_name}")
    except Exception as e:
        print(f"  NIC delete skipped: {e}")
    try:
        network.public_ip_addresses.begin_delete(RESOURCE_GROUP, pip_name).result()
        print(f"  IP deleted: {pip_name}")
    except Exception as e:
        print(f"  IP delete skipped: {e}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Azure VM backend experiments")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("list-images", help="List managed images in resource group")

    p_launch = sub.add_parser("launch", help="Launch VM from managed image")
    p_launch.add_argument("--image", required=True)
    p_launch.add_argument("--size", default="Standard_D4s_v3")

    p_stop = sub.add_parser("stop", help="Delete a VM and its resources")
    p_stop.add_argument("--vm", required=True)

    args = parser.parse_args()

    if args.cmd == "list-images":
        list_images()
    elif args.cmd == "launch":
        info = launch_vm(args.image, args.size)
        print(f"\nVM ready: {info}")
    elif args.cmd == "stop":
        stop_vm(args.vm)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
