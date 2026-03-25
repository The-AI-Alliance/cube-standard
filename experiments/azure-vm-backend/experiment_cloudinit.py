"""
Experiment: Cloud-init guest agent injection on a fresh Azure VM.

Tests the key architectural question:
  Can CUBE inject its guest agent at VM launch time (via cloud-init)
  WITHOUT modifying the benchmark's original image?

Pipeline:
  1. Generate throwaway SSH keypair (per-VM, no pre-existing key needed)
  2. Find Ubuntu 22.04 image in Azure Marketplace
  3. Launch VM with cloud-init that installs + starts a fake guest agent
  4. SSH tunnel to port 5000 (bypasses Zscaler)
  5. Hit /screenshot and verify response

What this validates:
  - cloud-init injection works for any Generalized Linux image
  - SSH key injection works (no pre-baked keys needed)
  - SSH tunnel approach resolves the Zscaler/port-blocking problem
  - The overall UX: user provides cloud creds, CUBE handles everything else

Run: python experiment_cloudinit.py
"""

import base64
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import requests
from azure.identity import AzureCliCredential
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.network import NetworkManagementClient

from track import record
from pipeline import SUBSCRIPTION, RESOURCE_GROUP, LOCATION, TAGS, VNET_NAME, SUBNET_NAME, NSG_NAME

VM_SIZE = "Standard_B2s"   # cheapest, 2 vCPU 4 GB — no KVM needed for this test
GUEST_PORT = 5000

# ── Cloud-init payload ────────────────────────────────────────────────────────
# This is what CUBE would inject into any Generalized Linux benchmark image.
# Installs a minimal Flask server that mimics the CUBE guest agent.
CLOUD_INIT = """#cloud-config
packages:
  - python3-pip
  - python3-flask
  - python3-pillow

write_files:
  - path: /opt/cube-guest-agent/server.py
    content: |
      import io, struct, zlib
      from flask import Flask, send_file, jsonify
      app = Flask(__name__)

      def make_png(w=100, h=100):
          # Minimal 100x100 grey PNG
          raw = b'\\x00' + bytes([128] * w * 3)  # one row
          rows = raw * h
          def chunk(tag, data):
              c = zlib.crc32(tag + data) & 0xffffffff
              return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', c)
          ihdr = struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)
          idat = zlib.compress(rows)
          return (b'\\x89PNG\\r\\n\\x1a\\n' +
                  chunk(b'IHDR', ihdr) +
                  chunk(b'IDAT', idat) +
                  chunk(b'IEND', b''))

      @app.route('/screenshot')
      def screenshot():
          return send_file(io.BytesIO(make_png()), mimetype='image/png')

      @app.route('/health')
      def health():
          return jsonify({"status": "ok", "agent": "cube-fake-agent"})

      @app.route('/execute', methods=['POST'])
      def execute():
          return jsonify({"stdout": "", "stderr": "", "returncode": 0})

      if __name__ == '__main__':
          app.run(host='0.0.0.0', port=5000)

runcmd:
  - python3 /opt/cube-guest-agent/server.py &
  - sleep 2
  - curl -s http://localhost:5000/health
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def _cred():
    return AzureCliCredential()

def _compute():
    return ComputeManagementClient(_cred(), SUBSCRIPTION)

def _network():
    return NetworkManagementClient(_cred(), SUBSCRIPTION)


def generate_ssh_keypair(tmpdir: Path) -> tuple[str, str]:
    """Generate a throwaway ED25519 keypair. Returns (private_path, public_key_str)."""
    key_path = tmpdir / "cube_exp_key"
    subprocess.run(
        ["ssh-keygen", "-t", "ed25519", "-N", "", "-f", str(key_path)],
        check=True, capture_output=True,
    )
    pub_key = (key_path.with_suffix(".pub")).read_text().strip()
    return str(key_path), pub_key


def find_ubuntu_image(compute: ComputeManagementClient) -> dict:
    """Find latest Ubuntu 22.04 LTS image in Azure Marketplace."""
    images = list(compute.virtual_machine_images.list(
        LOCATION, "Canonical", "0001-com-ubuntu-server-jammy", "22_04-lts",
        top=5,
    ))
    if not images:
        # Fallback to older offer name
        images = list(compute.virtual_machine_images.list(
            LOCATION, "Canonical", "UbuntuServer", "22.04-LTS",
            top=5,
        ))
    latest = sorted(images, key=lambda x: x.name)[-1]
    print(f"  Found Ubuntu image: {latest.name}")
    return {
        "publisher": "Canonical",
        "offer": "0001-com-ubuntu-server-jammy",
        "sku": "22_04-lts",
        "version": "latest",
    }


def create_pip(network, name) -> str:
    print(f"  Creating public IP: {name}")
    p = network.public_ip_addresses.begin_create_or_update(
        RESOURCE_GROUP, name,
        {"location": LOCATION, "tags": TAGS,
         "sku": {"name": "Standard"},
         "properties": {"publicIPAllocationMethod": "Static"}},
    ).result()
    record(name, "Microsoft.Network/publicIPAddresses", p.id)
    return p.id


def create_nic(network, name, pip_id) -> str:
    print(f"  Creating NIC: {name}")
    subnet_id = (
        f"/subscriptions/{SUBSCRIPTION}/resourceGroups/{RESOURCE_GROUP}"
        f"/providers/Microsoft.Network/virtualNetworks/{VNET_NAME}/subnets/{SUBNET_NAME}"
    )
    nsg_id = (
        f"/subscriptions/{SUBSCRIPTION}/resourceGroups/{RESOURCE_GROUP}"
        f"/providers/Microsoft.Network/networkSecurityGroups/{NSG_NAME}"
    )
    n = network.network_interfaces.begin_create_or_update(
        RESOURCE_GROUP, name,
        {"location": LOCATION, "tags": TAGS, "properties": {
            "networkSecurityGroup": {"id": nsg_id},
            "ipConfigurations": [{"name": "ip1", "properties": {
                "subnet": {"id": subnet_id},
                "publicIPAddress": {"id": pip_id},
            }}],
        }},
    ).result()
    record(name, "Microsoft.Network/networkInterfaces", n.id)
    return n.id


def wait_for_ssh(ip: str, timeout: int = 180) -> bool:
    """Poll until SSH port is open."""
    import socket
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            s = socket.create_connection((ip, 22), timeout=3)
            s.close()
            return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            time.sleep(5)
    return False


def ssh_tunnel(ip: str, key_path: str, local_port: int, remote_port: int):
    """Open SSH tunnel: local_port -> vm:remote_port. Returns subprocess."""
    print(f"  Opening SSH tunnel: localhost:{local_port} -> {ip}:{remote_port}")
    return subprocess.Popen([
        "ssh", "-N", "-L", f"127.0.0.1:{local_port}:localhost:{remote_port}",
        "-i", key_path,
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        f"azureuser@{ip}",
    ], stderr=subprocess.DEVNULL)


def probe_guest_agent(local_port: int, timeout: int = 120) -> bool:
    deadline = time.time() + timeout
    url = f"http://localhost:{local_port}/screenshot"
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                print(f"  /screenshot → HTTP {r.status_code}, {len(r.content)} bytes, content-type: {r.headers.get('content-type')}")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(5)
    return False


# ── Main experiment ───────────────────────────────────────────────────────────

def run():
    uid = uuid.uuid4().hex[:6]
    vm_name  = f"cube-ci-vm-{uid}"
    pip_name = f"cube-ci-ip-{uid}"
    nic_name = f"cube-ci-nic-{uid}"

    compute = _compute()
    network = _network()

    findings = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        print("\n=== EXPERIMENT: Cloud-init guest agent injection ===\n")

        # 1. Generate throwaway SSH keypair
        print("Step 1: Generate throwaway SSH keypair")
        key_path, pub_key = generate_ssh_keypair(tmpdir)
        print(f"  Generated: {Path(key_path).name}")
        findings["ssh_key_injection"] = "generated throwaway keypair"

        # 2. Find Ubuntu image
        print("\nStep 2: Find Ubuntu 22.04 Marketplace image")
        try:
            image_ref = find_ubuntu_image(compute)
            findings["marketplace_image"] = "found"
        except Exception as e:
            print(f"  ERROR: {e}")
            findings["marketplace_image"] = f"error: {e}"
            image_ref = {"publisher": "Canonical", "offer": "0001-com-ubuntu-server-jammy",
                         "sku": "22_04-lts", "version": "latest"}

        # 3. Create networking
        print("\nStep 3: Create networking resources")
        pip_id = create_pip(network, pip_name)
        nic_id = create_nic(network, nic_name, pip_id)

        # 4. Launch VM with cloud-init
        print(f"\nStep 4: Launch VM with cloud-init guest agent injection ({vm_name})")
        cloud_init_b64 = base64.b64encode(CLOUD_INIT.encode()).decode()

        vm = compute.virtual_machines.begin_create_or_update(
            RESOURCE_GROUP, vm_name,
            {
                "location": LOCATION,
                "tags": TAGS,
                "hardware_profile": {"vm_size": VM_SIZE},
                "storage_profile": {
                    "image_reference": image_ref,
                    "os_disk": {"create_option": "FromImage", "delete_option": "Delete"},
                },
                "os_profile": {
                    "computer_name": vm_name,
                    "admin_username": "azureuser",
                    "custom_data": cloud_init_b64,   # ← cloud-init injection
                    "linux_configuration": {
                        "disable_password_authentication": True,
                        "ssh": {"public_keys": [{
                            "path": "/home/azureuser/.ssh/authorized_keys",
                            "key_data": pub_key,     # ← throwaway key
                        }]},
                    },
                },
                "network_profile": {
                    "network_interfaces": [{"id": nic_id, "properties": {"primary": True}}]
                },
            }
        ).result()
        record(vm_name, "Microsoft.Compute/virtualMachines", vm.id)

        # Get public IP
        pip = network.public_ip_addresses.get(RESOURCE_GROUP, pip_name)
        public_ip = pip.ip_address
        print(f"  VM up: {public_ip}")
        findings["vm_launch"] = "success"
        findings["public_ip"] = public_ip

        # 5. Wait for SSH
        print(f"\nStep 5: Wait for SSH ({public_ip}:22)")
        t0 = time.time()
        ssh_ready = wait_for_ssh(public_ip)
        ssh_time = time.time() - t0
        findings["ssh_ready"] = ssh_ready
        findings["ssh_ready_seconds"] = round(ssh_time)
        if ssh_ready:
            print(f"  SSH ready in {ssh_time:.0f}s")
        else:
            print("  SSH timed out — VM may still be booting")

        # 6. Wait for cloud-init to finish (check via SSH)
        if ssh_ready:
            print("\nStep 6: Wait for cloud-init to install guest agent")
            time.sleep(30)  # cloud-init runs asynchronously after SSH opens
            result = subprocess.run([
                "ssh", "-i", key_path,
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                f"azureuser@{public_ip}",
                "cloud-init status --wait; curl -s http://localhost:5000/health",
            ], capture_output=True, text=True, timeout=120)
            findings["cloud_init_output"] = result.stdout.strip()
            findings["cloud_init_stderr"] = result.stderr.strip()[:200]
            print(f"  cloud-init: {result.stdout.strip()[:200]}")

        # 7. SSH tunnel test
        print(f"\nStep 7: SSH tunnel to bypass port blocking")
        local_port = 15000
        tunnel_proc = None
        if ssh_ready:
            tunnel_proc = ssh_tunnel(public_ip, key_path, local_port, GUEST_PORT)
            time.sleep(3)  # let tunnel establish

            t0 = time.time()
            agent_ready = probe_guest_agent(local_port, timeout=60)
            findings["guest_agent_via_tunnel"] = agent_ready
            findings["guest_agent_seconds"] = round(time.time() - t0)
            if agent_ready:
                print(f"  Guest agent reachable via SSH tunnel!")
                # Also test /health
                try:
                    h = requests.get(f"http://localhost:{local_port}/health", timeout=5)
                    print(f"  /health → {h.json()}")
                    findings["health_response"] = h.json()
                except Exception as e:
                    findings["health_error"] = str(e)
            else:
                print("  Guest agent not reachable via tunnel (cloud-init may not have finished)")

            if tunnel_proc:
                tunnel_proc.terminate()

        # 8. Direct port test (without tunnel — will Zscaler block it?)
        print(f"\nStep 8: Direct port test (no tunnel) — testing Zscaler behavior")
        direct_results = {}
        for port in [22, 5000, 80, 443]:
            try:
                r = requests.get(f"http://{public_ip}:{port}/", timeout=5)
                direct_results[port] = r.status_code
            except requests.exceptions.ConnectionError:
                direct_results[port] = "connection_refused"
            except requests.exceptions.Timeout:
                direct_results[port] = "timeout"
            except Exception as e:
                direct_results[port] = str(e)
        findings["direct_port_results"] = direct_results
        print(f"  {direct_results}")

    # ── Save findings ──────────────────────────────────────────────────────────
    import json
    findings_path = Path(__file__).parent / "findings_cloudinit.json"
    findings_path.write_text(json.dumps(findings, indent=2))
    print(f"\n=== FINDINGS ===")
    print(json.dumps(findings, indent=2))
    print(f"\nSaved to: {findings_path}")
    print(f"\nVM still running: {vm_name} @ {public_ip}")
    print("Run: python track.py delete --yes   to clean up")

    return findings


if __name__ == "__main__":
    run()
