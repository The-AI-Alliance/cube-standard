"""
AzureInfraConfig — InfraConfig implementation for Microsoft Azure.

Migrated from experiments/azure-vm-backend/azure_backend.py into the
cube-standard resource lifecycle protocol (design/resource_lifecycle.md).

Provisioning pipeline (~30-90 min, idempotent):
    source_url (HuggingFace qcow2.zip)
        → bootstrap VM downloads + converts to fixed VHD  (in-cloud speed)
        → Blob Storage (PageBlob)
        → Managed Disk (import)
        → Compute Gallery image definition + version
        → ProvisionStore {"image_def": ..., "version": ...}

Launch (~5-10 min per VM):
    Gallery image version
        → NIC + public IP
        → VM (Generalized — cloud-init injects caller's SSH key at first boot)
        → SSH tunnel localhost:{port} → VM:{guest_port}
        → AzureResourceHandle(endpoint="http://localhost:{port}")

Resource lifetime hierarchy:
    Gallery image (long-lived, manual):
        osworld-ubuntu-vm/1.0.0 — created by provision(), persists until
        unprovision() is called explicitly.  Shared across all task runs.
        Represented in ProvisionStore as "osworld-ubuntu-vm@azure:westus2".

    VM instances (short-lived, automatic):
        cube-<run_id>-vm-<uid> — created by launch() at task start, deleted
        by handle.close() at task end.  Orphaned VMs (process crash, timeout)
        are swept by cleanup_stale() using the cube:expires_at ARM tag.

Authentication:
    Uses AzureCliCredential — run `az login` once before using.
    Credentials are never stored in Pydantic fields.

Required Azure resources (pre-existing):
    - Resource group:      resource_group
    - VNet + Subnet:       vnet_name / subnet_name
    - NSG:                 nsg_name (must allow SSH inbound)
    - Compute Gallery:     gallery_name
    - Bootstrap image:     bootstrap_gallery_image (Ubuntu 22.04 + qemu-utils)
      Pre-exists in the gallery; needed to bypass Golden Image Policy.

Usage::

    from cube_infra_azure import AzureInfraConfig
    from cube.resource import VMResourceConfig

    resource = VMResourceConfig(
        name="osworld-ubuntu-vm",
        source_url="https://huggingface.co/datasets/xlangai/ubuntu_osworld/resolve/main/Ubuntu.qcow2.zip",
    )
    infra = AzureInfraConfig(
        subscription="...",
        resource_group="...",
        storage_account="...",
        vnet_name="...",
        subnet_name="...",
        nsg_name="...",
    )
    infra.provision(resource)          # ~30-90 min, idempotent
    run_debug_agent(my_benchmark, infra)
"""

from __future__ import annotations

import base64
import logging
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from cube.infra_utils import build_volume_setup_script
from cube.provision_store import ProvisionStore
from cube.resource import (
    DockerServiceConfig,
    InfraConfig,
    ResourceConfig,
    ResourceHandle,
    ResourceNotReadyError,
    UnsupportedResourceType,
    VMResourceConfig,
)
from cube_infra_azure._utils import BootstrapMonitor, free_port, open_tunnel, open_tunnels, ssh_run, wait_for_ssh

logger = logging.getLogger(__name__)


# ── VM size selection ─────────────────────────────────────────────────────────
# Ordered list of (cpu, ram_gb, size_name) — smallest first.
# Used by _select_vm_size() to satisfy VMResourceConfig.min_cpu_cores / min_ram_gb.
_AZURE_VM_SIZES: list[tuple[int, int, str]] = [
    (2, 8, "Standard_D2s_v3"),
    (4, 16, "Standard_D4s_v3"),
    (8, 32, "Standard_D8s_v3"),
    (16, 64, "Standard_D16s_v3"),
    (32, 128, "Standard_D32s_v3"),
]


def _select_vm_size(default: str, min_cpu: int | None, min_ram: int | None) -> str:
    """Return the smallest Standard_D*s_v3 size satisfying min_cpu and min_ram.

    Falls back to default if no constraint is set or no size matches.
    """
    if min_cpu is None and min_ram is None:
        return default
    for cpu, ram, name in _AZURE_VM_SIZES:
        if (min_cpu is None or cpu >= min_cpu) and (min_ram is None or ram >= min_ram):
            return name
    return default


# ── Bootstrap script ───────────────────────────────────────────────────────────
# Placeholders: {hf_url}, {vhd_sas_url}, {sentinel_sas_url}, {failed_sas_url},
#               {os_type_sas_url}, {winrm_password}

_AZURE_BOOTSTRAP_SCRIPT = """\
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
apt-get install -y -qq qemu-utils qemu-system-x86 ovmf wget curl unzip netcat-openbsd aria2

wget -q "https://aka.ms/downloadazcopy-v10-linux" -O /tmp/azcopy.tar.gz
tar -xzf /tmp/azcopy.tar.gz -C /tmp --wildcards "*/azcopy" 2>/dev/null || \\
    tar -xzf /tmp/azcopy.tar.gz -C /tmp
find /tmp -name azcopy -type f | head -1 | xargs -I{{}} mv {{}} /usr/local/bin/azcopy
chmod +x /usr/local/bin/azcopy
echo "[bootstrap] Tools ready"

# ── download (use Azure cache blob if available, else HuggingFace) ────────────
if [ -n "{cache_sas_url}" ]; then
    echo "[bootstrap] Downloading from Azure cache: {cache_sas_url}"
    azcopy copy "{cache_sas_url}" /data/source.download --blob-type BlockBlob
else
    echo "[bootstrap] Downloading: {hf_url}"
    # aria2c with 16 parallel connections — single-threaded wget capped at 2-8 MB/s
    # from HF→Azure long-haul; parallel chunks typically reach 50-200 MB/s.
    aria2c --console-log-level=warn --summary-interval=10 -x 16 -s 16 \\
           -d /data -o source.download "{hf_url}"
fi
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

# ── detect OS type from qcow2 partition table ────────────────────────────────
echo "[bootstrap] Detecting OS type from partition table..."
apt-get install -y -qq gdisk ntfs-3g qemu-utils
modprobe nbd max_part=16
qemu-nbd --connect=/dev/nbd0 "$QCOW2"
sleep 2
partprobe /dev/nbd0 2>/dev/null || true
sleep 1
ROOT_EXT4=$(lsblk -rno NAME,FSTYPE /dev/nbd0 | awk '$2=="ext4" {{print "/dev/"$1}}' | tail -1)
ROOT_NTFS=$(lsblk -rno NAME,FSTYPE /dev/nbd0 | awk '$2=="ntfs" {{print "/dev/"$1}}' | tail -1)
qemu-nbd --disconnect /dev/nbd0 2>/dev/null || true

if [ -n "$ROOT_NTFS" ]; then
    OS_TYPE="windows"
elif [ -n "$ROOT_EXT4" ]; then
    OS_TYPE="linux"
else
    echo "[bootstrap] WARNING: no ext4 or ntfs partition found — assuming linux"
    OS_TYPE="linux"
fi
echo "[bootstrap] Detected OS type: $OS_TYPE"

# ── write os_type metadata blob ───────────────────────────────────────────────
curl -s -X PUT -H "x-ms-blob-type: BlockBlob" \\
     -H "Content-Length: ${{#OS_TYPE}}" -d "$OS_TYPE" "{os_type_sas_url}" || true
echo "[bootstrap] OS type blob written"

if [ "$OS_TYPE" = "windows" ]; then
SPECIALIZED="{specialized}"
if [ "$SPECIALIZED" = "true" ]; then
# ── Windows Specialized: just convert qcow2 → fixed VHD, no sysprep needed ──
echo "[bootstrap] Windows Specialized image — converting directly to VHD (no sysprep)..."
qemu-img convert -p -O vpc -o subformat=fixed,force_size "$QCOW2" /data/output.vhd
echo "[bootstrap] Converted: $(du -sh /data/output.vhd)"

else
# ── Windows Generalized: inject RunOnce offline, boot, sysprep, then convert ─
echo "[bootstrap] Windows Generalized image — injecting offline setup..."

# Mount NTFS partition via qemu-nbd (losetup cannot read qcow2 partition tables)
apt-get install -y -qq ntfs-3g python3-hivex
modprobe nbd max_part=16
qemu-nbd --connect=/dev/nbd0 "$QCOW2"
sleep 2
partprobe /dev/nbd0 2>/dev/null || true
sleep 1
WIN_PART=$(lsblk -rno NAME,FSTYPE /dev/nbd0 | awk '$2=="ntfs" {{print "/dev/"$1}}' | head -1)
echo "[bootstrap] Windows partition: $WIN_PART"
if [ -z "$WIN_PART" ]; then
    echo "[bootstrap] FAILED: could not find NTFS partition on /dev/nbd0"
    sudo lsblk -rno NAME,FSTYPE /dev/nbd0
    exit 1
fi
mkdir -p /mnt/win
mount -t ntfs-3g -o rw,uid=0,gid=0 "$WIN_PART" /mnt/win
echo "[bootstrap] NTFS mounted: $(ls /mnt/win | head -5)"

# Write the setup PowerShell script to C:\\cube-setup.ps1
cat > /mnt/win/cube-setup.ps1 << 'PSEOF'
# Enable WinRM with Basic auth
winrm quickconfig -q
winrm set winrm/config/service '@{{AllowUnencrypted="true"}}'
winrm set winrm/config/service/auth '@{{Basic="true"}}'
netsh advfirewall firewall add rule name="WinRM-HTTP" dir=in action=allow protocol=TCP localport=5985
# Install OpenSSH
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0 | Out-Null
Set-Service -Name sshd -StartupType Automatic
New-ItemProperty -Path "HKLM:\\SOFTWARE\\OpenSSH" -Name DefaultShell `
    -Value "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" `
    -PropertyType String -Force | Out-Null
$setupDir = "C:\\Windows\\Setup\\Scripts"
New-Item -ItemType Directory -Force -Path $setupDir | Out-Null
Set-Content -Path "$setupDir\\SetupComplete.cmd" -Value @"
netsh advfirewall firewall add rule name="OpenSSH-Server-In-TCP" dir=in action=allow protocol=TCP localport=22
net start sshd
"@
# Run sysprep
& "C:\\Windows\\System32\\Sysprep\\sysprep.exe" /generalize /oobe /shutdown /quiet
PSEOF
echo "[bootstrap] Setup script written"

# Add RunOnce registry key using python-hivex
SOFTWARE_HIVE="/mnt/win/Windows/System32/config/SOFTWARE"
python3 - "$SOFTWARE_HIVE" << 'PYEOF'
import sys
import hivex

hive = hivex.Hivex(sys.argv[1], write=True)
root = hive.root()

def find_or_create(node, *parts):
    for part in parts:
        try:
            node = hive.node_get_child(node, part)
        except RuntimeError:
            node = hive.node_add_child(node, part)
    return node

run_once = find_or_create(
    root,
    "Microsoft", "Windows", "CurrentVersion", "RunOnce"
)
hive.node_set_value(run_once, {{
    "key": "CubeSetup",
    "t": 1,  # REG_SZ
    "value": b"powershell.exe -NonInteractive -ExecutionPolicy Bypass -File C:\\cube-setup.ps1\x00",
}})
hive.commit(None)
print("RunOnce key written")
PYEOF

umount /mnt/win
qemu-nbd --disconnect /dev/nbd0 2>/dev/null || true
echo "[bootstrap] Offline setup complete, unmounted"

# Boot QEMU from raw image (no OVMF — SeaBIOS handles UEFI via CSM, or use OVMF)
QEMU_FMT=$(qemu-img info --output=json "$QCOW2" | python3 -c "import sys,json; print(json.load(sys.stdin)['format'])")
echo "[bootstrap] Image format: $QEMU_FMT"
BIOS_ARGS="-bios /usr/share/ovmf/OVMF.fd"
if [ ! -f /usr/share/ovmf/OVMF.fd ]; then
    echo "[bootstrap] OVMF not found — using SeaBIOS"
    BIOS_ARGS=""
fi
qemu-system-x86_64 \\
    -m 4096 -smp 4 \\
    -drive file="$QCOW2",format="$QEMU_FMT",if=ide \\
    $BIOS_ARGS \\
    -net nic,model=e1000 -net user,hostfwd=tcp::5985-:5985 \\
    -display none \\
    -serial file:/tmp/qemu-serial.log \\
    -daemonize -pidfile /tmp/qemu.pid
echo "[bootstrap] QEMU started (pid $(cat /tmp/qemu.pid))"

echo "[bootstrap] Waiting for Windows WinRM (port 5985, up to 30 min)..."
timeout 1800 bash -c '
    set +e
    while true; do
        HTTP_CODE=$(curl -s -o /dev/null -w "%{{http_code}}" --max-time 5 \
            -X POST http://localhost:5985/wsman \
            -H "Content-Type: application/soap+xml;charset=UTF-8" \
            -d "<test/>" 2>/dev/null)
        if [ "$HTTP_CODE" = "401" ] || [ "$HTTP_CODE" = "200" ]; then
            break
        fi
        echo "[bootstrap] WinRM not ready yet (HTTP $HTTP_CODE) — sleeping 15s..."
        sleep 15
    done
' || {{ echo "[bootstrap] FAILED: WinRM not ready within 1800s"; tail -50 /tmp/qemu-serial.log || true; exit 1; }}
echo "[bootstrap] WinRM reachable"

PS_SCRIPT='
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0 | Out-Null
Set-Service -Name sshd -StartupType Automatic
New-ItemProperty -Path "HKLM:\\SOFTWARE\\OpenSSH" -Name DefaultShell `
    -Value "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" `
    -PropertyType String -Force | Out-Null
$setupDir = "C:\\Windows\\Setup\\Scripts"
New-Item -ItemType Directory -Force -Path $setupDir | Out-Null
Set-Content -Path "$setupDir\\SetupComplete.cmd" -Value @"
netsh advfirewall firewall add rule name=""OpenSSH-Server-In-TCP"" dir=in action=allow protocol=TCP localport=22
net start sshd
"@
& "C:\\Windows\\System32\\Sysprep\\sysprep.exe" /generalize /oobe /shutdown /quiet
'
PS_ENCODED=$(echo "$PS_SCRIPT" | iconv -t UTF-16LE | base64 -w 0)

WINRM_URL="http://localhost:5985/wsman"
WINRM_AUTH="Administrator:{winrm_password}"
WINRM_HDRS=(-H "Content-Type: application/soap+xml;charset=UTF-8" -u "$WINRM_AUTH")

# Step 1: Create a WinRM cmd shell, get the shell ID back
CREATE_BODY='<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
            xmlns:wsa="http://schemas.xmlsoap.org/ws/2004/08/addressing"
            xmlns:wsman="http://schemas.dmtf.org/wbem/wsman/1/wsman.xsd"
            xmlns:rsp="http://schemas.microsoft.com/wbem/wsman/1/windows/shell">
  <s:Header>
    <wsa:Action>http://schemas.xmlsoap.org/ws/2004/09/transfer/Create</wsa:Action>
    <wsa:To>'"$WINRM_URL"'</wsa:To>
    <wsman:ResourceURI>http://schemas.microsoft.com/wbem/wsman/1/windows/shell/cmd</wsman:ResourceURI>
    <wsa:MessageID>uuid:create-1</wsa:MessageID>
    <wsa:ReplyTo><wsa:Address>http://schemas.xmlsoap.org/ws/2004/08/addressing/role/anonymous</wsa:Address></wsa:ReplyTo>
    <wsman:OperationTimeout>PT60.000S</wsman:OperationTimeout>
  </s:Header>
  <s:Body>
    <rsp:Shell><rsp:OutputStreams>stdout stderr</rsp:OutputStreams><rsp:InputStreams>stdin</rsp:InputStreams></rsp:Shell>
  </s:Body>
</s:Envelope>'

echo "[bootstrap] Running PowerShell via WinRM (OpenSSH install + sysprep)..."
set +e
CREATE_RESPONSE=$(curl -s -X POST "$WINRM_URL" "${{WINRM_HDRS[@]}}" --max-time 60 -d "$CREATE_BODY")
CREATE_EXIT=$?
set -e
echo "[bootstrap] WinRM Create exit=$CREATE_EXIT response=$CREATE_RESPONSE"
if [ $CREATE_EXIT -ne 0 ]; then
    echo "[bootstrap] FAILED: WinRM Create returned exit=$CREATE_EXIT"
    echo "[bootstrap] qemu-serial.log tail:"; tail -50 /tmp/qemu-serial.log || true
    exit 1
fi
SHELL_ID=$(echo "$CREATE_RESPONSE" | grep -oP '(?<=<rsp:ShellId>)[^<]+' || true)
if [ -z "$SHELL_ID" ]; then
    echo "[bootstrap] FAILED: could not extract ShellId from Create response"
    echo "[bootstrap] Create response: $CREATE_RESPONSE"
    echo "[bootstrap] qemu-serial.log tail:"; tail -50 /tmp/qemu-serial.log || true
    exit 1
fi
echo "[bootstrap] Shell created: $SHELL_ID"

# Step 2: Send the Command (encoded PowerShell)
CMD_BODY="<?xml version=\\"1.0\\" encoding=\\"UTF-8\\"?>
<s:Envelope xmlns:s=\\"http://www.w3.org/2003/05/soap-envelope\\"
            xmlns:wsa=\\"http://schemas.xmlsoap.org/ws/2004/08/addressing\\"
            xmlns:wsman=\\"http://schemas.dmtf.org/wbem/wsman/1/wsman.xsd\\"
            xmlns:rsp=\\"http://schemas.microsoft.com/wbem/wsman/1/windows/shell\\">
  <s:Header>
    <wsa:Action>http://schemas.microsoft.com/wbem/wsman/1/windows/shell/Command</wsa:Action>
    <wsa:To>$WINRM_URL</wsa:To>
    <wsman:ResourceURI>http://schemas.microsoft.com/wbem/wsman/1/windows/shell/cmd</wsman:ResourceURI>
    <wsa:MessageID>uuid:cmd-1</wsa:MessageID>
    <wsa:ReplyTo><wsa:Address>http://schemas.xmlsoap.org/ws/2004/08/addressing/role/anonymous</wsa:Address></wsa:ReplyTo>
    <wsman:SelectorSet><wsman:Selector Name=\\"ShellId\\">$SHELL_ID</wsman:Selector></wsman:SelectorSet>
    <wsman:OperationTimeout>PT600.000S</wsman:OperationTimeout>
  </s:Header>
  <s:Body>
    <rsp:CommandLine>
      <rsp:Command>powershell.exe</rsp:Command>
      <rsp:Arguments>-NonInteractive -EncodedCommand $PS_ENCODED</rsp:Arguments>
    </rsp:CommandLine>
  </s:Body>
</s:Envelope>"

set +e
CMD_RESPONSE=$(curl -s -X POST "$WINRM_URL" "${{WINRM_HDRS[@]}}" --max-time 60 -d "$CMD_BODY")
CMD_EXIT=$?
set -e
echo "[bootstrap] WinRM Command exit=$CMD_EXIT response=$CMD_RESPONSE"
if [ $CMD_EXIT -ne 0 ]; then
    echo "[bootstrap] FAILED: WinRM Command returned exit=$CMD_EXIT"
    echo "[bootstrap] qemu-serial.log tail:"; tail -50 /tmp/qemu-serial.log || true
    exit 1
fi
CMD_ID=$(echo "$CMD_RESPONSE" | grep -oP '(?<=<rsp:CommandId>)[^<]+' || true)
if [ -z "$CMD_ID" ]; then
    echo "[bootstrap] FAILED: could not extract CommandId from Command response"
    echo "[bootstrap] Command response: $CMD_RESPONSE"
    exit 1
fi
echo "[bootstrap] Command dispatched: $CMD_ID — waiting for sysprep shutdown..."

QEMU_PID=$(cat /tmp/qemu.pid)
WAIT_START=$(date +%s)
SYSPREP_DEADLINE=$(( $(date +%s) + 1800 ))
while kill -0 "$QEMU_PID" 2>/dev/null; do
    ELAPSED=$(( $(date +%s) - WAIT_START ))
    if [ $(date +%s) -gt $SYSPREP_DEADLINE ]; then
        echo "[bootstrap] TIMEOUT: sysprep did not complete within 30 minutes"
        echo "[bootstrap] qemu-serial.log tail:"; tail -100 /tmp/qemu-serial.log || true
        kill "$QEMU_PID" 2>/dev/null || true
        exit 1
    fi
    echo "[bootstrap] Waiting for sysprep... ${{ELAPSED}}s elapsed (QEMU pid=$QEMU_PID)"
    sleep 30
done
echo "[bootstrap] Windows VM shut down (sysprep complete) after $(( $(date +%s) - WAIT_START ))s"

# ── convert sysprepped qcow2 → fixed VHD for Azure upload ────────────────────
echo "[bootstrap] Converting sysprepped image → fixed VHD..."
qemu-img convert -p -O vpc -o subformat=fixed,force_size "$QCOW2" /data/output.vhd
echo "[bootstrap] Converted: $(du -sh /data/output.vhd)"

fi  # end Specialized/Generalized Windows branch

else
# ── Linux: chroot install openssh + walinuxagent + deprovision ───────────────
echo "[bootstrap] Linux image detected — preparing via chroot..."
# Convert first for Linux (chroot path works on VHD via losetup)
echo "[bootstrap] Converting image → fixed VHD..."
qemu-img convert -O vpc -o subformat=fixed,force_size "$QCOW2" /data/output.vhd
echo "[bootstrap] Converted: $(du -sh /data/output.vhd)"
LOOP=$(losetup -f --show -P /data/output.vhd)
sleep 2
ROOT_PART=$(lsblk -rno NAME,FSTYPE "$LOOP" | awk '$2=="ext4" {{print "/dev/"$1}}' | tail -1)
if [ -z "$ROOT_PART" ]; then
    echo "[bootstrap] WARNING: no ext4 partition found, trying whole device"
    ROOT_PART="$LOOP"
fi
mkdir -p /mnt/guest
mount "$ROOT_PART" /mnt/guest
for fs in dev dev/pts proc sys run; do mount --bind "/$fs" "/mnt/guest/$fs" 2>/dev/null || true; done
cp /etc/resolv.conf /mnt/guest/etc/resolv.conf 2>/dev/null || true
chroot /mnt/guest /bin/bash -c "
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
which sshd 2>/dev/null || apt-get install -y -qq openssh-server
dpkg -l walinuxagent 2>/dev/null | grep -q '^ii' || apt-get install -y -qq walinuxagent
ls /etc/ssh/ssh_host_*_key 2>/dev/null | grep -q . || ssh-keygen -A
rm -f /etc/ssh/sshd_not_to_be_run
waagent -force -deprovision+user
"
[ -L /mnt/guest/etc/systemd/system/ssh.service ] && \\
    readlink /mnt/guest/etc/systemd/system/ssh.service | grep -q '/dev/null' && \\
    rm -f /mnt/guest/etc/systemd/system/ssh.service && \\
    echo "[bootstrap] Removed ssh.service mask"
rm -f /mnt/guest/etc/systemd/system/sockets.target.wants/ssh.socket
rm -f /mnt/guest/etc/systemd/system/ssh.socket
echo "[bootstrap] Removed ssh.socket (conflict with ssh.service)"
SSH_SVC=/mnt/guest/lib/systemd/system/ssh.service
SSH_SVC_ALT=/mnt/guest/usr/lib/systemd/system/ssh.service
for svc in "$SSH_SVC" "$SSH_SVC_ALT"; do
    [ -f "$svc" ] && \\
        mkdir -p /mnt/guest/etc/systemd/system/multi-user.target.wants && \\
        ln -sf "${{svc#/mnt/guest}}" \\
            /mnt/guest/etc/systemd/system/multi-user.target.wants/ssh.service && \\
        echo "[bootstrap] Enabled sshd via $svc" && break
done
for fs in run sys proc dev/pts dev; do umount "/mnt/guest/$fs" 2>/dev/null || true; done
umount /mnt/guest
losetup -d "$LOOP" 2>/dev/null || true
echo "[bootstrap] Linux VHD prepared"

fi  # end OS_TYPE branch

# ── upload ────────────────────────────────────────────────────────────────────
echo "[bootstrap] Uploading to Azure Blob Storage..."
azcopy copy /data/output.vhd "{vhd_sas_url}" --blob-type PageBlob
echo "[bootstrap] Upload complete"

# ── signal done ───────────────────────────────────────────────────────────────
curl -s -X PUT -H "x-ms-blob-type: BlockBlob" -H "Content-Length: 0" "{sentinel_sas_url}"
echo "[bootstrap] Done at $(date)"
"""


# ── Docker-host bootstrap script ──────────────────────────────────────────────
# Placeholders: {docker_pull_commands}, {sentinel_sas_url}, {failed_sas_url}
# Runs via cloud-init (custom_data) on the gallery bootstrap Ubuntu VM.
# Installs Docker, pre-pulls all images, deprovisiones for Generalized image,
# then writes sentinel blob.  No SSH key is baked in — Azure injects the
# caller's key at launch time via os_profile (Generalized image pattern).

_DOCKER_BOOTSTRAP_SCRIPT = """\
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

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq docker.io curl walinuxagent

systemctl enable docker
systemctl start docker

# ── add cube user to docker group (no sudo needed at launch time) ─────────────
usermod -aG docker cube

{volume_setup_commands}

# ── pre-pull Docker images ─────────────────────────────────────────────────────
{docker_pull_commands}
echo "[bootstrap] Docker images ready"

# ── deprovision for Generalized gallery image ──────────────────────────────────
# Clears SSH authorized_keys, machine-id, and cloud-init state so Azure can
# inject the caller's SSH key + hostname at first boot (same as VM images).
echo "[bootstrap] Deprovisioning for Generalized image..."
waagent -force -deprovision+user

# ── signal done ───────────────────────────────────────────────────────────────
curl -s -X PUT -H "x-ms-blob-type: BlockBlob" -H "Content-Length: 0" "{sentinel_sas_url}"
echo "[bootstrap] Done at $(date)"
"""


# ── AzureResourceHandle ───────────────────────────────────────────────────────


@dataclass
class AzureResourceHandle(ResourceHandle):
    """ResourceHandle for a running Azure VM with one or more open SSH tunnels."""

    _vm_name: str = field(default="", repr=False)
    _pip_name: str = field(default="", repr=False)
    _nic_name: str = field(default="", repr=False)
    _tunnels: list[subprocess.Popen] = field(default_factory=list, repr=False)

    def close(self) -> None:
        """Terminate SSH tunnel(s) and delete the Azure VM + network resources."""
        for proc in self._tunnels:
            try:
                proc.terminate()
            except Exception:
                pass
        self._tunnels = []
        if self._vm_name:
            logger.info("SSH tunnel(s) closed for run %s", self.run_id[:8])
            assert isinstance(self.infra, AzureInfraConfig)
            self.infra._delete_vm(self._vm_name, self._pip_name, self._nic_name)


# ── AzureInfraConfig ──────────────────────────────────────────────────────────


class AzureInfraConfig(InfraConfig):
    """Azure InfraConfig: provisions and launches CUBE VM resources on Azure.

    Authentication uses AzureCliCredential — run ``az login`` once before using.
    Azure credentials are never stored in fields.

    Typical usage::

        # Minimal — most fields auto-discovered from the resource group:
        infra = AzureInfraConfig(resource_group="my-rg")

        # With explicit overrides for ambiguous resource groups:
        infra = AzureInfraConfig(
            resource_group="shared-rg",
            storage_account="mystorageaccount",
            vnet_name="vnet-westus2",
            nsg_name="cube-nsg",
        )

        infra.provision(resource)       # ~30-90 min, idempotent
        run_debug_agent(benchmark, infra)

    ── Prerequisites (must exist before calling provision()) ─────────────────
    The following Azure resources must be created once per resource group.
    They are never created automatically.

    Resource group:
        az group create --name <rg> --location westus2

    Storage account (must be in the same region as location=):
        az storage account create --name <sa> --resource-group <rg> \
            --location westus2 --sku Standard_LRS --kind StorageV2
        az storage container create --name vhds --account-name <sa>
        # List existing: az storage account list --resource-group <rg> -o table

    VNet + Subnet:
        az network vnet create --name <vnet> --resource-group <rg> \
            --location westus2 --address-prefix 10.0.0.0/16 \
            --subnet-name default --subnet-prefix 10.0.0.0/24
        # List existing: az network vnet list --resource-group <rg> -o table

    NSG (must allow inbound SSH from your IP or 0.0.0.0/0 for testing):
        az network nsg create --name <nsg> --resource-group <rg>
        az network nsg rule create --nsg-name <nsg> --resource-group <rg> \
            --name AllowSSH --priority 1000 --protocol Tcp \
            --destination-port-ranges 22 --access Allow
        # List existing: az network nsg list --resource-group <rg> -o table

    Compute Gallery + bootstrap image:
        az sig create --gallery-name cube_exp_gallery --resource-group <rg>
        # The bootstrap image definition (cube-ubuntu-22-04) must exist in the
        # gallery.  Contact your team admin if provisioning in a new account.

    ── Required ──────────────────────────────────────────────────────────────
    resource_group  str
        Resource group for all managed Azure resources.
        All other auto-discovered fields are scoped to this resource group.

    ── Auto-discovered (leave as None unless there are multiple in the RG) ──
    subscription    str | None = None
        Azure subscription ID.  Auto-populated via ``az account show``.
    storage_account str | None = None
        Storage account used for intermediate VHD blobs during provisioning.
        Must be in the same region as location=.
        Auto-populated if only one storage account exists in the resource group.
    vnet_name       str | None = None
        VNet for launched VMs.
        Auto-populated if only one VNet exists in the resource group.
    subnet_name     str | None = None
        Subnet within vnet_name for launched VMs.
        Auto-populated if only one subnet exists in the VNet.
    nsg_name        str | None = None
        NSG attached to VM NICs.  Must allow inbound SSH from your network.
        Auto-populated if only one NSG exists in the resource group.

    ── Overrideable defaults ─────────────────────────────────────────────────
    location        str = "westus2"
        Azure region.  Gallery images are region-specific.
    container_name  str = "vhds"
        Blob container within storage_account used for VHD blobs.
    gallery_name    str = "cube_exp_gallery"
        Compute Gallery where provisioned image versions are stored.
        Must exist before calling provision().
    vm_size         str = "Standard_D4s_v3"
        VM size for task VMs launched by launch().
    guest_port      int = 5000
        Port the CUBE guest agent listens on inside the VM.
        The SSH tunnel maps a free local port → VM:guest_port.
    ssh_privkey_path    str | None = None
        Path to the SSH private key used to open tunnels into launched VMs.
        Auto-discovered from ~/.ssh/ in priority order: id_ed25519, id_ecdsa,
        id_rsa, id_dsa.  Only the file path is stored — the key is never read
        into memory; it is passed as ``-i {path}`` to SSH subprocess calls.
    ssh_pubkey_path     str | None = None
        Path to the SSH public key.  Auto-derived as ssh_privkey_path + ".pub"
        if not set.  Its *content* is read at launch time and injected into the
        VM via os_profile.linux_configuration.ssh.public_keys (Generalized image).
        Public keys are designed to be distributed.
    tags            dict[str, str] = {"project": "cube"}
        Base tags applied to all Azure resources created by this config.

    ── Bootstrap pipeline (advanced) ─────────────────────────────────────────
    These control the in-cloud bootstrap VM that converts the source qcow2 to
    a fixed VHD.  Only relevant if you need to re-bootstrap or use a different
    base image.  The defaults assume cube-ubuntu-22-04/1.0.0 exists in gallery_name.

    bootstrap_vm_size           str = "Standard_D4s_v3"
    bootstrap_disk_sku          str = "Standard_LRS"
    bootstrap_gallery_image     str = "cube-ubuntu-22-04"
        Gallery image definition used to launch the bootstrap VM.
        Must have qemu-utils installed and exist in gallery_name.
    bootstrap_gallery_image_ver str = "1.0.0"
    bootstrap_os_disk_gb        int = 128
        OS disk size for the bootstrap VM (must fit the qcow2 + VHD in /data).

    """

    # ── Required ──────────────────────────────────────────────────────────────
    resource_group: str

    # ── Auto-discovered ───────────────────────────────────────────────────────
    subscription: str | None = None
    storage_account: str | None = None
    vnet_name: str | None = None
    subnet_name: str | None = None
    nsg_name: str | None = None

    # ── Overrideable defaults ─────────────────────────────────────────────────
    location: str = "westus2"
    container_name: str = "vhds"
    gallery_name: str = "cube_exp_gallery"
    vm_size: str = "Standard_D4s_v3"
    guest_port: int = 5000
    ssh_privkey_path: str | None = Field(default=None, repr=False, exclude=True)
    ssh_pubkey_path: str | None = Field(default=None, repr=False, exclude=True)
    tags: dict[str, str] = Field(default_factory=lambda: {"project": "cube"})

    # ── Bootstrap pipeline ────────────────────────────────────────────────────
    bootstrap_vm_size: str = "Standard_D4s_v3"
    bootstrap_disk_sku: str = "Standard_LRS"
    bootstrap_gallery_image: str = "cube-ubuntu-22-04"
    bootstrap_gallery_image_ver: str = "1.0.0"
    bootstrap_os_disk_gb: int = 128
    windows_admin_username: str = "Docker"
    """Administrator username for Windows VMs. Must match the admin user baked into
    the image (for Specialized images) or the username injected via os_profile
    (for Generalized images)."""
    windows_admin_password: str | None = Field(default=None, repr=False, exclude=True)
    """Administrator password for Windows VMs (Generalized images only).
    Set via WAA_WINDOWS_ADMIN_PASSWORD env var in your recipe.
    Not used for Specialized images — credentials are baked into the image.
    """

    source_cache_blob: str = ""
    """Blob name of a cached source image (e.g. 'sources/waa-windows-vm.img').
    If set and the blob exists, bootstrap downloads from Azure instead of source_url (~3 min vs ~85 min).
    """

    # ── Auto-discovery ────────────────────────────────────────────────────────

    @model_validator(mode="after")
    def _autodiscover(self) -> "AzureInfraConfig":
        """Fill in any None fields by querying the Azure SDK.

        Only runs discovery for fields that are None — explicitly set values
        are always respected.  Subscription falls back to ``az account show``
        (AzureCliCredential already requires ``az login`` anyway).

        Raises ValueError if a resource type cannot be uniquely identified
        (e.g. multiple VNets in the resource group — set ``vnet_name`` explicitly).
        """
        # ── Azure resource discovery (requires SDK calls) ─────────────────────
        needs_azure_discovery = not all(
            [
                self.subscription,
                self.storage_account,
                self.vnet_name,
                self.subnet_name,
                self.nsg_name,
            ]
        )
        if needs_azure_discovery:
            from azure.identity import AzureCliCredential
            from azure.mgmt.network import NetworkManagementClient
            from azure.mgmt.storage import StorageManagementClient

            rg = self.resource_group

            def _pick(items: list, kind: str, param: str) -> str:
                names = [i.name for i in items]
                if not names:
                    raise ValueError(f"No {kind} found in resource group '{rg}'.")
                if len(names) == 1:
                    return names[0]
                raise ValueError(f"Multiple {kind} in '{rg}': {names}.\nSet {param}= explicitly.")

            if not self.subscription:
                result = subprocess.run(
                    ["az", "account", "show", "--query", "id", "-o", "tsv"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        "Could not resolve subscription — run 'az login' or set subscription= explicitly.\n"
                        + result.stderr.strip()
                    )
                object.__setattr__(self, "subscription", result.stdout.strip())

            cred = AzureCliCredential()
            nc = NetworkManagementClient(cred, self.subscription)
            sc = StorageManagementClient(cred, self.subscription)

            if not self.storage_account:
                object.__setattr__(
                    self,
                    "storage_account",
                    _pick(
                        list(sc.storage_accounts.list_by_resource_group(rg)),
                        "storage accounts",
                        "storage_account",
                    ),
                )

            if not self.vnet_name:
                object.__setattr__(
                    self,
                    "vnet_name",
                    _pick(
                        list(nc.virtual_networks.list(rg)),
                        "VNets",
                        "vnet_name",
                    ),
                )

            if not self.subnet_name:
                object.__setattr__(
                    self,
                    "subnet_name",
                    _pick(
                        list(nc.subnets.list(rg, self.vnet_name)),
                        "subnets",
                        "subnet_name",
                    ),
                )

            if not self.nsg_name:
                object.__setattr__(
                    self,
                    "nsg_name",
                    _pick(
                        list(nc.network_security_groups.list(rg)),
                        "NSGs",
                        "nsg_name",
                    ),
                )

            # ── P1: validate storage account is in the same region as location ─
            try:
                sa = sc.storage_accounts.get_properties(rg, self.storage_account)
                sa_location = (sa.location or "").replace(" ", "").lower()
                cfg_location = self.location.replace(" ", "").lower()
                if sa_location != cfg_location:
                    raise ValueError(
                        f"Storage account '{self.storage_account}' is in region "
                        f"'{sa.location}' but AzureInfraConfig.location='{self.location}'.\n"
                        f"VHD blobs and managed disks must be in the same region.\n"
                        f"Either use a storage account in '{self.location}' or set "
                        f"location='{sa.location}'."
                    )
            except ValueError:
                raise
            except Exception:
                pass  # storage account doesn't exist yet — will be created by provision()

        # ── SSH key discovery (local filesystem, no SDK needed) ───────────────
        if self.ssh_privkey_path is None:
            ssh_dir = Path.home() / ".ssh"
            for _name in ["id_ed25519", "id_ecdsa", "id_rsa", "id_dsa"]:
                candidate = ssh_dir / _name
                if candidate.exists():
                    object.__setattr__(self, "ssh_privkey_path", str(candidate))
                    break
            else:
                raise ValueError(
                    "No SSH private key found in ~/.ssh/ "
                    "(tried: id_ed25519, id_ecdsa, id_rsa, id_dsa).\n"
                    "Generate one with: ssh-keygen -t ed25519\n"
                    "Or set ssh_privkey_path= explicitly."
                )

        if self.ssh_pubkey_path is None:
            object.__setattr__(self, "ssh_pubkey_path", self.ssh_privkey_path + ".pub")

        return self

    # ── InfraConfig interface ─────────────────────────────────────────────────

    def fingerprint(self) -> str:
        """Stable key: provider + region only (not VM size or storage SKU).

        Two AzureInfraConfig objects with the same subscription/location share
        the same provisioned gallery images.
        """
        return f"azure:{self.location}"

    def capabilities(self) -> set[str]:
        """Azure can satisfy VM and Docker resources (native hypervisor + Docker in VM)."""
        return {"kvm", "docker"}

    def provision(self, resource: ResourceConfig) -> None:
        """Bootstrap OSWorld (or any VM image) from source_url into the Compute Gallery.

        Pipeline (in-cloud, idempotent at every step):
            source_url → bootstrap VM (download + qemu-img convert + azcopy upload)
                       → Blob Storage → Managed Disk → Gallery image version
                       → ProvisionStore

        Skips the bootstrap VM phase if the sentinel blob already exists.
        Skips the disk/gallery phase if the gallery image version already exists.
        Always no-ops if already registered in the ProvisionStore.

        Raises:
            UnsupportedResourceType: if resource is not VMResourceConfig.
            ValueError: if resource.source_url is not set and no manual registration exists.
        """
        if not isinstance(resource, (VMResourceConfig, DockerServiceConfig)):
            raise UnsupportedResourceType(resource, self)

        shim = self._resource_shim(resource)
        image_name = self._image_name(resource)
        store = ProvisionStore()
        existing = store.get(shim, self)
        if existing:
            logger.info(
                "provision: %r already registered for %s — skipping",
                image_name,
                self.fingerprint(),
            )
            return

        version = "1.0.0"

        if isinstance(resource, DockerServiceConfig):
            if not resource.docker_images:
                raise ValueError(
                    f"Cannot provision {image_name!r}: DockerServiceConfig.docker_images is empty. "
                    f"Specify the Docker Hub images to pre-pull."
                )
            logger.info("provision: building Docker-host image %r …", image_name)
            image_id = self._provision_docker_service(resource, image_name, version)
        else:
            if not resource.source_url:
                raise ValueError(
                    f"Cannot provision {image_name!r}: no source_url set and "
                    f"no registration found for {self.fingerprint()!r}.\n"
                    f'  Manual: infra.register(resource, {{"image_def": ..., "version": ...}})'
                )
            logger.info(
                "provision: bootstrapping %r → gallery image (version %s)",
                image_name,
                version,
            )
            image_id = self._bootstrap(
                url=resource.source_url,
                image_name=image_name,
                version=version,
                uefi=resource.uefi,
                trusted_launch=resource.uefi or resource.tpm,
                specialized=resource.specialized,
            )

        store.put(
            shim,
            self,
            {
                "image_def": image_name,
                "version": version,
                "image_id": image_id,
            },
        )
        logger.info("provision: %r registered for %s", image_name, self.fingerprint())

    def unprovision(self, resource: ResourceConfig) -> None:
        """Delete the gallery image version, VHD blob, sentinel, and ProvisionStore entry.

        Safe to call when not provisioned — no-ops if not registered.

        Raises:
            UnsupportedResourceType: if resource is not VMResourceConfig.
        """
        if not isinstance(resource, (VMResourceConfig, DockerServiceConfig)):
            raise UnsupportedResourceType(resource, self)

        shim = self._resource_shim(resource)
        image_name = self._image_name(resource)
        store = ProvisionStore()
        resource_info = store.get(shim, self)

        if resource_info is None:
            logger.info("unprovision: %r not registered — nothing to do", image_name)
            return

        image_def = resource_info.get("image_def", image_name)
        version = resource_info.get("version", "1.0.0")

        compute = self._compute()
        try:
            logger.info("unprovision: deleting gallery image %s/%s …", image_def, version)
            compute.gallery_image_versions.begin_delete(
                self.resource_group, self.gallery_name, image_def, version
            ).result()
            logger.info("unprovision: gallery image %s/%s deleted", image_def, version)
        except Exception as exc:
            logger.warning("unprovision: could not delete gallery image %s/%s: %s", image_def, version, exc)

        if isinstance(resource, DockerServiceConfig):
            # Docker-host bootstrap uses different sentinel names (no .vhd blob).
            for b in (image_name + ".docker_bootstrap_done", image_name + ".docker_bootstrap_failed"):
                self._delete_blob(b)
        else:
            blob_name = image_name + ".vhd"
            for b in (blob_name, blob_name + ".bootstrap_done", blob_name + ".bootstrap_failed"):
                self._delete_blob(b)

        store.delete(shim, self)
        logger.info("unprovision: %r removed from ProvisionStore", image_name)

    def launch(self, resource: ResourceConfig) -> AzureResourceHandle:
        """Launch a VM from the Compute Gallery, open SSH tunnel, return handle.

        Reads image_def + version from the ProvisionStore.
        Raises ResourceNotReadyError if provision() was never called.

        run_id is generated internally. TTL resolves as:
        self.default_ttl_seconds ?? resource.default_ttl_seconds.
        The VM is tagged with cube: tags for ARM-based cleanup.
        SSH tunnel: localhost:{local_port} → VM:{guest_port}
        """
        if not isinstance(resource, (VMResourceConfig, DockerServiceConfig)):
            raise UnsupportedResourceType(resource, self)

        resource_info = ProvisionStore().get(self._resource_shim(resource), self)
        if resource_info is None:
            raise ResourceNotReadyError(resource, self)

        image_def = resource_info["image_def"]
        version = resource_info["version"]

        run_id = str(uuid.uuid4())
        uid = uuid.uuid4().hex[:6]
        run_id_short = run_id[:8]
        vm_name = f"cube-{run_id_short}-vm-{uid}"

        image_id = (
            f"/subscriptions/{self.subscription}/resourceGroups/{self.resource_group}"
            f"/providers/Microsoft.Compute/galleries/{self.gallery_name}"
            f"/images/{image_def}/versions/{version}"
        )

        # Compute timestamps before creating anything so they can go into tags.
        effective_ttl = (
            self.default_ttl_seconds if self.default_ttl_seconds is not None else resource.default_ttl_seconds
        )
        created_at = datetime.now(timezone.utc)
        expires_at = created_at + timedelta(seconds=effective_ttl) if effective_ttl else None

        # Spec-required tags applied to every launched resource (VM, NIC, IP).
        cube_tags: dict[str, str] = {
            "cube:infra": self.fingerprint(),
            "cube:run_id": run_id,
            "cube:resource": resource.name,
            "cube:created_at": created_at.isoformat(),
        }
        if expires_at:
            cube_tags["cube:expires_at"] = expires_at.isoformat()

        compute = self._compute()

        logger.info("launch: creating network resources for %s", vm_name)
        pip, nic, pip_name, nic_name = self._create_network_resources(run_id_short, uid, cube_tags)

        # VM tags also record NIC/IP names so list_active() needs no string manipulation.
        vm_tags = {
            **self.tags,
            **cube_tags,
            "cube:nic_name": nic_name,
            "cube:ip_name": pip_name,
        }
        pubkey = Path(self.ssh_pubkey_path).read_text().strip()  # type: ignore[arg-type]  # set by _autodiscover

        is_windows = isinstance(resource, VMResourceConfig) and resource.os_type == "windows"
        uefi = isinstance(resource, VMResourceConfig) and resource.uefi
        tpm = isinstance(resource, VMResourceConfig) and resource.tpm
        specialized = isinstance(resource, VMResourceConfig) and resource.specialized

        effective_vm_size = _select_vm_size(
            self.vm_size,
            resource.min_cpu_cores if isinstance(resource, VMResourceConfig) else None,
            resource.min_ram_gb if isinstance(resource, VMResourceConfig) else None,
        )
        logger.info(
            "launch: creating VM %s (%s)  image=%s/%s  os=%s%s",
            vm_name,
            effective_vm_size,
            image_def,
            version,
            "windows" if is_windows else "linux",
            " specialized" if specialized else "",
        )
        t0 = time.time()

        if specialized:
            logger.info("launch: specialized image — skipping os_profile")
            os_profile_spec: dict | None = None
        elif is_windows:
            os_profile_spec = {
                "computer_name": vm_name[:15],  # Windows NetBIOS limit
                "admin_username": self.windows_admin_username,
                "admin_password": self.windows_admin_password,
                "windows_configuration": {
                    "provision_vm_agent": True,
                    "enable_automatic_updates": False,
                },
            }
        else:
            os_profile_spec = {
                "computer_name": vm_name,
                "admin_username": "cube",
                "linux_configuration": {
                    "disable_password_authentication": True,
                    "ssh": {
                        "public_keys": [
                            {
                                "path": "/home/cube/.ssh/authorized_keys",
                                "key_data": pubkey,
                            }
                        ]
                    },
                },
            }

        vm_spec: dict[str, Any] = {
            "location": self.location,
            "tags": vm_tags,
            "hardware_profile": {"vm_size": effective_vm_size},
            "storage_profile": {
                "image_reference": {"id": image_id},
                "os_disk": {
                    "create_option": "FromImage",
                    "managed_disk": {"storage_account_type": "Standard_LRS"},
                    "delete_option": "Delete",
                    **(
                        {"disk_size_gb": resource.os_disk_gb}
                        if isinstance(resource, VMResourceConfig) and resource.os_disk_gb
                        else {}
                    ),
                },
            },
            "network_profile": {"network_interfaces": [{"id": nic.id, "properties": {"primary": True}}]},
        }
        if os_profile_spec is not None:
            vm_spec["os_profile"] = os_profile_spec
        if uefi or tpm:
            vm_spec["security_profile"] = {
                "security_type": "TrustedLaunch",
                "uefi_settings": {
                    "secure_boot_enabled": True,
                    "v_tpm_enabled": tpm,
                },
            }

        poller = compute.virtual_machines.begin_create_or_update(  # type: ignore[call-overload]
            self.resource_group,
            vm_name,
            vm_spec,  # type: ignore[arg-type]
        )
        # Wait up to 10 min for ARM to confirm provisioning. Windows VMs sometimes
        # report PowerState/running before the Azure guest agent checks in, so we
        # proceed as long as the VM is running even if ARM says still Creating.
        _vm_create_timeout = 600
        try:
            poller.result(timeout=_vm_create_timeout)
        except Exception as exc:
            vm_view = compute.virtual_machines.instance_view(self.resource_group, vm_name)
            power_states = [s.code for s in (vm_view.statuses or []) if s.code and s.code.startswith("PowerState/")]
            if "PowerState/running" not in power_states:
                self._delete_vm(vm_name, pip_name, nic_name)
                raise RuntimeError(f"VM {vm_name} did not reach running state: {exc}") from exc
            logger.warning("launch: ARM poller timed out but VM is running — continuing (power=%s)", power_states)
        elapsed = time.time() - t0

        pip_info = self._network().public_ip_addresses.get(self.resource_group, pip_name)
        assert pip_info.ip_address, "Public IP address was not assigned"
        public_ip = pip_info.ip_address
        logger.info("launch: VM ready in %.0fs: %s @ %s", elapsed, vm_name, public_ip)

        # For Windows: use VMAccessAgent to open firewall + inject SSH public key.
        # For Windows: inject SSH key + open firewall via RunCommand.
        # RunCommand goes through the Azure VM Agent and works for both Specialized
        # and Generalized images. VMAccessAgent is unreliable for Specialized images
        # because it may not overwrite a pre-existing administrators_authorized_keys.
        if is_windows:
            logger.info("launch: injecting SSH key + firewall rule via RunCommand for %s", vm_name)
            escaped_pubkey = pubkey.replace("'", "''")
            run_cmd_payload = {
                "command_id": "RunPowerShellScript",
                "script": [
                    f"Set-Content -Path 'C:\\ProgramData\\ssh\\administrators_authorized_keys' -Value '{escaped_pubkey}'",
                    "icacls 'C:\\ProgramData\\ssh\\administrators_authorized_keys' /inheritance:r /grant 'SYSTEM:(F)' /grant 'BUILTIN\\Administrators:(F)'",
                    'netsh advfirewall firewall add rule name="OpenSSH-Server-In-TCP" dir=in action=allow protocol=TCP localport=22',
                    "Start-Service sshd",
                ],
            }
            last_exc: Exception | None = None
            for attempt in range(1, 6):
                try:
                    compute.virtual_machines.begin_run_command(
                        self.resource_group, vm_name, run_cmd_payload
                    ).result(timeout=300)
                    logger.info("launch: SSH key injected and firewall rule opened for %s", vm_name)
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    logger.warning(
                        "launch: RunCommand attempt %d/5 failed for %s: %s", attempt, vm_name, exc
                    )
                    time.sleep(min(2**attempt, 30))
            if last_exc is not None:
                raise RuntimeError(
                    f"RunCommand failed for {vm_name} after 5 attempts; last error: {last_exc}"
                ) from last_exc

        # SSH + tunnel — clean up VM on any failure to avoid orphaned resources.
        primary_user = self.windows_admin_username if is_windows else "cube"
        fallback_users = ["Administrator"] if is_windows else ["ubuntu", "azureuser", "root"]
        try:
            logger.info("launch: waiting for SSH on %s…", public_ip)
            active_user = wait_for_ssh(
                public_ip,
                primary_user,
                self.ssh_privkey_path,
                fallback_users=fallback_users,
                timeout=900,
            )

            if isinstance(resource, DockerServiceConfig):
                # Start services inside the VM, then open one tunnel per service port.
                if resource.launch_script:
                    logger.info("launch: starting Docker services on %s", vm_name)
                    self._ssh_run(public_ip, active_user, resource.launch_script)
                    logger.info("launch: Docker services started")
                endpoints, tunnels = open_tunnels(public_ip, active_user, self.ssh_privkey_path, resource.services)
                logger.info("launch: opened %d tunnel(s): %s", len(tunnels), list(endpoints.keys()))
                # Use the first endpoint as the canonical single endpoint for compat.
                endpoint = next(iter(endpoints.values())) if endpoints else None
            else:
                local_port = free_port()
                logger.info(
                    "launch: opening tunnel localhost:%d → %s:%d",
                    local_port,
                    public_ip,
                    self.guest_port,
                )
                tunnel = open_tunnel(public_ip, active_user, self.ssh_privkey_path, local_port, self.guest_port)
                endpoint = f"http://localhost:{local_port}"
                endpoints = {}
                tunnels = [tunnel]
                # Open additional tunnels for ports the resource asks to expose.
                # Each gets a unique host freeport so parallel workers don't collide.
                extra_ports = getattr(resource, "forwarded_ports", []) or []
                for vm_port in extra_ports:
                    extra_local = free_port()
                    logger.info(
                        "launch: opening extra tunnel localhost:%d → %s:%d",
                        extra_local,
                        public_ip,
                        vm_port,
                    )
                    tunnels.append(
                        open_tunnel(public_ip, active_user, self.ssh_privkey_path, extra_local, vm_port)
                    )
                    endpoints[f"vm_port_{vm_port}"] = f"http://localhost:{extra_local}"
        except Exception:
            logger.warning("launch: SSH/tunnel failed — cleaning up VM %s", vm_name)
            self._delete_vm(vm_name, pip_name, nic_name)
            raise

        return AzureResourceHandle(
            run_id=run_id,
            resource=resource,
            infra=self,
            endpoint=endpoint,
            endpoints=endpoints,
            created_at=created_at,
            expires_at=expires_at,
            _vm_name=vm_name,
            _pip_name=pip_name,
            _nic_name=nic_name,
            _tunnels=tunnels,
        )

    def list_active(self, run_id: str | None = None) -> list[AzureResourceHandle]:
        """List active CUBE VMs in this resource group, filtered by run_id if provided.

        Queries ARM directly via tags. Cannot reconstruct SSH tunnels — handles
        are returned with endpoint=None. Use run_id to call cleanup() from any process.
        """
        compute = self._compute()
        handles: list[AzureResourceHandle] = []

        try:
            vms = list(compute.virtual_machines.list(self.resource_group))
        except Exception as e:
            logger.warning("list_active: failed to list VMs: %s", e)
            return handles

        for vm in vms:
            vm_tags = vm.tags or {}
            if vm_tags.get("cube:infra") != self.fingerprint():
                continue
            if run_id and vm_tags.get("cube:run_id") != run_id:
                continue

            vm_run_id = vm_tags.get("cube:run_id", "unknown")
            resource_name = vm_tags.get("cube:resource", "unknown")
            vm_name = vm.name or ""
            # NIC/IP names are stored in tags — no fragile string manipulation.
            pip_name = vm_tags.get("cube:ip_name", "")
            nic_name = vm_tags.get("cube:nic_name", "")

            handles.append(
                AzureResourceHandle(
                    run_id=vm_run_id,
                    resource=VMResourceConfig(name=resource_name),
                    infra=self,
                    endpoint=None,  # tunnel cannot be reconstructed
                    _vm_name=vm_name,
                    _pip_name=pip_name,
                    _nic_name=nic_name,
                    _tunnels=[],
                )
            )

        return handles

    def cleanup(self, run_id: str) -> None:
        """Delete all CUBE VMs tagged with run_id."""
        handles = self.list_active(run_id=run_id)
        if not handles:
            logger.info("cleanup: no active VMs for run %s", run_id[:8])
            return
        for handle in handles:
            self._delete_vm(handle._vm_name, handle._pip_name, handle._nic_name)
        logger.info("cleanup: removed %d VM(s) for run %s", len(handles), run_id[:8])

    def cleanup_stale(self, max_age_seconds: int | None = None) -> list[str]:
        """Delete CUBE VMs that have expired or exceeded max_age_seconds.

        Checks in priority order:
          1. cube:expires_at tag < now  →  delete (TTL set at launch time)
          2. max_age_seconds set and cube:created_at age > max_age_seconds  →  delete

        Returns list of deleted VM names.
        """
        compute = self._compute()
        deleted: list[str] = []
        now = datetime.now(timezone.utc)

        try:
            vms = list(compute.virtual_machines.list(self.resource_group))
        except Exception as e:
            logger.warning("cleanup_stale: failed to list VMs: %s", e)
            return deleted

        for vm in vms:
            vm_tags = vm.tags or {}
            if vm_tags.get("cube:infra") != self.fingerprint():
                continue

            should_delete = False
            has_valid_expires_at = False

            # Priority 1: explicit TTL tag written at launch time.
            expires_at_str = vm_tags.get("cube:expires_at")
            if expires_at_str:
                try:
                    expires_at = datetime.fromisoformat(expires_at_str)
                    has_valid_expires_at = True
                    if expires_at < now:
                        should_delete = True
                except ValueError:
                    logger.warning("cleanup_stale: invalid cube:expires_at %r on %s", expires_at_str, vm.name)

            # Priority 2: age-based fallback (skipped if expires_at is set).
            if not has_valid_expires_at and not should_delete and max_age_seconds is not None:
                created_at_str = vm_tags.get("cube:created_at")
                try:
                    if created_at_str:
                        created_at = datetime.fromisoformat(created_at_str)
                        age = (now - created_at).total_seconds()
                    elif hasattr(vm, "time_created") and vm.time_created:
                        age = (now - vm.time_created).total_seconds()
                    else:
                        age = 0
                    should_delete = age > max_age_seconds
                except (ValueError, TypeError):
                    pass

            if should_delete:
                vm_name = vm.name or ""
                pip_name = vm_tags.get("cube:ip_name", "")
                nic_name = vm_tags.get("cube:nic_name", "")
                self._delete_vm(vm_name, pip_name, nic_name)
                deleted.append(vm_name)

        if deleted:
            logger.info("cleanup_stale: removed %d VM(s): %s", len(deleted), deleted)
        return deleted

    def cleanup_orphaned_resources(self) -> dict[str, list[str]]:
        """Delete NICs, public IPs, and managed disks left behind by crashed runs.

        These are not reachable via cleanup_stale() because cleanup_stale() finds
        resources through their parent VM — once the VM is gone the NIC/IP/disk
        becomes invisible to tag-based queries.

        Identifies orphans by naming convention:
          - NICs:   cube-*-nic-*  with no VM attached
          - IPs:    cube-*-ip-*   with no NIC attached
          - Disks:  cube-disk-*   that are Unattached

        Returns dict with keys "nics", "ips", "disks" listing deleted resource names.
        """
        from azure.core.exceptions import ResourceNotFoundError

        compute = self._compute()
        network = self._network()
        rg = self.resource_group
        result: dict[str, list[str]] = {"nics": [], "ips": [], "disks": []}

        # Orphaned NICs — no VM attached, name matches cube convention
        try:
            for nic in network.network_interfaces.list(rg):
                if not (nic.name and "nic" in nic.name and nic.name.startswith("cube-")):
                    continue
                if nic.virtual_machine:
                    continue  # still attached to a VM
                logger.info("cleanup_orphaned_resources: deleting NIC %s", nic.name)
                try:
                    network.network_interfaces.begin_delete(rg, nic.name).result()
                    result["nics"].append(nic.name)
                except (ResourceNotFoundError, Exception) as exc:
                    logger.warning("cleanup_orphaned_resources: NIC %s: %s", nic.name, exc)
        except Exception as exc:
            logger.warning("cleanup_orphaned_resources: failed to list NICs: %s", exc)

        # Orphaned IPs — no NIC attached (NIC deletion above may free them),
        # name matches cube convention
        try:
            for pip in network.public_ip_addresses.list(rg):
                if not (pip.name and "ip" in pip.name and pip.name.startswith("cube-")):
                    continue
                if pip.ip_configuration:
                    continue  # still attached to a NIC
                logger.info("cleanup_orphaned_resources: deleting IP %s", pip.name)
                try:
                    network.public_ip_addresses.begin_delete(rg, pip.name).result()
                    result["ips"].append(pip.name)
                except (ResourceNotFoundError, Exception) as exc:
                    logger.warning("cleanup_orphaned_resources: IP %s: %s", pip.name, exc)
        except Exception as exc:
            logger.warning("cleanup_orphaned_resources: failed to list IPs: %s", exc)

        # Orphaned intermediate disks — Unattached, name matches cube-disk-* or
        # cube-dockerhost-disk-* (bootstrap OS disks left after interrupted provision).
        try:
            for disk in compute.disks.list_by_resource_group(rg):
                if not (
                    disk.name and (disk.name.startswith("cube-disk-") or disk.name.startswith("cube-dockerhost-disk-"))
                ):
                    continue
                if disk.disk_state != "Unattached":
                    continue
                logger.info("cleanup_orphaned_resources: deleting disk %s (%dGB)", disk.name, disk.disk_size_gb or 0)
                try:
                    compute.disks.begin_delete(rg, disk.name).result()
                    result["disks"].append(disk.name)
                except (ResourceNotFoundError, Exception) as exc:
                    logger.warning("cleanup_orphaned_resources: disk %s: %s", disk.name, exc)
        except Exception as exc:
            logger.warning("cleanup_orphaned_resources: failed to list disks: %s", exc)

        # Stale docker-host bootstrap VMs (cube-dockerhost-*) — these should
        # have been deleted by _provision_docker_service() but can linger if the
        # process crashed mid-provision.  Delete VM + NIC + IP; keep the OS disk
        # (delete_option=Detach) so that a re-provision can continue.
        result["bootstrap_vms"] = []
        try:
            for vm in compute.virtual_machines.list(rg):
                if not (vm.name and vm.name.startswith("cube-dockerhost-")):
                    continue
                logger.info("cleanup_orphaned_resources: deleting stale bootstrap VM %s", vm.name)
                try:
                    # Deallocate first so NIC/IP can be freed
                    compute.virtual_machines.begin_deallocate(rg, vm.name).result()
                    compute.virtual_machines.begin_delete(rg, vm.name).result()
                    result["bootstrap_vms"].append(vm.name)
                    # Best-effort network cleanup using naming convention.
                    # _create_network_resources(uid, uid) → "cube-{uid}-ip-{uid}"
                    uid = vm.name[len("cube-dockerhost-") :]
                    self._delete_network_resources(
                        f"cube-{uid}-ip-{uid}",
                        f"cube-{uid}-nic-{uid}",
                    )
                except Exception as exc:
                    logger.warning("cleanup_orphaned_resources: VM %s: %s", vm.name, exc)
        except Exception as exc:
            logger.warning("cleanup_orphaned_resources: failed to list VMs: %s", exc)

        total = sum(len(v) for v in result.values())
        if total:
            logger.info("cleanup_orphaned_resources: deleted %d resource(s): %s", total, result)
        else:
            logger.info("cleanup_orphaned_resources: nothing to clean up")
        return result

    # ── Private Azure SDK methods ─────────────────────────────────────────────

    def _cred(self) -> Any:
        from azure.identity import AzureCliCredential

        return AzureCliCredential()

    def _compute(self) -> Any:
        from azure.mgmt.compute import ComputeManagementClient

        return ComputeManagementClient(self._cred(), self.subscription)

    def _network(self) -> Any:
        from azure.mgmt.network import NetworkManagementClient

        return NetworkManagementClient(self._cred(), self.subscription)

    def _storage(self) -> Any:
        from azure.mgmt.storage import StorageManagementClient

        return StorageManagementClient(self._cred(), self.subscription)

    def _get_storage_key(self) -> str:
        storage = self._storage()
        try:
            storage.storage_accounts.get_properties(self.resource_group, self.storage_account)
        except Exception:
            logger.info("Creating storage account: %s", self.storage_account)
            storage.storage_accounts.begin_create(  # type: ignore[call-overload]
                self.resource_group,
                self.storage_account,
                {  # type: ignore[arg-type]
                    "location": self.location,
                    "tags": self.tags,
                    "sku": {"name": "Standard_LRS"},
                    "kind": "StorageV2",
                },
            ).result()
        keys = storage.storage_accounts.list_keys(self.resource_group, self.storage_account)
        assert keys.keys, "Storage account returned no keys"
        return keys.keys[0].value  # type: ignore[return-value]

    def _blob_service_client(self) -> Any:
        from azure.storage.blob import BlobServiceClient

        account_key = self._get_storage_key()
        conn_str = (
            f"DefaultEndpointsProtocol=https;AccountName={self.storage_account};"
            f"AccountKey={account_key};EndpointSuffix=core.windows.net"
        )
        return BlobServiceClient.from_connection_string(
            conn_str,
            max_single_put_size=4 * 1024 * 1024,
            max_page_size=4 * 1024 * 1024,
            connection_timeout=300,
            read_timeout=600,
        )

    def _delete_blob(self, blob_name: str) -> None:
        """Delete a blob from the VHD container (no-op if it doesn't exist)."""
        from azure.core.exceptions import ResourceNotFoundError

        try:
            svc = self._blob_service_client()
            svc.get_blob_client(self.container_name, blob_name).delete_blob()
            logger.info("_delete_blob: deleted %s", blob_name)
        except ResourceNotFoundError:
            logger.debug("_delete_blob: %s not found, skipping", blob_name)
        except Exception as exc:
            logger.warning("_delete_blob: could not delete %s: %s", blob_name, exc)

    def blob_exists(self, blob_name: str) -> bool:
        """Return True if a blob exists in the VHD container."""
        try:
            svc = self._blob_service_client()
            svc.get_blob_client(self.container_name, blob_name).get_blob_properties()
            return True
        except Exception:
            return False

    def generate_sas_url(self, blob_name: str, expiry_hours: int = 8, write: bool = True) -> str:
        """Generate a pre-authorized SAS URL for a blob (read or write)."""
        from azure.storage.blob import BlobSasPermissions, BlobServiceClient, generate_blob_sas

        account_key = self._get_storage_key()
        svc = BlobServiceClient(
            f"https://{self.storage_account}.blob.core.windows.net",
            credential=account_key,
        )
        container = svc.get_container_client(self.container_name)
        try:
            container.get_container_properties()
        except Exception:
            container.create_container()

        expiry = datetime.now(timezone.utc) + timedelta(hours=expiry_hours)
        perms = BlobSasPermissions(read=True, write=write, create=write, add=write)
        sas = generate_blob_sas(
            account_name=self.storage_account,
            container_name=self.container_name,
            blob_name=blob_name,
            account_key=account_key,
            permission=perms,
            expiry=expiry,
        )
        return f"https://{self.storage_account}.blob.core.windows.net/{self.container_name}/{blob_name}?{sas}"

    def _create_network_resources(self, run_id_short: str, uid: str, cube_tags: dict[str, str] | None = None) -> tuple:
        """Create a static public IP and NIC. Returns (pip, nic, pip_name, nic_name).

        cube_tags are applied alongside self.tags so all spec-required tags
        (cube:infra, cube:run_id, cube:resource, cube:created_at, cube:expires_at)
        appear on every network resource.
        """
        network = self._network()
        pip_name = f"cube-{run_id_short}-ip-{uid}"
        nic_name = f"cube-{run_id_short}-nic-{uid}"
        resource_tags = {**self.tags, **(cube_tags or {})}

        pip = network.public_ip_addresses.begin_create_or_update(  # type: ignore[call-overload]
            self.resource_group,
            pip_name,
            {  # type: ignore[arg-type]
                "location": self.location,
                "tags": resource_tags,
                "sku": {"name": "Standard"},
                "properties": {"publicIPAllocationMethod": "Static"},
            },
        ).result()

        subnet_id = (
            f"/subscriptions/{self.subscription}/resourceGroups/{self.resource_group}"
            f"/providers/Microsoft.Network/virtualNetworks/{self.vnet_name}/subnets/{self.subnet_name}"
        )
        nsg_id = (
            f"/subscriptions/{self.subscription}/resourceGroups/{self.resource_group}"
            f"/providers/Microsoft.Network/networkSecurityGroups/{self.nsg_name}"
        )
        nic = network.network_interfaces.begin_create_or_update(  # type: ignore[call-overload]
            self.resource_group,
            nic_name,
            {  # type: ignore[arg-type]
                "location": self.location,
                "tags": resource_tags,
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
        ).result()

        return pip, nic, pip_name, nic_name

    def _delete_vm(self, vm_name: str, pip_name: str, nic_name: str) -> None:
        """Delete a VM and its associated NIC + public IP."""
        compute = self._compute()
        network = self._network()
        logger.info("_delete_vm: deleting %s", vm_name)
        try:
            compute.virtual_machines.begin_delete(self.resource_group, vm_name).result()
            logger.info("_delete_vm: VM deleted: %s", vm_name)
        except Exception as e:
            logger.warning("_delete_vm: VM delete failed: %s", e)

        for fn, label, name in [
            (network.network_interfaces.begin_delete, "NIC", nic_name),
            (network.public_ip_addresses.begin_delete, "IP", pip_name),
        ]:
            if not name:
                continue
            try:
                fn(self.resource_group, name).result()
                logger.info("_delete_vm: %s deleted: %s", label, name)
            except Exception as exc:
                logger.warning("_delete_vm: %s deletion failed for %s: %s", label, name, exc)

    # ── Provisioning internals ────────────────────────────────────────────────

    def _import_disk(self, blob_url: str, disk_name: str, os_type: str = "linux") -> str:
        """Create a Managed Disk from a VHD blob. Returns the disk name.

        Always deletes any existing disk first (import is a no-op if disk exists).
        """
        arm_os_type = "Windows" if os_type == "windows" else "Linux"
        logger.info("_import_disk: %s → %s (%s)", blob_url.split("/")[-1].split("?")[0], disk_name, arm_os_type)
        t0 = time.time()
        compute = self._compute()

        try:
            compute.disks.begin_delete(self.resource_group, disk_name).result()
            logger.info("_import_disk: deleted existing disk %s", disk_name)
        except Exception:
            pass

        poller = compute.disks.begin_create_or_update(  # type: ignore[call-overload]
            self.resource_group,
            disk_name,
            {  # type: ignore[arg-type]
                "location": self.location,
                "tags": self.tags,
                "sku": {"name": "Standard_LRS"},
                "properties": {
                    "creationData": {
                        "createOption": "Import",
                        "sourceUri": blob_url,
                        "storageAccountId": (
                            f"/subscriptions/{self.subscription}/resourceGroups/{self.resource_group}"
                            f"/providers/Microsoft.Storage/storageAccounts/{self.storage_account}"
                        ),
                    },
                    "osType": arm_os_type,
                },
            },
        )
        disk = poller.result()
        logger.info(
            "_import_disk: done in %.0fs: %s (%s GB)",
            time.time() - t0,
            disk_name,
            disk.disk_size_gb,
        )
        return disk_name

    def _ensure_gallery(self) -> str:
        """Create Compute Gallery if it doesn't exist. Returns gallery name."""
        compute = self._compute()
        try:
            compute.galleries.get(self.resource_group, self.gallery_name)
        except Exception:
            logger.info("_ensure_gallery: creating %s", self.gallery_name)
            compute.galleries.begin_create_or_update(  # type: ignore[call-overload]
                self.resource_group,
                self.gallery_name,
                {  # type: ignore[arg-type]
                    "location": self.location,
                    "tags": self.tags,
                    "description": "CUBE benchmark VM image gallery",
                },
            ).result()
        return self.gallery_name

    def _create_image_definition(
        self,
        name: str,
        os_state: Literal["Generalized", "Specialized"] = "Generalized",
        os_type: Literal["linux", "windows"] = "linux",
        uefi: bool = False,
        trusted_launch: bool = False,
    ) -> str:
        """Create a gallery image definition (idempotent). Returns definition name."""
        self._ensure_gallery()
        compute = self._compute()
        try:
            compute.gallery_images.get(self.resource_group, self.gallery_name, name)
            logger.info("_create_image_definition: %s already exists", name)
            return name
        except Exception:
            pass

        hyper_v = "V2" if uefi else "V1"
        arm_os_type = "Windows" if os_type == "windows" else "Linux"
        sku = "windows" if os_type == "windows" else "linux"
        logger.info(
            "_create_image_definition: %s (%s, HyperV %s, os=%s%s)",
            name,
            os_state,
            hyper_v,
            os_type,
            ", TrustedLaunch" if trusted_launch else "",
        )
        props: dict[str, Any] = {
            "location": self.location,
            "tags": self.tags,
            "os_type": arm_os_type,
            "os_state": os_state,
            "hyper_v_generation": hyper_v,
            "identifier": {"publisher": "cube", "offer": name, "sku": sku},
        }
        if trusted_launch:
            props["features"] = [{"name": "SecurityType", "value": "TrustedLaunchSupported"}]
        compute.gallery_images.begin_create_or_update(  # type: ignore[call-overload]
            self.resource_group,
            self.gallery_name,
            name,
            props,
        ).result()
        logger.info("_create_image_definition: created %s", name)
        return name

    def _create_image_version(self, image_def: str, version: str, disk_name: str) -> str:
        """Publish a Managed Disk as a gallery image version (idempotent).

        Returns the full gallery image version resource ID.
        """
        compute = self._compute()
        try:
            existing = compute.gallery_image_versions.get(self.resource_group, self.gallery_name, image_def, version)
            if existing.provisioning_state == "Succeeded":
                logger.info("_create_image_version: %s/%s already exists", image_def, version)
                return existing.id or ""
        except Exception:
            pass

        disk = compute.disks.get(self.resource_group, disk_name)
        logger.info(
            "_create_image_version: publishing %s/%s from %s (%s GB)…",
            image_def,
            version,
            disk_name,
            disk.disk_size_gb,
        )
        t0 = time.time()
        poller = compute.gallery_image_versions.begin_create_or_update(  # type: ignore[call-overload]
            self.resource_group,
            self.gallery_name,
            image_def,
            version,
            {  # type: ignore[arg-type]
                "location": self.location,
                "tags": self.tags,
                "publishing_profile": {
                    "replica_count": 1,
                    "storage_account_type": "Standard_LRS",
                    "target_regions": [
                        {
                            "name": self.location,
                            "regional_replica_count": 1,
                            "storage_account_type": "Standard_LRS",
                        }
                    ],
                    "exclude_from_latest": False,
                },
                "storage_profile": {
                    "os_disk_image": {
                        "source": {"id": disk.id},
                        "host_caching": "ReadWrite",
                    }
                },
            },
        )
        version_obj = poller.result()
        logger.info("_create_image_version: done in %.0fs: %s", time.time() - t0, version_obj.id)
        return version_obj.id or ""

    def _ensure_resource_from_blob(
        self,
        vhd_blob_name: str,
        name: str,
        version: str = "1.0.0",
        os_type: Literal["linux", "windows"] = "linux",
        uefi: bool = False,
        trusted_launch: bool = False,
        specialized: bool = False,
    ) -> str:
        """Import VHD blob → disk → gallery image definition + version.

        Idempotent at each step. Returns the gallery image version resource ID.
        """
        blob_url = f"https://{self.storage_account}.blob.core.windows.net/{self.container_name}/{vhd_blob_name}"
        disk_name = f"cube-disk-{name}"
        self._import_disk(blob_url, disk_name, os_type=os_type)
        self._create_image_definition(
            name,
            os_state="Specialized" if specialized else "Generalized",
            os_type=os_type,
            uefi=uefi,
            trusted_launch=trusted_launch,
        )
        image_id = self._create_image_version(name, version, disk_name)
        logger.info("_ensure_resource_from_blob: image ready: %s/%s", name, version)

        # Gallery image has its own replicated storage — the source disk is no longer needed.
        try:
            self._compute().disks.begin_delete(self.resource_group, disk_name).result()
            logger.info("_ensure_resource_from_blob: deleted intermediate disk %s", disk_name)
        except Exception as exc:
            logger.warning("_ensure_resource_from_blob: could not delete disk %s: %s", disk_name, exc)

        return image_id

    def _launch_bootstrap_vm(self, script: str) -> dict:
        """Launch a lightweight Ubuntu VM with the bootstrap script.

        Returns {vm_name, pip_name, nic_name, public_ip}.
        """
        uid = uuid.uuid4().hex[:6]
        vm_name = f"cube-bootstrap-{uid}"
        pubkey = Path(self.ssh_pubkey_path).read_text().strip()
        custom_data_b64 = base64.b64encode(script.encode()).decode()
        compute = self._compute()

        logger.info("_launch_bootstrap_vm: creating network resources")
        pip, nic, pip_name, nic_name = self._create_network_resources(uid, uid)

        logger.info(
            "_launch_bootstrap_vm: launching %s (%s, %d GB OS disk)",
            vm_name,
            self.bootstrap_vm_size,
            self.bootstrap_os_disk_gb,
        )
        t0 = time.time()
        poller = compute.virtual_machines.begin_create_or_update(  # type: ignore[call-overload]
            self.resource_group,
            vm_name,
            {  # type: ignore[arg-type]
                "location": self.location,
                "tags": {**self.tags, "role": "bootstrap"},
                "hardware_profile": {"vm_size": self.bootstrap_vm_size},
                "storage_profile": {
                    "image_reference": {
                        "id": (
                            f"/subscriptions/{self.subscription}/resourceGroups/{self.resource_group}"
                            f"/providers/Microsoft.Compute/galleries/{self.gallery_name}"
                            f"/images/{self.bootstrap_gallery_image}"
                            f"/versions/{self.bootstrap_gallery_image_ver}"
                        )
                    },
                    "os_disk": {
                        "create_option": "FromImage",
                        "managed_disk": {"storage_account_type": self.bootstrap_disk_sku},
                        "disk_size_gb": self.bootstrap_os_disk_gb,
                        "delete_option": "Delete",
                    },
                },
                "os_profile": {
                    "computer_name": vm_name,
                    "admin_username": "azureuser",
                    "custom_data": custom_data_b64,
                    "linux_configuration": {
                        "disable_password_authentication": True,
                        "ssh": {
                            "public_keys": [
                                {
                                    "path": "/home/azureuser/.ssh/authorized_keys",
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

        pip_info = self._network().public_ip_addresses.get(self.resource_group, pip_name)
        assert pip_info.ip_address
        public_ip = pip_info.ip_address
        logger.info(
            "_launch_bootstrap_vm: VM ready in %ds: %s @ %s",
            int(time.time() - t0),
            vm_name,
            public_ip,
        )
        logger.info("_launch_bootstrap_vm: SSH: ssh -i %s azureuser@%s", self.ssh_privkey_path, public_ip)
        return {"vm_name": vm_name, "pip_name": pip_name, "nic_name": nic_name, "public_ip": public_ip}

    def _ssh_run(self, public_ip: str, ssh_user: str, script: str) -> None:
        """Run a shell script on the VM over SSH and wait for it to finish."""
        ssh_run(public_ip, ssh_user, self.ssh_privkey_path, script)

    def _provision_docker_service(
        self, resource: "DockerServiceConfig", image_name: str, version: str = "1.0.0"
    ) -> str:
        """Bootstrap a Docker-host gallery image for a DockerServiceConfig.

        Pipeline (idempotent at every step):
            Marketplace Ubuntu 22.04 VM
                → install Docker + docker pull all images (via cloud-init)
                → sentinel blob written when ready
            VM deallocated → OS disk retained (delete_option=Detach)
            OS disk → gallery image definition + version
            Bootstrap VM + OS disk deleted

        Returns the gallery image version resource ID.
        """
        sentinel_name = f"{image_name}.docker_bootstrap_done"
        failed_name = f"{image_name}.docker_bootstrap_failed"

        logger.info("_provision_docker_service: %s  images=%s", image_name, resource.docker_images)

        if not self.blob_exists(sentinel_name):
            sentinel_sas = self.generate_sas_url(sentinel_name, expiry_hours=8, write=True)
            failed_sas = self.generate_sas_url(failed_name, expiry_hours=8, write=True)
            pull_cmds = "\n".join(
                f"echo '[bootstrap] Pulling {img}...'\ndocker pull {img}" for img in resource.docker_images
            )
            volume_cmds = build_volume_setup_script(resource.volumes)
            script = _DOCKER_BOOTSTRAP_SCRIPT.format(
                docker_pull_commands=pull_cmds,
                volume_setup_commands=volume_cmds,
                sentinel_sas_url=sentinel_sas,
                failed_sas_url=failed_sas,
            )
            disk_name = f"cube-dockerhost-disk-{image_name}"
            # If a stale disk exists from a previous interrupted run (sentinel absent
            # but disk persisted via delete_option=Detach), delete it so the new VM
            # can be created with create_option=FromImage.
            compute = self._compute()
            try:
                from azure.core.exceptions import ResourceNotFoundError

                existing = compute.disks.get(self.resource_group, disk_name)
                if existing:
                    logger.info("_provision_docker_service: deleting stale disk %s (no sentinel)", disk_name)
                    compute.disks.begin_delete(self.resource_group, disk_name).result()
            except ResourceNotFoundError:
                pass  # disk doesn't exist — expected
            vm_info = self._launch_docker_host_vm(script, disk_name)
            t0 = time.time()
            try:
                logger.info("_provision_docker_service: VM running, streaming logs from %s", vm_info["public_ip"])
                logger.info(
                    "_provision_docker_service: SSH: ssh -i %s cube@%s",
                    self.ssh_privkey_path,
                    vm_info["public_ip"],
                )
                with BootstrapMonitor(
                    public_ip=vm_info["public_ip"],
                    ssh_privkey=self.ssh_privkey_path,
                    ssh_user="cube",
                    sentinel_fn=lambda: self.blob_exists(sentinel_name),
                ) as monitor:
                    monitor.wait(timeout=3600)
            finally:
                # Deallocate (not delete) so the OS disk is retained.
                compute = self._compute()
                logger.info("_provision_docker_service: deallocating %s", vm_info["vm_name"])
                compute.virtual_machines.begin_deallocate(self.resource_group, vm_info["vm_name"]).result()
                # Now safe to delete the VM resource; delete_option=Detach keeps the disk.
                compute.virtual_machines.begin_delete(self.resource_group, vm_info["vm_name"]).result()
                self._delete_network_resources(vm_info["pip_name"], vm_info["nic_name"])
            logger.info(
                "_provision_docker_service: Docker images pulled in %.1f min",
                (time.time() - t0) / 60,
            )
        else:
            logger.info("_provision_docker_service: sentinel exists — skipping VM phase")
            disk_name = f"cube-dockerhost-disk-{image_name}"

        self._create_image_definition(image_name)
        image_id = self._create_image_version(image_name, version, disk_name)
        logger.info("_provision_docker_service: gallery image ready: %s/%s", image_name, version)

        # Gallery image has its own storage — source disk is no longer needed.
        try:
            self._compute().disks.begin_delete(self.resource_group, disk_name).result()
            logger.info("_provision_docker_service: deleted source disk %s", disk_name)
        except Exception as exc:
            logger.warning("_provision_docker_service: could not delete disk %s: %s", disk_name, exc)

        # Clean up sentinel blobs.
        for b in (sentinel_name, failed_name):
            self._delete_blob(b)

        return image_id

    def _launch_docker_host_vm(self, script: str, disk_name: str) -> dict:
        """Launch a gallery-image Ubuntu VM for Docker image bootstrapping.

        Uses the same bootstrap_gallery_image as _launch_bootstrap_vm (subscription
        policy requires gallery images — marketplace images are blocked).
        Uses cloud-init (custom_data) to run the bootstrap script.
        OS disk is named explicitly and created with delete_option=Detach so it
        persists after VM deletion for snapshotting into a gallery image.

        Returns {vm_name, pip_name, nic_name, public_ip}.
        """
        uid = uuid.uuid4().hex[:6]
        vm_name = f"cube-dockerhost-{uid}"
        pubkey = Path(self.ssh_pubkey_path).read_text().strip()
        custom_data_b64 = base64.b64encode(script.encode()).decode()
        compute = self._compute()

        logger.info("_launch_docker_host_vm: creating network resources")
        pip, nic, pip_name, nic_name = self._create_network_resources(uid, uid)

        image_id = (
            f"/subscriptions/{self.subscription}/resourceGroups/{self.resource_group}"
            f"/providers/Microsoft.Compute/galleries/{self.gallery_name}"
            f"/images/{self.bootstrap_gallery_image}"
            f"/versions/{self.bootstrap_gallery_image_ver}"
        )
        logger.info(
            "_launch_docker_host_vm: launching %s (%s, 64 GB OS disk)  image=%s/%s",
            vm_name,
            self.bootstrap_vm_size,
            self.bootstrap_gallery_image,
            self.bootstrap_gallery_image_ver,
        )
        t0 = time.time()
        poller = compute.virtual_machines.begin_create_or_update(  # type: ignore[call-overload]
            self.resource_group,
            vm_name,
            {  # type: ignore[arg-type]
                "location": self.location,
                "tags": {**self.tags, "role": "docker-bootstrap"},
                "hardware_profile": {"vm_size": self.bootstrap_vm_size},
                "storage_profile": {
                    "image_reference": {"id": image_id},
                    "os_disk": {
                        "name": disk_name,
                        "create_option": "FromImage",
                        "managed_disk": {"storage_account_type": self.bootstrap_disk_sku},
                        "disk_size_gb": 64,
                        # Detach keeps the disk after VM deletion — required for snapshotting.
                        "delete_option": "Detach",
                    },
                },
                "os_profile": {
                    "computer_name": vm_name,
                    "admin_username": "cube",
                    "custom_data": custom_data_b64,
                    "linux_configuration": {
                        "disable_password_authentication": True,
                        "ssh": {
                            "public_keys": [
                                {
                                    "path": "/home/cube/.ssh/authorized_keys",
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

        pip_info = self._network().public_ip_addresses.get(self.resource_group, pip_name)
        assert pip_info.ip_address
        public_ip = pip_info.ip_address
        logger.info(
            "_launch_docker_host_vm: VM ready in %ds: %s @ %s",
            int(time.time() - t0),
            vm_name,
            public_ip,
        )
        logger.info("_launch_docker_host_vm: SSH: ssh -i %s cube@%s", self.ssh_privkey_path, public_ip)
        return {"vm_name": vm_name, "pip_name": pip_name, "nic_name": nic_name, "public_ip": public_ip}

    def _delete_network_resources(self, pip_name: str, nic_name: str) -> None:
        """Delete a NIC and public IP by name (best-effort, logs warnings on failure)."""
        network = self._network()
        for resource_type, name, delete_fn in [
            ("NIC", nic_name, lambda: network.network_interfaces.begin_delete(self.resource_group, nic_name).result()),
            ("IP", pip_name, lambda: network.public_ip_addresses.begin_delete(self.resource_group, pip_name).result()),
        ]:
            if not name:
                continue
            try:
                delete_fn()
                logger.debug("_delete_network_resources: deleted %s %s", resource_type, name)
            except Exception as exc:
                logger.warning("_delete_network_resources: could not delete %s %s: %s", resource_type, name, exc)

    def _read_blob_text(self, blob_name: str, default: str = "linux") -> str:
        """Read a small text blob and return its content stripped. Returns default on error."""
        try:
            svc = self._blob_service_client()
            blob = svc.get_blob_client(self.container_name, blob_name)
            return blob.download_blob().readall().decode().strip()
        except Exception as exc:
            logger.warning("_read_blob_text: could not read %s: %s — defaulting to %r", blob_name, exc, default)
            return default

    def _bootstrap(
        self,
        url: str,
        image_name: str,
        version: str = "1.0.0",
        uefi: bool = False,
        trusted_launch: bool = False,
        specialized: bool = False,
    ) -> str:
        """In-cloud bootstrap: spin up Azure VM to download, convert, and upload the image.

        Idempotent — skips the VM phase if the sentinel blob already exists.
        Returns the gallery image version resource ID.
        """
        blob_name = image_name + ".vhd"
        sentinel_name = blob_name + ".bootstrap_done"
        failed_name = blob_name + ".bootstrap_failed"
        os_type_blob_name = image_name + ".os_type"

        logger.info("_bootstrap: %s  source=%s", image_name, url)
        logger.info("_bootstrap: blob=%s", blob_name)

        if not self.blob_exists(sentinel_name):
            vhd_sas_url = self.generate_sas_url(blob_name, expiry_hours=8, write=True)
            sentinel_sas_url = self.generate_sas_url(sentinel_name, expiry_hours=8, write=True)
            failed_sas_url = self.generate_sas_url(failed_name, expiry_hours=8, write=True)
            os_type_sas_url = self.generate_sas_url(os_type_blob_name, expiry_hours=8, write=True)
            cache_sas_url = (
                self.generate_sas_url(self.source_cache_blob, expiry_hours=8, write=False)
                if self.source_cache_blob and self.blob_exists(self.source_cache_blob)
                else ""
            )
            if cache_sas_url:
                logger.info("_bootstrap: source cache found — skipping HuggingFace download")
            script = _AZURE_BOOTSTRAP_SCRIPT.format(
                hf_url=url,
                cache_sas_url=cache_sas_url,
                vhd_sas_url=vhd_sas_url,
                sentinel_sas_url=sentinel_sas_url,
                failed_sas_url=failed_sas_url,
                os_type_sas_url=os_type_sas_url,
                winrm_password=self.windows_admin_password or "",
                specialized="true" if specialized else "false",
            )
            vm_info = self._launch_bootstrap_vm(script)
            t0 = time.time()
            logger.info("_bootstrap: VM running, streaming logs from %s", vm_info["public_ip"])
            logger.info("_bootstrap: SSH: ssh -i %s azureuser@%s", self.ssh_privkey_path, vm_info["public_ip"])
            try:
                with BootstrapMonitor(
                    public_ip=vm_info["public_ip"],
                    ssh_privkey=self.ssh_privkey_path,
                    ssh_user="azureuser",
                    sentinel_fn=lambda: self.blob_exists(sentinel_name),
                ) as monitor:
                    monitor.wait(timeout=7200)
            except TimeoutError:
                logger.error("_bootstrap: timed out — VM kept alive for debugging")
                logger.error("_bootstrap: SSH:  ssh -i %s azureuser@%s", self.ssh_privkey_path, vm_info["public_ip"])
                logger.error("_bootstrap: logs: sudo tail -f /var/log/cube-bootstrap.log")
                logger.error("_bootstrap: qemu: sudo tail -f /tmp/qemu-serial.log")
                logger.error(
                    "_bootstrap: delete when done: INFRA._delete_vm(%r, %r, %r)",
                    vm_info["vm_name"],
                    vm_info["pip_name"],
                    vm_info["nic_name"],
                )
                raise
            except Exception:
                self._delete_vm(vm_info["vm_name"], vm_info["pip_name"], vm_info["nic_name"])
                raise
            self._delete_vm(vm_info["vm_name"], vm_info["pip_name"], vm_info["nic_name"])
            logger.info("_bootstrap: VHD ready in blob storage (%.1f min)", (time.time() - t0) / 60)
        else:
            logger.info("_bootstrap: sentinel exists — skipping VM phase")

        os_type = self._read_blob_text(os_type_blob_name)
        logger.info("_bootstrap: os_type=%s", os_type)

        return self._ensure_resource_from_blob(
            blob_name,
            image_name,
            version,
            os_type=os_type,
            uefi=uefi,
            trusted_launch=trusted_launch,
            specialized=specialized,
        )

    def list_images(self) -> list[dict]:
        """Return all image definitions in the gallery (informational)."""
        compute = self._compute()
        try:
            defs = list(compute.gallery_images.list_by_gallery(self.resource_group, self.gallery_name))
        except Exception:
            logger.warning("list_images: gallery '%s' not found", self.gallery_name)
            return []

        result = []
        for d in defs:
            entry: dict = {
                "name": d.name,
                "os_state": d.os_state,
                "hyper_v_generation": d.hyper_v_generation,
                "versions": [],
            }
            if d.name:
                versions = list(
                    compute.gallery_image_versions.list_by_gallery_image(self.resource_group, self.gallery_name, d.name)
                )
                entry["versions"] = [{"version": v.name, "state": v.provisioning_state} for v in versions]
            result.append(entry)
        return result
