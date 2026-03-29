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

Launch (~3-5 min per VM):
    Gallery image version
        → NIC + public IP
        → VM (Specialized — no cloud-init)
        → SSH tunnel localhost:{port} → VM:{guest_port}
        → AzureResourceHandle(endpoint="http://localhost:{port}")

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
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, model_validator

from cube.resource import (
    InfraConfig,
    ResourceConfig,
    ResourceHandle,
    ResourceNotReadyError,
    UnsupportedResourceType,
    VMResourceConfig,
)
from cube_infra_azure._utils import BootstrapMonitor, free_port, open_tunnel, wait_for_ssh

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ── Bootstrap script ───────────────────────────────────────────────────────────
# Placeholders: {hf_url}, {vhd_sas_url}, {sentinel_sas_url}, {failed_sas_url},
#               {ssh_pubkey}

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

# ── inject SSH into VHD ────────────────────────────────────────────────────────
echo "[bootstrap] Injecting SSH into VHD..."
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
which sshd 2>/dev/null || (apt-get update -qq && apt-get install -y -qq openssh-server)
ls /etc/ssh/ssh_host_*_key 2>/dev/null | grep -q . || ssh-keygen -A
rm -f /etc/ssh/sshd_not_to_be_run
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
SSH_PUBKEY='{ssh_pubkey}'
for USER_HOME in /mnt/guest/home/user /mnt/guest/home/ubuntu /mnt/guest/root; do
    [ -d "$USER_HOME" ] || continue
    mkdir -p "$USER_HOME/.ssh"
    grep -qxF "$SSH_PUBKEY" "$USER_HOME/.ssh/authorized_keys" 2>/dev/null \\
        || echo "$SSH_PUBKEY" >> "$USER_HOME/.ssh/authorized_keys"
    chmod 700 "$USER_HOME/.ssh"
    chmod 600 "$USER_HOME/.ssh/authorized_keys"
    OWNER=$(stat -c '%U' "$USER_HOME" 2>/dev/null || echo "root")
    chown -R "$OWNER:$OWNER" "$USER_HOME/.ssh" 2>/dev/null || true
done
for fs in run sys proc dev/pts dev; do umount "/mnt/guest/$fs" 2>/dev/null || true; done
umount /mnt/guest
losetup -d "$LOOP" 2>/dev/null || true
echo "[bootstrap] SSH injection done"

# ── upload ────────────────────────────────────────────────────────────────────
echo "[bootstrap] Uploading to Azure Blob Storage..."
azcopy copy /data/output.vhd "{vhd_sas_url}" --blob-type PageBlob
echo "[bootstrap] Upload complete"

# ── signal done ───────────────────────────────────────────────────────────────
curl -s -X PUT -H "x-ms-blob-type: BlockBlob" -H "Content-Length: 0" "{sentinel_sas_url}"
echo "[bootstrap] Done at $(date)"
"""


# ── AzureResourceHandle ───────────────────────────────────────────────────────


@dataclass
class AzureResourceHandle(ResourceHandle):
    """ResourceHandle for a running Azure VM with an open SSH tunnel."""

    _vm_name: str = field(default="", repr=False)
    _pip_name: str = field(default="", repr=False)
    _nic_name: str = field(default="", repr=False)
    _tunnel: subprocess.Popen | None = field(default=None, repr=False)

    def close(self) -> None:
        """Terminate the SSH tunnel and delete the Azure VM + network resources."""
        if self._tunnel is not None:
            try:
                self._tunnel.terminate()
            except Exception:
                pass
            self._tunnel = None
            logger.info("SSH tunnel closed for run %s", self.run_id[:8])

        if self._vm_name:
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

    ── Required ──────────────────────────────────────────────────────────────
    resource_group  str
        Resource group for all managed Azure resources.
        All other auto-discovered fields are scoped to this resource group.

    ── Auto-discovered (leave as None unless there are multiple in the RG) ──
    subscription    str | None = None
        Azure subscription ID.  Auto-populated via ``az account show``.
    storage_account str | None = None
        Storage account used for intermediate VHD blobs during provisioning.
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
        if not set.  Its *content* is read once during provisioning and injected
        into the VHD (bootstrap script writes it to authorized_keys).
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

    ── Testing ───────────────────────────────────────────────────────────────
    image_name_suffix   str = ""
        Suffix appended to gallery image definition names and ProvisionStore
        keys.  Use "-test" to isolate CI/test runs from manually-created
        production images without touching the production ProvisionStore entry.
        E.g. image_name_suffix="-test" → gallery def "osworld-ubuntu-vm-test",
        key "osworld-ubuntu-vm-test@azure:westus2".
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

    # ── Testing ───────────────────────────────────────────────────────────────
    image_name_suffix: str = ""

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
        needs_azure_discovery = not all([
            self.subscription, self.storage_account,
            self.vnet_name, self.subnet_name, self.nsg_name,
        ])
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
                raise ValueError(
                    f"Multiple {kind} in '{rg}': {names}.\n"
                    f"Set {param}= explicitly."
                )

            if not self.subscription:
                result = subprocess.run(
                    ["az", "account", "show", "--query", "id", "-o", "tsv"],
                    capture_output=True, text=True,
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
                object.__setattr__(self, "storage_account", _pick(
                    list(sc.storage_accounts.list_by_resource_group(rg)),
                    "storage accounts", "storage_account",
                ))

            if not self.vnet_name:
                object.__setattr__(self, "vnet_name", _pick(
                    list(nc.virtual_networks.list(rg)), "VNets", "vnet_name",
                ))

            if not self.subnet_name:
                object.__setattr__(self, "subnet_name", _pick(
                    list(nc.subnets.list(rg, self.vnet_name)), "subnets", "subnet_name",
                ))

            if not self.nsg_name:
                object.__setattr__(self, "nsg_name", _pick(
                    list(nc.network_security_groups.list(rg)), "NSGs", "nsg_name",
                ))

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

    def _image_name(self, resource: VMResourceConfig) -> str:
        """Gallery image definition name (resource.name + image_name_suffix)."""
        return resource.name + self.image_name_suffix

    def _resource_shim(self, resource: VMResourceConfig) -> Any:
        """Minimal object with .name = _image_name(resource) for ProvisionStore keys."""
        import types
        return types.SimpleNamespace(name=self._image_name(resource))

    def fingerprint(self) -> str:
        """Stable key: provider + region only (not VM size or storage SKU).

        Two AzureInfraConfig objects with the same subscription/location share
        the same provisioned gallery images.
        """
        return f"azure:{self.location}"

    def capabilities(self) -> set[str]:
        """Azure can satisfy any VMResourceConfig (native hypervisor = KVM-equivalent)."""
        return {"kvm"}

    def provision_status(self, resource: ResourceConfig) -> Literal["ready", "needs_provisioning"]:
        """Query ProvisionStore using the effective image name (respects image_name_suffix)."""
        from cube.provision_store import ProvisionStore

        if not isinstance(resource, VMResourceConfig):
            return "needs_provisioning"
        shim = self._resource_shim(resource)
        store = ProvisionStore()
        return "ready" if store.get(shim, self) is not None else "needs_provisioning"

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
        if not isinstance(resource, VMResourceConfig):
            raise UnsupportedResourceType(resource, self)

        from cube.provision_store import ProvisionStore

        shim = self._resource_shim(resource)
        image_name = self._image_name(resource)
        store = ProvisionStore()
        existing = store.get(shim, self)
        if existing:
            logger.info(
                "provision: %r already registered for %s — skipping",
                image_name, self.fingerprint(),
            )
            return

        if not resource.source_url:
            raise ValueError(
                f"Cannot provision {image_name!r}: no source_url set and "
                f"no registration found for {self.fingerprint()!r}.\n"
                f"  Manual: infra.register(resource, {{\"image_def\": ..., \"version\": ...}})"
            )

        version = "1.0.0"
        logger.info(
            "provision: bootstrapping %r → gallery image (version %s)",
            image_name, version,
        )
        image_id = self._bootstrap(
            url=resource.source_url,
            image_name=image_name,
            version=version,
        )
        store.put(shim, self, {
            "image_def": image_name,
            "version": version,
            "image_id": image_id,
        })
        logger.info("provision: %r registered for %s", image_name, self.fingerprint())

    def unprovision(self, resource: ResourceConfig) -> None:
        """Delete the gallery image version, VHD blob, sentinel, and ProvisionStore entry.

        Safe to call when not provisioned — no-ops if not registered.

        Raises:
            UnsupportedResourceType: if resource is not VMResourceConfig.
        """
        if not isinstance(resource, VMResourceConfig):
            raise UnsupportedResourceType(resource, self)

        from cube.provision_store import ProvisionStore

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
            logger.warning(
                "unprovision: could not delete gallery image %s/%s: %s", image_def, version, exc
            )

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
        if not isinstance(resource, VMResourceConfig):
            raise UnsupportedResourceType(resource, self)

        from cube.provision_store import ProvisionStore

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
        effective_ttl = self.default_ttl_seconds if self.default_ttl_seconds is not None else resource.default_ttl_seconds
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
        pip, nic, pip_name, nic_name = self._create_network_resources(
            run_id_short, uid, cube_tags
        )

        # VM tags also record NIC/IP names so list_active() needs no string manipulation.
        vm_tags = {
            **self.tags,
            **cube_tags,
            "cube:nic_name": nic_name,
            "cube:ip_name": pip_name,
        }
        logger.info(
            "launch: creating VM %s (%s)  image=%s/%s", vm_name, self.vm_size, image_def, version
        )
        t0 = time.time()

        # Specialized gallery image: no os_profile allowed.
        # ARM completes once the infrastructure is provisioned, without waiting
        # for waagent or cloud-init to signal back from the guest.
        poller = compute.virtual_machines.begin_create_or_update(  # type: ignore[call-overload]
            self.resource_group,
            vm_name,
            {  # type: ignore[arg-type]
                "location": self.location,
                "tags": vm_tags,
                "hardware_profile": {"vm_size": self.vm_size},
                "storage_profile": {
                    "image_reference": {"id": image_id},
                    "os_disk": {
                        "create_option": "FromImage",
                        "managed_disk": {"storage_account_type": "Standard_LRS"},
                        "delete_option": "Delete",
                    },
                },
                "network_profile": {
                    "network_interfaces": [{"id": nic.id, "properties": {"primary": True}}]
                },
            },
        )
        poller.result()
        elapsed = time.time() - t0

        pip_info = self._network().public_ip_addresses.get(self.resource_group, pip_name)
        assert pip_info.ip_address, "Public IP address was not assigned"
        public_ip = pip_info.ip_address
        logger.info("launch: VM ready in %.0fs: %s @ %s", elapsed, vm_name, public_ip)

        # SSH + tunnel — clean up VM on any failure to avoid orphaned resources.
        try:
            logger.info("launch: waiting for SSH on %s…", public_ip)
            active_user = wait_for_ssh(
                public_ip, "user", self.ssh_privkey_path,
                fallback_users=["ubuntu", "root"],
                timeout=600,  # OSWorld VM takes ~5-8 min to boot
            )

            local_port = free_port()
            logger.info(
                "launch: opening tunnel localhost:%d → %s:%d",
                local_port, public_ip, self.guest_port,
            )
            tunnel = open_tunnel(
                public_ip, active_user, self.ssh_privkey_path, local_port, self.guest_port
            )
        except Exception:
            logger.warning("launch: SSH/tunnel failed — cleaning up VM %s", vm_name)
            self._delete_vm(vm_name, pip_name, nic_name)
            raise

        endpoint = f"http://localhost:{local_port}"

        return AzureResourceHandle(
            run_id=run_id,
            resource=resource,
            infra=self,
            endpoint=endpoint,
            created_at=created_at,
            expires_at=expires_at,
            _vm_name=vm_name,
            _pip_name=pip_name,
            _nic_name=nic_name,
            _tunnel=tunnel,
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

            handles.append(AzureResourceHandle(
                run_id=vm_run_id,
                resource=VMResourceConfig(name=resource_name),
                infra=self,
                endpoint=None,  # tunnel cannot be reconstructed
                _vm_name=vm_name,
                _pip_name=pip_name,
                _nic_name=nic_name,
                _tunnel=None,
            ))

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

            # Priority 1: explicit TTL tag written at launch time.
            expires_at_str = vm_tags.get("cube:expires_at")
            if expires_at_str:
                try:
                    expires_at = datetime.fromisoformat(expires_at_str)
                    if expires_at < now:
                        should_delete = True
                except ValueError:
                    logger.warning("cleanup_stale: invalid cube:expires_at %r on %s", expires_at_str, vm.name)

            # Priority 2: age-based fallback (harness startup sweep).
            if not should_delete and max_age_seconds is not None:
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
        try:
            svc = self._blob_service_client()
            svc.get_blob_client(self.container_name, blob_name).delete_blob()
            logger.info("_delete_blob: deleted %s", blob_name)
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
        return (
            f"https://{self.storage_account}.blob.core.windows.net"
            f"/{self.container_name}/{blob_name}?{sas}"
        )

    def _create_network_resources(
        self, run_id_short: str, uid: str, cube_tags: dict[str, str] | None = None
    ) -> tuple:
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
            self.resource_group, pip_name,
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
            self.resource_group, nic_name,
            {  # type: ignore[arg-type]
                "location": self.location,
                "tags": resource_tags,
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
            (network.network_interfaces.begin_delete,  "NIC", nic_name),
            (network.public_ip_addresses.begin_delete, "IP",  pip_name),
        ]:
            try:
                fn(self.resource_group, name).result()
                logger.info("_delete_vm: %s deleted: %s", label, name)
            except Exception:
                pass

    # ── Provisioning internals ────────────────────────────────────────────────

    def _import_disk(self, blob_url: str, disk_name: str) -> str:
        """Create a Managed Disk from a VHD blob. Returns the disk name.

        Always deletes any existing disk first (import is a no-op if disk exists).
        """
        logger.info("_import_disk: %s → %s", blob_url.split("/")[-1].split("?")[0], disk_name)
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
                    "osType": "Linux",
                },
            },
        )
        disk = poller.result()
        logger.info(
            "_import_disk: done in %.0fs: %s (%s GB)",
            time.time() - t0, disk_name, disk.disk_size_gb,
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

    def _create_image_definition(self, name: str, os_state: str = "Specialized") -> str:
        """Create a gallery image definition (idempotent). Returns definition name."""
        self._ensure_gallery()
        compute = self._compute()
        try:
            compute.gallery_images.get(self.resource_group, self.gallery_name, name)
            logger.info("_create_image_definition: %s already exists", name)
            return name
        except Exception:
            pass

        logger.info("_create_image_definition: %s (%s, HyperV V1)", name, os_state)
        compute.gallery_images.begin_create_or_update(  # type: ignore[call-overload]
            self.resource_group,
            self.gallery_name,
            name,
            {  # type: ignore[arg-type]
                "location": self.location,
                "tags": self.tags,
                "os_type": "Linux",
                "os_state": os_state,
                "hyper_v_generation": "V1",
                "identifier": {"publisher": "cube", "offer": name, "sku": "linux"},
            },
        ).result()
        logger.info("_create_image_definition: created %s", name)
        return name

    def _create_image_version(self, image_def: str, version: str, disk_name: str) -> str:
        """Publish a Managed Disk as a gallery image version (idempotent).

        Returns the full gallery image version resource ID.
        """
        compute = self._compute()
        try:
            existing = compute.gallery_image_versions.get(
                self.resource_group, self.gallery_name, image_def, version
            )
            if existing.provisioning_state == "Succeeded":
                logger.info("_create_image_version: %s/%s already exists", image_def, version)
                return existing.id or ""
        except Exception:
            pass

        disk = compute.disks.get(self.resource_group, disk_name)
        logger.info(
            "_create_image_version: publishing %s/%s from %s (%s GB)…",
            image_def, version, disk_name, disk.disk_size_gb,
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
                    "target_regions": [{
                        "name": self.location,
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
                },
            },
        )
        version_obj = poller.result()
        logger.info(
            "_create_image_version: done in %.0fs: %s", time.time() - t0, version_obj.id
        )
        return version_obj.id or ""

    def _ensure_resource_from_blob(self, vhd_blob_name: str, name: str, version: str = "1.0.0") -> str:
        """Import VHD blob → disk → gallery image definition + version.

        Idempotent at each step. Returns the gallery image version resource ID.
        """
        blob_url = (
            f"https://{self.storage_account}.blob.core.windows.net"
            f"/{self.container_name}/{vhd_blob_name}"
        )
        disk_name = f"cube-disk-{name}"
        self._import_disk(blob_url, disk_name)
        self._create_image_definition(name)
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
            vm_name, self.bootstrap_vm_size, self.bootstrap_os_disk_gb,
        )
        t0 = time.time()
        poller = compute.virtual_machines.begin_create_or_update(  # type: ignore[call-overload]
            self.resource_group, vm_name,
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

        pip_info = self._network().public_ip_addresses.get(self.resource_group, pip_name)
        assert pip_info.ip_address
        public_ip = pip_info.ip_address
        logger.info(
            "_launch_bootstrap_vm: VM ready in %ds: %s @ %s",
            int(time.time() - t0), vm_name, public_ip,
        )
        logger.info("_launch_bootstrap_vm: SSH: ssh -i %s azureuser@%s", self.ssh_privkey_path, public_ip)
        return {"vm_name": vm_name, "pip_name": pip_name, "nic_name": nic_name, "public_ip": public_ip}

    def _bootstrap(self, url: str, image_name: str, version: str = "1.0.0") -> str:
        """In-cloud bootstrap: spin up Azure VM to download, convert, and upload the image.

        Idempotent — skips the VM phase if the sentinel blob already exists.
        Returns the gallery image version resource ID.
        """
        blob_name = image_name + ".vhd"
        sentinel_name = blob_name + ".bootstrap_done"
        failed_name = blob_name + ".bootstrap_failed"

        logger.info("_bootstrap: %s  source=%s", image_name, url)
        logger.info("_bootstrap: blob=%s", blob_name)

        if not self.blob_exists(sentinel_name):
            vhd_sas_url      = self.generate_sas_url(blob_name,      expiry_hours=8, write=True)
            sentinel_sas_url = self.generate_sas_url(sentinel_name,  expiry_hours=8, write=True)
            failed_sas_url   = self.generate_sas_url(failed_name,    expiry_hours=8, write=True)
            script = _AZURE_BOOTSTRAP_SCRIPT.format(
                hf_url=url,
                vhd_sas_url=vhd_sas_url,
                sentinel_sas_url=sentinel_sas_url,
                failed_sas_url=failed_sas_url,
                ssh_pubkey=Path(self.ssh_pubkey_path).read_text().strip(),
            )
            vm_info = self._launch_bootstrap_vm(script)
            t0 = time.time()
            try:
                logger.info(
                    "_bootstrap: VM running, streaming logs from %s", vm_info["public_ip"]
                )
                logger.info(
                    "_bootstrap: SSH: ssh -i %s azureuser@%s", self.ssh_privkey_path, vm_info["public_ip"]
                )
                with BootstrapMonitor(
                    public_ip=vm_info["public_ip"],
                    ssh_privkey=self.ssh_privkey_path,
                    ssh_user="azureuser",
                    sentinel_fn=lambda: self.blob_exists(sentinel_name),
                ) as monitor:
                    monitor.wait(timeout=7200)
            finally:
                self._delete_vm(vm_info["vm_name"], vm_info["pip_name"], vm_info["nic_name"])
            logger.info("_bootstrap: VHD ready in blob storage (%.1f min)", (time.time() - t0) / 60)
        else:
            logger.info("_bootstrap: sentinel exists — skipping VM phase")

        return self._ensure_resource_from_blob(blob_name, image_name, version)

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
                    compute.gallery_image_versions.list_by_gallery_image(
                        self.resource_group, self.gallery_name, d.name
                    )
                )
                entry["versions"] = [
                    {"version": v.name, "state": v.provisioning_state}
                    for v in versions
                ]
            result.append(entry)
        return result
