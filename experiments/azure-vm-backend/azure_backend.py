"""Azure VM backend for CUBE experiments."""
from __future__ import annotations

import base64
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from azure.identity import AzureCliCredential
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.network import NetworkManagementClient
from azure.mgmt.storage import StorageManagementClient
from azure.storage.blob import BlobSasPermissions, BlobServiceClient, generate_blob_sas

from _common import BootstrapMonitor, convert_image, free_port, open_tunnel as ssh_tunnel, probe, wait_for_ssh

log = logging.getLogger(__name__)

# ── Bootstrap script ──────────────────────────────────────────────────────────
# Placeholders: {hf_url}, {vhd_sas_url}, {sentinel_sas_url}, {failed_sas_url}

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

# ── upload ────────────────────────────────────────────────────────────────────
echo "[bootstrap] Uploading to Azure Blob Storage..."
azcopy copy /data/output.vhd "{vhd_sas_url}" --blob-type PageBlob
echo "[bootstrap] Upload complete"

# ── signal done ───────────────────────────────────────────────────────────────
curl -s -X PUT -H "x-ms-blob-type: BlockBlob" -H "Content-Length: 0" "{sentinel_sas_url}"
echo "[bootstrap] Done at $(date)"
"""


# ── Backend class ─────────────────────────────────────────────────────────────

@dataclass
class AzureBackend:
    """Azure VM backend: full lifecycle from image import to running VM."""

    subscription:                str  = "aeb958d3-a614-450e-94bc-88f284dc0664"
    resource_group:              str  = "ui_assist"
    location:                    str  = "westus2"
    storage_account:             str  = "cubeexpvhd"
    container_name:              str  = "vhds"
    vnet_name:                   str  = "vnet-westus2"
    subnet_name:                 str  = "snet-westus2-1"
    nsg_name:                    str  = "osworld-nsg"
    gallery_name:                str  = "cube_exp_gallery"
    vm_size:                     str  = "Standard_D4s_v3"
    guest_port:                  int  = 5000
    tags:                        dict = field(default_factory=lambda: {"project": "cube-experiment"})
    ssh_privkey:                 str  = field(default_factory=lambda: str(Path.home() / ".ssh" / "id_ed25519"))
    ssh_pubkey:                  str  = field(default_factory=lambda: str(Path.home() / ".ssh" / "id_ed25519.pub"))
    bootstrap_vm_size:           str  = "Standard_B2ms"
    bootstrap_gallery_image:     str  = "cube-ubuntu-22-04"
    bootstrap_gallery_image_ver: str  = "1.0.0"
    bootstrap_os_disk_gb:        int  = 128

    # ── Private Azure clients ─────────────────────────────────────────────────

    def _cred(self) -> AzureCliCredential:
        return AzureCliCredential()

    def _compute(self) -> ComputeManagementClient:
        return ComputeManagementClient(self._cred(), self.subscription)

    def _network(self) -> NetworkManagementClient:
        return NetworkManagementClient(self._cred(), self.subscription)

    def _storage(self) -> StorageManagementClient:
        return StorageManagementClient(self._cred(), self.subscription)

    def _create_network_resources(self, uid: str) -> tuple:
        """Create a static public IP and NIC for a new VM. Returns (pip, nic, pip_name, nic_name).

        Both launch() and launch_bootstrap_vm() need this — extracted to avoid duplication.
        The uid suffix makes names unique per VM. Resources are tagged and wired to the
        configured VNet/subnet and NSG.
        """
        network = self._network()
        pip_name = f"cube-ip-{uid}"
        nic_name = f"cube-nic-{uid}"

        pip = network.public_ip_addresses.begin_create_or_update(  # type: ignore[call-overload]
            self.resource_group, pip_name,
            {  # type: ignore[arg-type]
                "location": self.location,
                "tags": self.tags,
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
                "tags": self.tags,
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

    # ── Storage helpers ───────────────────────────────────────────────────────

    def _get_storage_key(self) -> str:
        """Get primary storage account key, creating the account if needed."""
        storage = self._storage()
        try:
            storage.storage_accounts.get_properties(self.resource_group, self.storage_account)
        except Exception:
            log.info("Creating storage account: %s", self.storage_account)
            poller = storage.storage_accounts.begin_create(  # type: ignore[call-overload]
                self.resource_group,
                self.storage_account,
                {  # type: ignore[arg-type]
                    "location": self.location,
                    "tags": self.tags,
                    "sku": {"name": "Standard_LRS"},
                    "kind": "StorageV2",
                },
            )
            poller.result()
        keys = storage.storage_accounts.list_keys(self.resource_group, self.storage_account)
        assert keys.keys, "Storage account returned no keys"
        return keys.keys[0].value  # type: ignore[return-value]

    def _blob_service_client(self) -> BlobServiceClient:
        """Return a BlobServiceClient using storage account key auth."""
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

    def upload_vhd(self, vhd_path: Path, blob_name: str | None = None) -> str:
        """Upload a fixed VHD to Azure Blob Storage as a PageBlob.

        Validates the VHD footer ('conectix' magic in last 512 bytes) before
        skipping re-upload — a partial/interrupted upload leaves a blob at the
        correct size but with a zeroed footer, which would cause import to fail
        silently. Deletes and re-uploads if the footer is invalid.

        Progress is logged every 512 MB (can take 60-90 min for a 50 GB image on
        home broadband; use bootstrap() for in-cloud speed).
        Returns the blob URL.
        """
        vhd = vhd_path.resolve()
        blob_name = blob_name or vhd.name
        size_gb = vhd.stat().st_size / 1024**3

        log.info("upload_vhd: %s (%.1f GB) → %s/%s", vhd.name, size_gb, self.storage_account, blob_name)

        svc = self._blob_service_client()
        container = svc.get_container_client(self.container_name)
        try:
            container.get_container_properties()
        except Exception:
            container.create_container()

        # Check if already uploaded — validate both size AND VHD footer magic.
        blob_client = svc.get_blob_client(self.container_name, blob_name)
        try:
            props = blob_client.get_blob_properties()
            if props.size == vhd.stat().st_size:
                footer_offset = props.size - 512
                footer_data = blob_client.download_blob(offset=footer_offset, length=512).readall()
                if footer_data[:8] == b"conectix":
                    blob_url = (
                        f"https://{self.storage_account}.blob.core.windows.net"
                        f"/{self.container_name}/{blob_name}"
                    )
                    log.info("upload_vhd: already uploaded (footer valid) — skipping")
                    return blob_url
                else:
                    log.info("upload_vhd: blob exists but footer is corrupt — deleting and re-uploading")
                    blob_client.delete_blob()
        except Exception:
            pass

        log.info("upload_vhd: uploading (50 GB takes ~60-90 min — progress every 512 MB)")
        t0 = time.time()
        uploaded = [0]

        def _progress(current: int, total: int) -> None:
            gb = current / 1024**3
            total_gb = total / 1024**3
            elapsed = time.time() - t0
            rate = gb / (elapsed / 60) if elapsed > 0 else 0
            eta_min = (total_gb - gb) / rate if rate > 0 else 0
            if current - uploaded[0] >= 512 * 1024 * 1024 or current == total:
                uploaded[0] = current
                pct = current / total * 100
                log.debug(
                    "upload_vhd: %d%%  %.1f/%.1f GB  %.2f GB/min  ETA %.0f min",
                    pct, gb, total_gb, rate, eta_min,
                )

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
        log.info("upload_vhd: done in %.1f min (%.2f GB/min)", elapsed / 60, speed)

        return (
            f"https://{self.storage_account}.blob.core.windows.net"
            f"/{self.container_name}/{blob_name}"
        )

    # ── Disk / Gallery ────────────────────────────────────────────────────────

    def import_disk(self, blob_url: str, disk_name: str) -> str:
        """Create a Managed Disk from a VHD blob. Returns the disk name."""
        log.info("import_disk: %s → %s", blob_url.split("/")[-1].split("?")[0], disk_name)
        t0 = time.time()

        compute = self._compute()
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
        log.info("import_disk: done in %.0fs: %s (%s GB)", time.time() - t0, disk_name, disk.disk_size_gb)
        return disk_name

    def ensure_gallery(self) -> str:
        """Create Compute Gallery if it doesn't exist. Returns gallery name."""
        compute = self._compute()
        try:
            compute.galleries.get(self.resource_group, self.gallery_name)
        except Exception:
            log.info("ensure_gallery: creating %s", self.gallery_name)
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

    def create_image_definition(
        self,
        name: str,
        os_state: str = "Generalized",
        hyper_v_gen: str = "V1",
    ) -> str:
        """Create a gallery image definition. Returns image definition name."""
        self.ensure_gallery()
        compute = self._compute()
        try:
            compute.gallery_images.get(self.resource_group, self.gallery_name, name)
            log.info("create_image_definition: %s already exists", name)
            return name
        except Exception:
            pass

        log.info("create_image_definition: %s (%s, HyperV %s)", name, os_state, hyper_v_gen)
        poller = compute.gallery_images.begin_create_or_update(  # type: ignore[call-overload]
            self.resource_group,
            self.gallery_name,
            name,
            {  # type: ignore[arg-type]
                "location": self.location,
                "tags": self.tags,
                "os_type": "Linux",
                "os_state": os_state,
                "hyper_v_generation": hyper_v_gen,
                "identifier": {"publisher": "cube", "offer": name, "sku": "linux"},
            },
        )
        poller.result()
        log.info("create_image_definition: created %s", name)
        return name

    def create_image_version(self, image_def: str, version: str, disk_name: str) -> str:
        """Publish a Managed Disk as a gallery image version.

        Idempotent — skips if version already exists.
        Returns the full gallery image version ID.
        """
        compute = self._compute()
        try:
            existing = compute.gallery_image_versions.get(
                self.resource_group, self.gallery_name, image_def, version
            )
            if existing.provisioning_state == "Succeeded":
                log.info("create_image_version: %s/%s already exists", image_def, version)
                return existing.id or ""
        except Exception:
            pass

        disk = compute.disks.get(self.resource_group, disk_name)
        log.info("create_image_version: publishing %s/%s from %s (%s GB)...", image_def, version, disk_name, disk.disk_size_gb)
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
        log.info("create_image_version: done in %.0fs: %s", time.time() - t0, version_obj.id)
        return version_obj.id or ""

    def ensure_resource_from_blob(self, vhd_blob_name: str, name: str, version: str = "1.0.0") -> str:
        """Post-upload steps: blob → managed disk → gallery image definition → version.

        Called after a VHD has landed in Blob Storage — either via upload_vhd()
        (local path) or the bootstrap VM (in-cloud path). Idempotent at each step.

        Pipeline:
            Blob Storage (PageBlob VHD)
              ↓  import_disk()           — createOption: Import, ~8 min
            Managed Disk
              ↓  create_image_definition() — gallery image definition (Generalized, HyperV V1)
              ↓  create_image_version()  — publishes disk as gallery version, ~8-15 min
            Compute Gallery image        ← reused for all subsequent launch() calls

        Returns the gallery image version resource ID.
        """
        blob_url = (
            f"https://{self.storage_account}.blob.core.windows.net"
            f"/{self.container_name}/{vhd_blob_name}"
        )
        disk_name = f"cube-disk-{name}"

        self.import_disk(blob_url, disk_name)
        self.create_image_definition(name)
        image_id = self.create_image_version(name, version, disk_name)
        log.info("ensure_resource_from_blob: image ready: %s/%s", name, version)
        return image_id

    def ensure_resource(self, image_path: Path, name: str, version: str = "1.0.0") -> str:
        """One-time setup from a local image file to a Compute Gallery image version.

        Local path (slow for large images on home broadband):
            local qcow2/vmdk/vhd → convert_to_vhd() → upload_vhd() → ensure_resource_from_blob()

        For large images (>10 GB) consider bootstrap() instead, which does the
        download + convert + upload inside the cloud at datacenter speed.

        image_path : local .qcow2, .img, .vmdk, or .vhd
        Returns the gallery image version ID.
        """
        log.info("ensure_resource: %s v%s  source=%s", name, version, image_path)
        t0 = time.time()

        vhd_path = self.convert_to_vhd(image_path)
        blob_url = self.upload_vhd(vhd_path)
        blob_name = vhd_path.name
        result = self.ensure_resource_from_blob(blob_name, name, version)

        log.info("ensure_resource: done in %.1f min", (time.time() - t0) / 60)
        return result

    # ── Bootstrap VM ─────────────────────────────────────────────────────────

    def launch_bootstrap_vm(self, script: str) -> dict:
        """Launch a lightweight Ubuntu VM with a bootstrap script and a large OS disk.

        Uses our gallery image (bypasses Golden Image Policy).
        Returns {vm_name, pip_name, nic_name, public_ip}.
        """
        uid = uuid.uuid4().hex[:6]
        vm_name  = f"cube-bootstrap-{uid}"

        pubkey = Path(self.ssh_pubkey).read_text().strip()
        custom_data_b64 = base64.b64encode(script.encode()).decode()

        compute = self._compute()

        log.info("launch_bootstrap_vm: creating network resources")
        pip, nic, pip_name, nic_name = self._create_network_resources(uid)

        log.info(
            "launch_bootstrap_vm: launching %s (%s, %d GB OS disk)",
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
                            f"/images/{self.bootstrap_gallery_image}/versions/{self.bootstrap_gallery_image_ver}"
                        )
                    },
                    "os_disk": {
                        "create_option": "FromImage",
                        "managed_disk": {"storage_account_type": "Standard_LRS"},
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

        pip_info = network.public_ip_addresses.get(self.resource_group, pip_name)
        assert pip_info.ip_address
        public_ip = pip_info.ip_address
        log.info("launch_bootstrap_vm: VM ready in %ds: %s @ %s", int(time.time() - t0), vm_name, public_ip)
        log.info("launch_bootstrap_vm: SSH: ssh -i %s azureuser@%s", self.ssh_privkey, public_ip)
        log.info("launch_bootstrap_vm: Logs: ssh ... 'sudo tail -f /var/log/cube-bootstrap.log'")

        return {
            "vm_name": vm_name,
            "pip_name": pip_name,
            "nic_name": nic_name,
            "public_ip": public_ip,
        }

    def cleanup_bootstrap_vm(self, vm_name: str, pip_name: str, nic_name: str) -> None:
        """Terminate bootstrap VM and associated network resources."""
        compute = self._compute()
        network = self._network()
        log.info("cleanup_bootstrap_vm: deleting %s", vm_name)
        compute.virtual_machines.begin_delete(self.resource_group, vm_name).result()
        for fn, label, name in [
            (network.network_interfaces.begin_delete,  "NIC", nic_name),
            (network.public_ip_addresses.begin_delete, "IP",  pip_name),
        ]:
            try:
                fn(self.resource_group, name).result()
                log.info("cleanup_bootstrap_vm: %s deleted: %s", label, name)
            except Exception:
                pass
        log.info("cleanup_bootstrap_vm: done")

    def bootstrap(
        self,
        url: str,
        image_name: str,
        version: str = "1.0.0",
        blob_name: str | None = None,
    ) -> str:
        """In-cloud bootstrap: spin up a cheap Azure VM to download, convert, and upload the image.

        Faster than ensure_resource() for large remote images (e.g. 50 GB OSWorld qcow2):
        - Bootstrap VM downloads from HuggingFace at ~55 MB/s (datacenter speed)
        - Converts to fixed VHD with qemu-img
        - Uploads to Blob Storage via azcopy at ~300 Mb/s
        - Writes a sentinel blob when done; local process polls every 30s
        - Bootstrap VM terminates automatically on success or failure

        Uses BootstrapMonitor to SSH-tail /var/log/cube-bootstrap.log in real time.
        Idempotent: skips the VM phase if the sentinel blob already exists.

        The bootstrap VM uses our Compute Gallery image (cube-ubuntu-22-04) to
        bypass ServiceNow's Golden Image Policy which blocks Marketplace images.

        After the VHD is in blob storage, calls ensure_resource_from_blob() to
        import it as a managed disk and publish it to the gallery.
        Returns the gallery image version ID.
        """
        src_filename = url.rstrip("/").split("/")[-1]
        base_name = src_filename.split(".")[0]
        blob_name = blob_name or (base_name + ".vhd")
        sentinel_name = blob_name + ".bootstrap_done"
        failed_name = blob_name + ".bootstrap_failed"

        log.info("bootstrap: %s  source=%s", image_name, url)
        log.info("bootstrap: blob=%s", blob_name)

        if not self.blob_exists(sentinel_name):
            vhd_sas_url      = self.generate_sas_url(blob_name,      expiry_hours=8, write=True)
            sentinel_sas_url = self.generate_sas_url(sentinel_name,  expiry_hours=8, write=True)
            failed_sas_url   = self.generate_sas_url(failed_name,    expiry_hours=8, write=True)
            script = _AZURE_BOOTSTRAP_SCRIPT.format(
                hf_url=url,
                vhd_sas_url=vhd_sas_url,
                sentinel_sas_url=sentinel_sas_url,
                failed_sas_url=failed_sas_url,
            )
            vm_info = self.launch_bootstrap_vm(script)
            t0 = time.time()
            try:
                log.info("bootstrap: VM running, streaming logs from %s", vm_info["public_ip"])
                log.info("bootstrap: SSH: ssh -i %s azureuser@%s", self.ssh_privkey, vm_info["public_ip"])
                with BootstrapMonitor(
                    public_ip=vm_info["public_ip"],
                    ssh_privkey=self.ssh_privkey,
                    ssh_user="azureuser",
                    sentinel_fn=lambda: self.blob_exists(sentinel_name),
                ) as monitor:
                    monitor.wait(timeout=7200)
            finally:
                self.cleanup_bootstrap_vm(vm_info["vm_name"], vm_info["pip_name"], vm_info["nic_name"])
            log.info("bootstrap: VHD ready in blob storage (%.1f min)", (time.time() - t0) / 60)
        else:
            log.info("bootstrap: sentinel exists — skipping VM phase")

        return self.ensure_resource_from_blob(blob_name, image_name, version)

    # ── VM Lifecycle ──────────────────────────────────────────────────────────

    def launch(
        self,
        name: str,
        version: str = "1.0.0",
        admin_user: str = "azureuser",
        open_tunnel: bool = True,
    ) -> dict:
        """Launch a VM from a Compute Gallery image. Returns a dict with connection info.

        Creates: static public IP → NIC → VM.
        Injects SSH key via cloud-init custom_data (not waagent os_profile.ssh.public_keys).

        Why cloud-init instead of waagent:
            The OSWorld image was not deprovisioned with 'waagent -deprovision', so
            waagent never signals "provisioning complete" — Azure times out with
            OSProvisioningTimedOut if we rely on it. With provision_vm_agent=False,
            Azure skips the wait. Cloud-init runs independently and creates the user
            + authorized_keys reliably.

        If open_tunnel=True: waits for SSH, opens localhost:{port} → VM:5000 tunnel
        (bypasses Zscaler, which blocks all ports except SSH on corporate networks).

        Returns {vm_name, public_ip, pip_name, nic_name, endpoint, tunnel, local_port}.
        endpoint is None if open_tunnel=False.
        """
        uid = uuid.uuid4().hex[:6]
        vm_name  = f"cube-vm-{uid}"

        image_id = (
            f"/subscriptions/{self.subscription}/resourceGroups/{self.resource_group}"
            f"/providers/Microsoft.Compute/galleries/{self.gallery_name}"
            f"/images/{name}/versions/{version}"
        )

        pubkey = Path(self.ssh_pubkey).read_text().strip()

        # Inject SSH key via cloud-init (custom_data) instead of waagent.
        # The OSWorld image was not deprovisioned with `waagent -deprovision`,
        # so waagent will never signal "provisioning complete" — Azure would
        # time out (OSProvisioningTimedOut) if we relied on it.  With
        # provision_vm_agent=False Azure skips that wait entirely.  Cloud-init
        # runs independently and reliably creates the user + authorized_keys.
        cloud_init = (
            "#cloud-config\n"
            "users:\n"
            f"  - name: {admin_user}\n"
            "    sudo: ALL=(ALL) NOPASSWD:ALL\n"
            "    groups: [sudo, adm]\n"
            "    shell: /bin/bash\n"
            "    ssh_authorized_keys:\n"
            f"      - {pubkey}\n"
        )
        custom_data = base64.b64encode(cloud_init.encode()).decode()

        compute = self._compute()

        log.info("launch: creating network resources")
        pip, nic, pip_name, nic_name = self._create_network_resources(uid)

        log.info("launch: creating VM %s (%s)  image=%s/%s", vm_name, self.vm_size, name, version)
        t0 = time.time()

        poller = compute.virtual_machines.begin_create_or_update(  # type: ignore[call-overload]
            self.resource_group,
            vm_name,
            {  # type: ignore[arg-type]
                "location": self.location,
                "tags": self.tags,
                "hardware_profile": {"vm_size": self.vm_size},
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
                    "custom_data": custom_data,
                    "linux_configuration": {
                        "disable_password_authentication": True,
                        "provision_vm_agent": False,
                    },
                },
                "network_profile": {
                    "network_interfaces": [{"id": nic.id, "properties": {"primary": True}}]
                },
            },
        )
        poller.result()
        elapsed = time.time() - t0

        pip_info = network.public_ip_addresses.get(self.resource_group, pip_name)
        assert pip_info.ip_address, "Public IP address was not assigned"
        public_ip = pip_info.ip_address
        log.info("launch: VM ready in %.0fs: %s @ %s", elapsed, vm_name, public_ip)
        log.info("launch: SSH: ssh -i %s -o IdentitiesOnly=yes %s@%s", self.ssh_privkey, admin_user, public_ip)

        result: dict = {
            "vm_name": vm_name,
            "public_ip": public_ip,
            "pip_name": pip_name,
            "nic_name": nic_name,
            "endpoint": None,
            "tunnel": None,
            "local_port": None,
        }

        if open_tunnel:
            log.info("launch: waiting for SSH...")
            wait_for_ssh(public_ip, admin_user, self.ssh_privkey)
            local_port = free_port()
            log.info("launch: opening tunnel localhost:%d → %s:%d", local_port, public_ip, self.guest_port)
            tunnel = ssh_tunnel(public_ip, admin_user, self.ssh_privkey, local_port, self.guest_port)
            result.update({
                "endpoint": f"http://localhost:{local_port}",
                "tunnel": tunnel,
                "local_port": local_port,
            })

        return result

    def stop(
        self,
        vm_name: str,
        pip_name: str | None = None,
        nic_name: str | None = None,
    ) -> None:
        """Delete a VM and its associated networking resources."""
        compute = self._compute()
        network = self._network()
        log.info("stop: deleting VM %s", vm_name)
        compute.virtual_machines.begin_delete(self.resource_group, vm_name).result()
        log.info("stop: VM deleted")

        if nic_name is None:
            nic_name = vm_name.replace("cube-vm-", "cube-nic-")
        if pip_name is None:
            pip_name = vm_name.replace("cube-vm-", "cube-ip-")

        for fn, resource, name in [
            (network.network_interfaces.begin_delete,  "NIC", nic_name),
            (network.public_ip_addresses.begin_delete, "IP",  pip_name),
        ]:
            try:
                fn(self.resource_group, name).result()
                log.info("stop: %s deleted: %s", resource, name)
            except Exception:
                pass

    def restore_snapshot(
        self,
        vm_name: str,
        name: str,
        version: str = "1.0.0",
        admin_user: str = "azureuser",
    ) -> dict:
        """Reset a VM to clean state by stopping it and launching a fresh copy from the gallery.

        For cloud VMs there is no snapshot mechanism — 'restore' means terminate + re-provision.
        This takes ~3-4 min on Azure (vs ~30s for local QEMU savestate).
        """
        self.stop(vm_name)
        return self.launch(name, version=version, admin_user=admin_user)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def probe(self, endpoint: str, timeout: int = 300) -> dict:
        """Poll the guest agent at endpoint. See _common.probe for details."""
        return probe(endpoint, timeout=timeout)

    def list_images(self) -> list[dict]:
        """Return all image definitions in the gallery."""
        compute = self._compute()
        try:
            defs = list(compute.gallery_images.list_by_gallery(self.resource_group, self.gallery_name))
        except Exception:
            log.warning("list_images: gallery '%s' not found", self.gallery_name)
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

    def convert_to_vhd(self, image_path: Path, output_path: Path | None = None) -> Path:
        """Convert a disk image to a fixed-size Azure-compatible VHD."""
        src = image_path.resolve()
        dst = output_path.resolve() if output_path else src.with_suffix(".vhd")
        convert_image(src, dst, "vpc", "subformat=fixed,force_size", log)
        return dst
