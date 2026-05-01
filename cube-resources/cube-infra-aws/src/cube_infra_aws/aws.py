"""
AWSInfraConfig — InfraConfig implementation for Amazon Web Services.

Provisioning pipeline (~30-90 min, idempotent):
    source_url (HuggingFace qcow2.zip)
        → bootstrap EC2 downloads + converts to fixed VHD  (in-cloud speed)
        → S3 (upload via boto3)
        → ec2:import-snapshot (EBS snapshot)
        → ec2:register-image (AMI)
        → ProvisionStore {"ami_id": "ami-..."}

Launch (~3-5 min per instance):
    AMI
        → EC2 instance (NetworkInterfaces with AssociatePublicIpAddress=True)
        → SSH tunnel localhost:{port} → instance:{guest_port}
        → AWSResourceHandle(endpoint="http://localhost:{port}")

Authentication:
    Uses boto3 default credential chain (env vars / ~/.aws/credentials / instance profile).
    Credentials are never stored in Pydantic fields.

Required AWS resources (created idempotently by provision/launch):
    - S3 bucket:            s3_bucket  (auto-named: cube-vmimages-{account_id})
    - IAM role:             vmimport   (required by EC2 VM Import service, hard-coded name)
    - IAM role + profile:   cube-bootstrap-role / cube-bootstrap  (for bootstrap EC2)
    - EC2 key pair:         key_pair_name  (imported from ~/.ssh/)
    - Security group:       security_group_name  (created with SSH inbound rule)

Usage::

    from cube_infra_aws import AWSInfraConfig
    from cube.resource import VMResourceConfig

    resource = VMResourceConfig(
        name="osworld-ubuntu-vm",
        source_url="https://huggingface.co/datasets/xlangai/ubuntu_osworld/resolve/main/Ubuntu.qcow2.zip",
    )

    # Minimal — region + VPC + S3 bucket all auto-discovered:
    infra = AWSInfraConfig()

    # With explicit region override:
    infra = AWSInfraConfig(region="us-west-2")

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
from typing import Any

import botocore.exceptions
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
from cube_infra_aws._utils import BootstrapMonitor, open_tunnel, open_tunnels, ssh_run, wait_for_ssh

logger = logging.getLogger(__name__)


# ── Bootstrap script ───────────────────────────────────────────────────────────
# Placeholders: {hf_url}, {s3_bucket}, {s3_key}, {sentinel_key}, {failed_key},
#               {region}, {ssh_pubkey}

_AWS_BOOTSTRAP_SCRIPT = """\
#!/bin/bash
set -eo pipefail
exec > /var/log/cube-bootstrap.log 2>&1

# Write a string to an S3 key using Python/boto3 (no CLI dependency)
s3_put() {{
    python3 -c "
import boto3, sys
boto3.client('s3', region_name='{region}').put_object(
    Bucket='{s3_bucket}', Key='$1', Body=sys.stdin.buffer.read()
)" <<< "$2"
}}

on_error() {{
    msg="[bootstrap] FAILED at line $1: $2"
    echo "$msg"
    s3_put "{failed_key}" "$msg" || true
    exit 1
}}
trap 'on_error $LINENO "$BASH_COMMAND"' ERR

echo "[bootstrap] Starting at $(date)"

# ── install tools ─────────────────────────────────────────────────────────────
if command -v yum &>/dev/null; then
    yum install -y qemu-img wget unzip python3-pip
else
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y -qq qemu-utils wget unzip python3-pip
fi
pip3 install boto3 -q
echo "[bootstrap] Tools ready"

# ── download ──────────────────────────────────────────────────────────────────
echo "[bootstrap] Downloading: {hf_url}"
wget --progress=dot:giga -O /tmp/source.download "{hf_url}"
echo "[bootstrap] Downloaded: $(du -sh /tmp/source.download)"

# ── unzip if needed ───────────────────────────────────────────────────────────
if file /tmp/source.download | grep -qi "zip archive"; then
    echo "[bootstrap] Unzipping..."
    unzip -q /tmp/source.download -d /tmp/
    QCOW2=$(find /tmp -name "*.qcow2" | head -1)
    echo "[bootstrap] Unzipped: $QCOW2"
else
    QCOW2=/tmp/source.download
fi

# ── convert ───────────────────────────────────────────────────────────────────
echo "[bootstrap] Converting qcow2 → fixed VHD..."
qemu-img convert -f qcow2 -O vpc -o subformat=fixed,force_size "$QCOW2" /tmp/output.vhd
echo "[bootstrap] Converted: $(du -sh /tmp/output.vhd)"

# ── inject SSH + headless display into VHD ────────────────────────────────────
echo "[bootstrap] Injecting SSH and xorg.conf into VHD..."
LOOP=$(losetup -f --show -P /tmp/output.vhd)
sleep 2
# Find the root partition (ext4)
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
dpkg -l xserver-xorg-video-dummy 2>/dev/null | grep -q '^ii' || \
    apt-get install -y -qq xserver-xorg-video-dummy
"
# Enable sshd at boot via direct symlink (systemctl in chroot fails without running systemd).
SSH_SVC=/mnt/guest/lib/systemd/system/ssh.service
SSH_SVC_ALT=/mnt/guest/usr/lib/systemd/system/ssh.service
for svc in "$SSH_SVC" "$SSH_SVC_ALT"; do
    [ -f "$svc" ] && \
        mkdir -p /mnt/guest/etc/systemd/system/multi-user.target.wants && \
        ln -sf "${{svc#/mnt/guest}}" \
            /mnt/guest/etc/systemd/system/multi-user.target.wants/ssh.service && \
        echo "[bootstrap] Enabled sshd via $svc" && break
done
# Create xorg.conf for dummy display (EC2 has no GPU; OSWorld needs a display).
mkdir -p /mnt/guest/etc/X11
cat > /mnt/guest/etc/X11/xorg.conf << 'XORGEOF'
Section "Device"
  Identifier "Configured Video Device"
  Driver "dummy"
  VideoRam 16384
EndSection

Section "Monitor"
  Identifier "Configured Monitor"
  HorizSync 5-1000
  VertRefresh 5-200
EndSection

Section "Screen"
  Identifier "Default Screen"
  Monitor "Configured Monitor"
  Device "Configured Video Device"
  DefaultDepth 24
  SubSection "Display"
    Depth 24
    Virtual 1920 1080
  EndSubSection
EndSection
XORGEOF
echo "[bootstrap] xorg.conf (dummy driver) written"
# Inject the bootstrap pubkey into all likely user homes
SSH_PUBKEY='{ssh_pubkey}'
for USER_HOME in /mnt/guest/home/user /mnt/guest/home/ubuntu /mnt/guest/root; do
    [ -d "$USER_HOME" ] || continue
    mkdir -p "$USER_HOME/.ssh"
    grep -qxF "$SSH_PUBKEY" "$USER_HOME/.ssh/authorized_keys" 2>/dev/null \
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

# ── upload via boto3 ──────────────────────────────────────────────────────────
echo "[bootstrap] Uploading to S3..."
python3 - << 'PYEOF'
import boto3
s3 = boto3.client('s3', region_name='{region}')
s3.upload_file('/tmp/output.vhd', '{s3_bucket}', '{s3_key}',
    Callback=lambda n: print(f'  uploaded {{n}} bytes', end='\\r', flush=True))
print()
PYEOF
echo "[bootstrap] Upload complete"

# ── signal done ───────────────────────────────────────────────────────────────
s3_put "{sentinel_key}" "done"
echo "[bootstrap] Done at $(date)"
"""


# ── Docker-host bootstrap script ─────────────────────────────────────────────
# Placeholders: {s3_bucket}, {sentinel_key}, {failed_key}, {region},
#               {docker_pull_commands}
# Runs as root via EC2 user-data (cloud-init).
# Writes sentinel to S3 when done; reads by _provision_docker_service().
# No SSH key is baked in — EC2 injects the key pair at launch time via
# cloud-init (same mechanism as all standard AMIs).

_AWS_DOCKER_BOOTSTRAP_SCRIPT = """\
#!/bin/bash
set -eo pipefail
exec > /var/log/cube-bootstrap.log 2>&1

s3_put() {{
    python3 -c "
import boto3, sys
boto3.client('s3', region_name='{region}').put_object(
    Bucket='{s3_bucket}', Key='$1', Body=sys.stdin.buffer.read()
)" <<< "$2"
}}

on_error() {{
    msg="[bootstrap] FAILED at line $1: $2"
    echo "$msg"
    s3_put "{failed_key}" "$msg" || true
    exit 1
}}
trap 'on_error $LINENO "$BASH_COMMAND"' ERR

echo "[bootstrap] Starting at $(date)"

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq docker.io curl python3-pip
pip3 install boto3 -q

systemctl enable docker
systemctl start docker
usermod -aG docker ubuntu

{volume_setup_commands}


{docker_pull_commands}
echo "[bootstrap] Docker images ready"

s3_put "{sentinel_key}" "done"
echo "[bootstrap] Done at $(date)"
"""


# ── Tag helpers ────────────────────────────────────────────────────────────────


def _dict_to_ec2_tags(d: dict[str, str]) -> list[dict[str, str]]:
    """Convert a plain dict to EC2 tag format: [{"Key": k, "Value": v}, ...]."""
    return [{"Key": k, "Value": v} for k, v in d.items()]


def _ec2_tags_to_dict(tags: list[dict[str, str]]) -> dict[str, str]:
    """Convert EC2 tag format to plain dict."""
    return {t["Key"]: t["Value"] for t in tags}


# ── AWSResourceHandle ──────────────────────────────────────────────────────────


@dataclass
class AWSResourceHandle(ResourceHandle):
    """ResourceHandle for a running EC2 instance with open SSH tunnel(s)."""

    _instance_id: str = field(default="", repr=False)
    _tunnels: list[subprocess.Popen] = field(default_factory=list, repr=False)

    def close(self) -> None:
        """Terminate SSH tunnel(s) and stop the EC2 instance."""
        for proc in self._tunnels:
            try:
                proc.terminate()
            except Exception:
                pass
        self._tunnels = []
        logger.info("SSH tunnel(s) closed for run %s", self.run_id[:8])

        if self._instance_id:
            assert isinstance(self.infra, AWSInfraConfig)
            self.infra._terminate_instance(self._instance_id)
            self._instance_id = ""


# ── AWSInfraConfig ─────────────────────────────────────────────────────────────


class AWSInfraConfig(InfraConfig):
    """AWS InfraConfig: provisions and launches CUBE VM resources on EC2.

    Authentication uses the boto3 default credential chain — set AWS credentials
    via environment variables, ``~/.aws/credentials``, or an EC2 instance profile.

    Typical usage::

        # Minimal — region, VPC, subnet, S3 bucket all auto-discovered:
        infra = AWSInfraConfig()

        # With explicit region:
        infra = AWSInfraConfig(region="us-west-2")

        # With explicit S3 bucket (globally unique name required):
        infra = AWSInfraConfig(s3_bucket="my-cube-vm-images")

        infra.provision(resource)       # ~30-90 min, idempotent
        run_debug_agent(benchmark, infra)

    ── Auto-discovered (leave as None unless you need to override) ──────────────
    region          str | None = None
        AWS region.  Auto-populated from ``AWS_DEFAULT_REGION`` env var or
        ``~/.aws/config``.  Raises if neither is set.
    account_id      str | None = None
        AWS account ID.  Auto-populated via ``sts:GetCallerIdentity``.
    s3_bucket       str | None = None
        S3 bucket for intermediate VHD storage.
        Auto-named as ``cube-vmimages-{account_id}`` and created if needed.
    vpc_id          str | None = None
        VPC for EC2 instances.  Auto-populated from the default VPC.
    subnet_id       str | None = None
        Subnet for EC2 instances.  Auto-populated from the first default subnet.

    ── Named resources (created idempotently during provision/launch) ───────────
    security_group_name str = "cube-sg"
        Security group with SSH inbound rule.  Created in vpc_id if absent.
    key_pair_name       str = "cube-key"
        EC2 key pair.  Imported from ssh_pubkey_path if absent.

    ── Overrideable defaults ─────────────────────────────────────────────────────
    instance_type   str = "t3.xlarge"
    guest_port      int = 5000
    ssh_privkey_path    str | None = None
        Path to the SSH private key.  Auto-discovered from ~/.ssh/ in priority
        order: id_ed25519, id_ecdsa, id_rsa, id_dsa.  Excluded from serialization
        (path is machine-local; each machine re-discovers its own key).
    ssh_pubkey_path     str | None = None
        Path to the SSH public key.  Auto-derived as ssh_privkey_path + ".pub".
        Content is read once during provisioning to inject into the VHD.
    tags            dict[str, str] = {"project": "cube"}

    ── Bootstrap pipeline (advanced) ─────────────────────────────────────────────
    bootstrap_instance_type     str = "t3.medium"
    bootstrap_root_volume_gb    int = 128
    bootstrap_role_name         str = "cube-bootstrap-role"
    bootstrap_profile_name      str = "cube-bootstrap"

    """

    # ── Auto-discovered ────────────────────────────────────────────────────────
    region: str | None = None
    account_id: str | None = None
    s3_bucket: str | None = None
    vpc_id: str | None = None
    subnet_id: str | None = None

    # ── Named resources (created lazily, not serialized as None) ──────────────
    security_group_name: str = "cube-sg"
    key_pair_name: str = "cube-key"

    # ── Overrideable defaults ─────────────────────────────────────────────────
    instance_type: str = "t3.xlarge"
    guest_port: int = 5000
    ssh_privkey_path: str | None = Field(default=None, repr=False, exclude=True)
    ssh_pubkey_path: str | None = Field(default=None, repr=False, exclude=True)
    tags: dict[str, str] = Field(default_factory=lambda: {"project": "cube"})

    # ── Bootstrap pipeline ────────────────────────────────────────────────────
    bootstrap_instance_type: str = "t3.medium"
    bootstrap_root_volume_gb: int = 128
    bootstrap_role_name: str = "cube-bootstrap-role"
    bootstrap_profile_name: str = "cube-bootstrap"

    # ── Auto-discovery ────────────────────────────────────────────────────────

    @model_validator(mode="after")
    def _autodiscover(self) -> "AWSInfraConfig":
        """Fill in any None fields by querying the AWS SDK.

        Only runs discovery for fields that are None — explicitly set values
        are always respected.

        Raises ValueError if required resources cannot be found (e.g. no default
        VPC in the region — set ``vpc_id=`` explicitly).
        """
        # ── AWS resource discovery (requires SDK calls) ────────────────────────
        needs_aws_discovery = not all(
            [
                self.region,
                self.account_id,
                self.s3_bucket,
                self.vpc_id,
                self.subnet_id,
            ]
        )
        if needs_aws_discovery:
            import boto3

            session = boto3.session.Session()

            if not self.region:
                region = session.region_name
                if not region:
                    raise ValueError(
                        "AWS region not set.  Set the AWS_DEFAULT_REGION environment "
                        "variable, configure ~/.aws/config, or pass region= explicitly."
                    )
                object.__setattr__(self, "region", region)

            try:
                sts = session.client("sts", region_name=self.region)
                if not self.account_id:
                    account_id = sts.get_caller_identity()["Account"]
                    object.__setattr__(self, "account_id", account_id)
            except botocore.exceptions.NoCredentialsError as exc:
                raise ValueError(
                    "No AWS credentials found.  Configure via environment variables "
                    "(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY), ~/.aws/credentials, "
                    "or an EC2 instance profile.\n" + str(exc)
                ) from exc
            except Exception as exc:
                raise ValueError(f"Could not resolve AWS account ID: {exc}") from exc

            if not self.s3_bucket:
                object.__setattr__(self, "s3_bucket", f"cube-vmimages-{self.account_id}")

            ec2 = session.client("ec2", region_name=self.region)

            if not self.vpc_id:
                resp = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])
                vpcs = resp.get("Vpcs", [])
                if not vpcs:
                    raise ValueError(
                        f"No default VPC found in region '{self.region}'.  "
                        "Create a default VPC (``aws ec2 create-default-vpc``) "
                        "or set vpc_id= explicitly."
                    )
                object.__setattr__(self, "vpc_id", vpcs[0]["VpcId"])

            if not self.subnet_id:
                resp = ec2.describe_subnets(
                    Filters=[
                        {"Name": "vpc-id", "Values": [self.vpc_id]},
                        {"Name": "default-for-az", "Values": ["true"]},
                    ]
                )
                subnets = sorted(
                    resp.get("Subnets", []),
                    key=lambda s: s["AvailabilityZone"],
                )
                if not subnets:
                    raise ValueError(f"No default subnets found in VPC '{self.vpc_id}'.  Set subnet_id= explicitly.")
                object.__setattr__(self, "subnet_id", subnets[0]["SubnetId"])

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
        """Stable key: provider + region only (not instance type or S3 bucket).

        Two AWSInfraConfig objects with the same region share provisioned AMIs.
        """
        return f"aws:{self.region}"

    def capabilities(self) -> set[str]:
        """EC2 HVM instances support KVM workloads and Docker-host provisioning."""
        return {"kvm", "docker"}

    def provision(self, resource: ResourceConfig) -> None:
        """Bootstrap a resource into an EC2 AMI.

        For VMResourceConfig: downloads qcow2, converts to VHD, imports as AMI.
        For DockerServiceConfig: installs Docker, pulls images, creates AMI.

        Both paths are idempotent; the ProvisionStore is checked first.

        Raises:
            UnsupportedResourceType: if resource is not VMResourceConfig or DockerServiceConfig.
            ValueError: if source_url (VM) or docker_images (Docker) are not set.
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

        if isinstance(resource, DockerServiceConfig):
            if not resource.docker_images:
                raise ValueError(f"Cannot provision {image_name!r}: docker_images is empty.")
            logger.info("provision: building Docker-host AMI %r …", image_name)
            ami_id = self._provision_docker_service(resource, image_name)
            store.put(shim, self, {"ami_id": ami_id})
            logger.info("provision: %r registered for %s", image_name, self.fingerprint())
            return

        if not resource.source_url:
            raise ValueError(
                f"Cannot provision {image_name!r}: no source_url set and "
                f"no registration found for {self.fingerprint()!r}.\n"
                f'  Manual: infra.register(resource, {{"ami_id": ...}})'
            )

        logger.info("provision: bootstrapping %r → AMI", image_name)
        ami_id = self._bootstrap(url=resource.source_url, image_name=image_name)
        store.put(shim, self, {"ami_id": ami_id})
        logger.info("provision: %r registered for %s", image_name, self.fingerprint())

    def unprovision(self, resource: ResourceConfig) -> None:
        """Deregister the AMI, delete the EBS snapshot, S3 objects, and ProvisionStore entry.

        Safe to call when not provisioned — no-ops if not registered.

        IMPORTANT: the snapshot_id is retrieved before deregistering the AMI because
        EC2 does not allow querying block devices of deregistered AMIs.

        Raises:
            UnsupportedResourceType: if resource is not VMResourceConfig or DockerServiceConfig.
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

        ami_id = resource_info.get("ami_id", "")
        ec2 = self._ec2()

        if ami_id:
            # Get snapshot ID before deregistering (lost afterwards).
            snapshot_id = None
            try:
                resp = ec2.describe_images(ImageIds=[ami_id])
                if resp["Images"]:
                    for mapping in resp["Images"][0].get("BlockDeviceMappings", []):
                        if "Ebs" in mapping:
                            snapshot_id = mapping["Ebs"].get("SnapshotId")
                            break
            except Exception as exc:
                logger.warning("unprovision: could not describe AMI %s: %s", ami_id, exc)

            try:
                logger.info("unprovision: deregistering AMI %s …", ami_id)
                ec2.deregister_image(ImageId=ami_id)
                logger.info("unprovision: AMI %s deregistered", ami_id)
            except Exception as exc:
                logger.warning("unprovision: could not deregister AMI %s: %s", ami_id, exc)

            if snapshot_id:
                try:
                    ec2.delete_snapshot(SnapshotId=snapshot_id)
                    logger.info("unprovision: deleted snapshot %s", snapshot_id)
                except Exception as exc:
                    logger.warning("unprovision: could not delete snapshot %s: %s", snapshot_id, exc)

        s3 = self._s3()
        if isinstance(resource, DockerServiceConfig):
            # Docker bootstrap uses sentinel keys without a .vhd prefix.
            s3_keys = (
                image_name + ".docker_bootstrap_done",
                image_name + ".docker_bootstrap_failed",
            )
        else:
            vhd_key = image_name + ".vhd"
            s3_keys = (vhd_key, vhd_key + ".bootstrap_done", vhd_key + ".bootstrap_failed")
        for key in s3_keys:
            try:
                s3.delete_object(Bucket=self.s3_bucket, Key=key)
                logger.info("unprovision: deleted s3://%s/%s", self.s3_bucket, key)
            except Exception as exc:
                logger.warning("unprovision: could not delete s3://%s/%s: %s", self.s3_bucket, key, exc)

        store.delete(shim, self)
        logger.info("unprovision: %r removed from ProvisionStore", image_name)

    def launch(self, resource: ResourceConfig) -> AWSResourceHandle:
        """Launch an EC2 instance from the AMI, open SSH tunnel(s), return handle.

        For VMResourceConfig: opens a single tunnel to guest_port.
        For DockerServiceConfig: runs launch_script via SSH, opens one tunnel
        per service in resource.services; handle.endpoints is populated.

        Reads ami_id from the ProvisionStore.
        Raises ResourceNotReadyError if provision() was never called.

        run_id is generated internally. TTL resolves as:
        self.default_ttl_seconds ?? resource.default_ttl_seconds.
        """
        if not isinstance(resource, (VMResourceConfig, DockerServiceConfig)):
            raise UnsupportedResourceType(resource, self)

        resource_info = ProvisionStore().get(self._resource_shim(resource), self)
        if resource_info is None:
            raise ResourceNotReadyError(resource, self)

        ami_id = resource_info["ami_id"]
        run_id = str(uuid.uuid4())
        uid = uuid.uuid4().hex[:6]
        run_id_short = run_id[:8]
        instance_name = f"cube-{run_id_short}-vm-{uid}"

        effective_ttl = (
            self.default_ttl_seconds if self.default_ttl_seconds is not None else resource.default_ttl_seconds
        )
        created_at = datetime.now(timezone.utc)
        expires_at = created_at + timedelta(seconds=effective_ttl) if effective_ttl else None

        # Spec-required tags applied to every instance.
        cube_tags: dict[str, str] = {
            "cube:infra": self.fingerprint(),
            "cube:run_id": run_id,
            "cube:resource": resource.name,
            "cube:created_at": created_at.isoformat(),
        }
        if expires_at:
            cube_tags["cube:expires_at"] = expires_at.isoformat()

        all_tags = {**self.tags, **cube_tags, "Name": instance_name}

        sg_id = self._ensure_security_group()
        self._ensure_key_pair()

        # User-data injects SSH key into the running VM as a belt-and-suspenders
        # measure.  The AMI already has the key from the bootstrap phase.
        pubkey = Path(self.ssh_pubkey_path).read_text().strip()  # type: ignore[arg-type]
        user_data = self._make_user_data(pubkey)

        ec2 = self._ec2()
        logger.info(
            "launch: starting instance from AMI %s (%s)  type=%s  name=%s",
            ami_id,
            resource.name,
            self.instance_type,
            instance_name,
        )
        t0 = time.time()

        # NetworkInterfaces is required to get a public IP; SubnetId and
        # SecurityGroupIds cannot be specified at the top level in this case.
        resp = ec2.run_instances(
            ImageId=ami_id,
            InstanceType=self.instance_type,
            MinCount=1,
            MaxCount=1,
            KeyName=self.key_pair_name,
            UserData=user_data,
            NetworkInterfaces=[
                {
                    "DeviceIndex": 0,
                    "SubnetId": self.subnet_id,
                    "Groups": [sg_id],
                    "AssociatePublicIpAddress": True,
                }
            ],
            TagSpecifications=[
                {
                    "ResourceType": "instance",
                    "Tags": _dict_to_ec2_tags(all_tags),
                }
            ],
            BlockDeviceMappings=[
                {
                    "DeviceName": "/dev/sda1",
                    "Ebs": {"VolumeType": "gp3", "DeleteOnTermination": True},
                }
            ],
        )
        instance_id = resp["Instances"][0]["InstanceId"]
        logger.info("launch: instance %s", instance_id)

        ec2.get_waiter("instance_running").wait(InstanceIds=[instance_id])

        desc = ec2.describe_instances(InstanceIds=[instance_id])
        instance = desc["Reservations"][0]["Instances"][0]
        public_ip = instance.get("PublicIpAddress", "")
        logger.info(
            "launch: running in %.0fs: %s @ %s",
            time.time() - t0,
            instance_id,
            public_ip,
        )
        logger.info(
            "launch: SSH: ssh -i %s -o IdentitiesOnly=yes user@%s",
            self.ssh_privkey_path,
            public_ip,
        )

        # SSH + tunnel(s) — clean up instance on any failure to avoid orphaned resources.
        try:
            logger.info("launch: waiting for SSH on %s…", public_ip)
            active_user = wait_for_ssh(
                public_ip,
                "ubuntu",
                self.ssh_privkey_path,  # type: ignore[arg-type]
                fallback_users=["user", "root"],
                timeout=600,
            )

            if isinstance(resource, DockerServiceConfig):
                logger.info("launch: starting Docker services on %s", instance_id)
                self._ssh_run(public_ip, active_user, resource.launch_script)
                logger.info("launch: Docker services started")
                endpoints, tunnels = open_tunnels(
                    public_ip,
                    active_user,
                    self.ssh_privkey_path,  # type: ignore[arg-type]
                    resource.services,
                )
                logger.info("launch: opened %d tunnel(s): %s", len(tunnels), list(endpoints))
                endpoint = next(iter(endpoints.values())) if endpoints else None
                return AWSResourceHandle(
                    run_id=run_id,
                    resource=resource,
                    infra=self,
                    endpoint=endpoint,
                    endpoints=endpoints,
                    created_at=created_at,
                    expires_at=expires_at,
                    _instance_id=instance_id,
                    _tunnels=tunnels,
                )

            tunnel, local_port = open_tunnel(
                public_ip,
                active_user,
                self.ssh_privkey_path,  # type: ignore[arg-type]
                self.guest_port,
            )
            logger.info(
                "launch: opened tunnel localhost:%d → %s:%d",
                local_port,
                public_ip,
                self.guest_port,
            )
        except Exception:
            logger.warning("launch: SSH/tunnel failed — terminating instance %s", instance_id)
            self._terminate_instance(instance_id)
            raise

        endpoint = f"http://localhost:{local_port}"

        return AWSResourceHandle(
            run_id=run_id,
            resource=resource,
            infra=self,
            endpoint=endpoint,
            created_at=created_at,
            expires_at=expires_at,
            _instance_id=instance_id,
            _tunnels=[tunnel],
        )

    def list_active(self, run_id: str | None = None) -> list[AWSResourceHandle]:
        """List running/pending CUBE instances, filtered by run_id if provided.

        Queries EC2 directly via tags.  Cannot reconstruct SSH tunnels — handles
        are returned with endpoint=None.  Use run_id to call cleanup() from any process.
        """
        ec2 = self._ec2()
        handles: list[AWSResourceHandle] = []

        filters: list[dict] = [
            {"Name": "tag:cube:infra", "Values": [self.fingerprint()]},
            {"Name": "instance-state-name", "Values": ["pending", "running"]},
        ]
        if run_id:
            filters.append({"Name": "tag:cube:run_id", "Values": [run_id]})

        try:
            resp = ec2.describe_instances(Filters=filters)
        except Exception as e:
            logger.warning("list_active: failed to describe instances: %s", e)
            return handles

        for reservation in resp.get("Reservations", []):
            for instance in reservation.get("Instances", []):
                tags = _ec2_tags_to_dict(instance.get("Tags", []))
                instance_run_id = tags.get("cube:run_id", "unknown")
                resource_name = tags.get("cube:resource", "unknown")
                instance_id = instance["InstanceId"]
                handles.append(
                    AWSResourceHandle(
                        run_id=instance_run_id,
                        resource=VMResourceConfig(name=resource_name),
                        infra=self,
                        endpoint=None,
                        _instance_id=instance_id,
                        _tunnels=[],
                    )
                )

        return handles

    def cleanup(self, run_id: str) -> None:
        """Terminate all CUBE instances tagged with run_id."""
        handles = self.list_active(run_id=run_id)
        if not handles:
            logger.info("cleanup: no active instances for run %s", run_id[:8])
            return
        ids = [h._instance_id for h in handles if h._instance_id]
        if ids:
            self._ec2().terminate_instances(InstanceIds=ids)
            logger.info("cleanup: terminated %d instance(s) for run %s", len(ids), run_id[:8])

    def cleanup_stale(self, max_age_seconds: int | None = None) -> list[str]:
        """Terminate CUBE instances that have expired or exceeded max_age_seconds.

        Checks in priority order:
          1. cube:expires_at tag < now  →  terminate (TTL set at launch time)
          2. max_age_seconds set and cube:created_at age > max_age_seconds  →  terminate

        Returns list of terminated instance IDs.
        """
        ec2 = self._ec2()
        terminated: list[str] = []
        now = datetime.now(timezone.utc)

        filters: list[dict] = [
            {"Name": "tag:cube:infra", "Values": [self.fingerprint()]},
            {"Name": "instance-state-name", "Values": ["pending", "running"]},
        ]

        try:
            resp = ec2.describe_instances(Filters=filters)
        except Exception as e:
            logger.warning("cleanup_stale: failed to describe instances: %s", e)
            return terminated

        to_terminate: list[str] = []

        for reservation in resp.get("Reservations", []):
            for instance in reservation.get("Instances", []):
                tags = _ec2_tags_to_dict(instance.get("Tags", []))
                instance_id = instance["InstanceId"]

                should_delete = False
                has_valid_expires_at = False

                # Priority 1: explicit TTL tag written at launch time.
                expires_at_str = tags.get("cube:expires_at")
                if expires_at_str:
                    try:
                        expires_at = datetime.fromisoformat(expires_at_str)
                        has_valid_expires_at = True
                        if expires_at < now:
                            should_delete = True
                    except ValueError:
                        logger.warning(
                            "cleanup_stale: invalid cube:expires_at %r on %s",
                            expires_at_str,
                            instance_id,
                        )

                # Priority 2: age-based fallback (skipped if expires_at is set).
                if not has_valid_expires_at and not should_delete and max_age_seconds is not None:
                    created_at_str = tags.get("cube:created_at")
                    try:
                        if created_at_str:
                            created_at = datetime.fromisoformat(created_at_str)
                            age = (now - created_at).total_seconds()
                        else:
                            launch_time = instance.get("LaunchTime")
                            age = (now - launch_time).total_seconds() if launch_time else 0
                        should_delete = age > max_age_seconds
                    except (ValueError, TypeError):
                        pass

                if should_delete:
                    to_terminate.append(instance_id)

        if to_terminate:
            ec2.terminate_instances(InstanceIds=to_terminate)
            terminated.extend(to_terminate)
            logger.info(
                "cleanup_stale: terminated %d instance(s): %s",
                len(terminated),
                terminated,
            )

        return terminated

    # ── Private AWS SDK clients ───────────────────────────────────────────────

    def _ec2(self) -> Any:
        import boto3

        return boto3.client("ec2", region_name=self.region)

    def _s3(self) -> Any:
        import boto3

        return boto3.client("s3", region_name=self.region)

    def _iam(self) -> Any:
        import boto3

        return boto3.client("iam")

    # ── Idempotent setup helpers ──────────────────────────────────────────────

    def _ensure_s3_bucket(self) -> None:
        """Create the S3 bucket for VHD uploads if it doesn't exist.

        NOTE: us-east-1 does NOT accept a LocationConstraint — must omit
        CreateBucketConfiguration for that region.
        """
        s3 = self._s3()
        try:
            s3.head_bucket(Bucket=self.s3_bucket)
            return
        except botocore.exceptions.ClientError:
            pass

        logger.info("_ensure_s3_bucket: creating s3://%s in %s", self.s3_bucket, self.region)
        if self.region == "us-east-1":
            s3.create_bucket(Bucket=self.s3_bucket)
        else:
            s3.create_bucket(
                Bucket=self.s3_bucket,
                CreateBucketConfiguration={"LocationConstraint": self.region},
            )
        s3.put_public_access_block(
            Bucket=self.s3_bucket,
            PublicAccessBlockConfiguration={
                "BlockPublicAcls": True,
                "IgnorePublicAcls": True,
                "BlockPublicPolicy": True,
                "RestrictPublicBuckets": True,
            },
        )
        logger.info("_ensure_s3_bucket: created s3://%s", self.s3_bucket)

    def _ensure_key_pair(self) -> str:
        """Import our local SSH public key as an EC2 key pair. Idempotent.

        Returns the key pair name.
        """
        ec2 = self._ec2()
        try:
            ec2.describe_key_pairs(KeyNames=[self.key_pair_name])
            return self.key_pair_name
        except botocore.exceptions.ClientError:
            pass

        pubkey = Path(self.ssh_pubkey_path).read_bytes()  # type: ignore[arg-type]
        ec2.import_key_pair(KeyName=self.key_pair_name, PublicKeyMaterial=pubkey)
        logger.info("_ensure_key_pair: imported key pair: %s", self.key_pair_name)
        return self.key_pair_name

    def _ensure_security_group(self) -> str:
        """Create a security group that allows SSH inbound. Returns group ID. Idempotent.

        The security group is created in self.vpc_id so that it is valid for
        instances launched into that VPC.
        """
        ec2 = self._ec2()

        # Always filter by vpc_id to avoid returning groups from other VPCs.
        try:
            resp = ec2.describe_security_groups(
                Filters=[
                    {"Name": "group-name", "Values": [self.security_group_name]},
                    {"Name": "vpc-id", "Values": [self.vpc_id]},
                ]
            )
            if resp["SecurityGroups"]:
                return resp["SecurityGroups"][0]["GroupId"]
        except botocore.exceptions.ClientError:
            pass

        sg = ec2.create_security_group(
            GroupName=self.security_group_name,
            Description="CUBE experiment - SSH inbound only",
            VpcId=self.vpc_id,
            TagSpecifications=[
                {
                    "ResourceType": "security-group",
                    "Tags": _dict_to_ec2_tags(self.tags),
                }
            ],
        )
        sg_id = sg["GroupId"]
        ec2.authorize_security_group_ingress(
            GroupId=sg_id,
            IpPermissions=[
                {
                    "IpProtocol": "tcp",
                    "FromPort": 22,
                    "ToPort": 22,
                    "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "SSH"}],
                }
            ],
        )
        logger.info("_ensure_security_group: created %s (%s)", sg_id, self.security_group_name)
        return sg_id

    def _ensure_vmimport_role(self) -> None:
        """Create the 'vmimport' IAM role required by ec2:import-snapshot. Idempotent.

        The role name must be exactly 'vmimport' — AWS VM Import service looks for
        it by this exact name (not configurable).
        """
        import json as _json

        iam = self._iam()

        trust_policy = _json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"Service": "vmie.amazonaws.com"},
                        "Action": "sts:AssumeRole",
                        "Condition": {"StringEquals": {"sts:ExternalId": "vmimport"}},
                    }
                ],
            }
        )
        role_policy = _json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["s3:GetBucketLocation", "s3:GetObject", "s3:ListBucket"],
                        "Resource": [
                            f"arn:aws:s3:::{self.s3_bucket}",
                            f"arn:aws:s3:::{self.s3_bucket}/*",
                        ],
                    },
                    {
                        "Effect": "Allow",
                        "Action": [
                            "ec2:ModifySnapshotAttribute",
                            "ec2:CopySnapshot",
                            "ec2:RegisterImage",
                            "ec2:Describe*",
                        ],
                        "Resource": "*",
                    },
                ],
            }
        )

        try:
            iam.create_role(
                RoleName="vmimport",
                AssumeRolePolicyDocument=trust_policy,
                Description="Allows EC2 VM Import service to access S3 and create snapshots",
            )
            logger.info("_ensure_vmimport_role: created vmimport IAM role")
        except iam.exceptions.EntityAlreadyExistsException:
            pass

        iam.put_role_policy(
            RoleName="vmimport",
            PolicyName="vmimport-s3-ec2",
            PolicyDocument=role_policy,
        )

    def _ensure_bootstrap_instance_profile(self) -> str:
        """Create an IAM instance profile granting bootstrap VMs S3 write access. Idempotent.

        The instance profile is attached at launch and provides temporary S3 credentials
        via the EC2 metadata service (IMDSv2) — no credentials in user-data scripts.

        Returns the instance profile name.
        """
        import json as _json

        iam = self._iam()

        trust_policy = _json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"Service": "ec2.amazonaws.com"},
                        "Action": "sts:AssumeRole",
                    }
                ],
            }
        )
        s3_policy = _json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["s3:PutObject", "s3:GetObject", "s3:HeadObject"],
                        "Resource": f"arn:aws:s3:::{self.s3_bucket}/*",
                    }
                ],
            }
        )

        try:
            iam.create_role(
                RoleName=self.bootstrap_role_name,
                AssumeRolePolicyDocument=trust_policy,
                Description="Bootstrap VMs: S3 write for image conversion",
            )
            logger.info(
                "_ensure_bootstrap_instance_profile: created IAM role %s",
                self.bootstrap_role_name,
            )
        except iam.exceptions.EntityAlreadyExistsException:
            pass

        iam.put_role_policy(
            RoleName=self.bootstrap_role_name,
            PolicyName="s3-bootstrap-write",
            PolicyDocument=s3_policy,
        )

        try:
            iam.create_instance_profile(InstanceProfileName=self.bootstrap_profile_name)
            iam.add_role_to_instance_profile(
                InstanceProfileName=self.bootstrap_profile_name,
                RoleName=self.bootstrap_role_name,
            )
            logger.info(
                "_ensure_bootstrap_instance_profile: created profile %s",
                self.bootstrap_profile_name,
            )
            time.sleep(10)  # IAM propagation delay — required before EC2 can use it
        except iam.exceptions.EntityAlreadyExistsException:
            pass

        return self.bootstrap_profile_name

    def _s3_object_exists(self, key: str) -> bool:
        """Return True if an S3 object exists in the bootstrap bucket."""
        try:
            self._s3().head_object(Bucket=self.s3_bucket, Key=key)
            return True
        except botocore.exceptions.ClientError:
            return False

    # ── Provisioning internals ────────────────────────────────────────────────

    def _latest_ubuntu_ami(self) -> str:
        """Return the latest Ubuntu 22.04 LTS AMI ID for the configured region."""
        ec2 = self._ec2()
        resp = ec2.describe_images(
            Owners=["099720109477"],  # Canonical
            Filters=[
                {
                    "Name": "name",
                    "Values": ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"],
                },
                {"Name": "state", "Values": ["available"]},
                {"Name": "architecture", "Values": ["x86_64"]},
            ],
        )
        images = sorted(resp["Images"], key=lambda x: x["CreationDate"], reverse=True)
        if not images:
            raise RuntimeError(f"No Ubuntu 22.04 AMI found in region {self.region}")
        ami_id = images[0]["ImageId"]
        logger.info("_latest_ubuntu_ami: %s", ami_id)
        return ami_id

    def _make_user_data(self, pubkey: str) -> str:
        """Build a cloud-init user-data script that injects the SSH public key.

        Belt-and-suspenders: the AMI already has the key from the bootstrap phase,
        but this ensures it works even if the key path changed since provisioning.
        Returns a plain (not base64) bash script — run_instances accepts either form.
        """
        return f"""#!/bin/bash
set -e
export DEBIAN_FRONTEND=noninteractive
if ! command -v sshd &>/dev/null; then
    apt-get update -qq && apt-get install -y -qq openssh-server
fi
systemctl enable ssh || true
systemctl start ssh || true
ufw allow ssh 2>/dev/null || true
for _home in /home/user /home/ubuntu /root; do
    [ -d "$_home" ] || continue
    mkdir -p "$_home/.ssh"
    echo '{pubkey}' >> "$_home/.ssh/authorized_keys"
    chmod 700 "$_home/.ssh"
    chmod 600 "$_home/.ssh/authorized_keys"
    chown -R "$(stat -c '%U' "$_home"):" "$_home/.ssh" 2>/dev/null || true
done
"""

    def _import_snapshot(self, s3_uri: str, description: str, disk_format: str = "VHD") -> str:
        """Import a fixed VHD from S3 as an EBS snapshot. Returns snapshot_id.

        Polls until the import task completes (~5-15 min).
        disk_format must be 'VHD' (all caps) — EC2 is case-sensitive here.
        """
        parts = s3_uri.removeprefix("s3://").split("/", 1)
        bucket, key = parts[0], parts[1]

        ec2 = self._ec2()
        logger.info("_import_snapshot: %s (format=%s)", key, disk_format)
        resp = ec2.import_snapshot(
            Description=description,
            DiskContainer={
                "Description": description,
                "Format": disk_format,
                "UserBucket": {"S3Bucket": bucket, "S3Key": key},
            },
        )
        task_id = resp["ImportTaskId"]
        logger.info("_import_snapshot: ImportTaskId=%s", task_id)

        logger.info("_import_snapshot: polling until complete (~5-15 min)…")
        t0 = time.time()
        last_log_t = 0.0
        last_pct = ""
        while True:
            tasks = ec2.describe_import_snapshot_tasks(ImportTaskIds=[task_id])
            task = tasks["ImportSnapshotTasks"][0]["SnapshotTaskDetail"]
            status = task["Status"]
            progress = task.get("Progress", "0")
            elapsed = int(time.time() - t0)

            if status == "completed":
                snapshot_id = task["SnapshotId"]
                logger.info(
                    "_import_snapshot: done in %dm%02ds: %s",
                    elapsed // 60,
                    elapsed % 60,
                    snapshot_id,
                )
                ec2.create_tags(
                    Resources=[snapshot_id],
                    Tags=_dict_to_ec2_tags(self.tags),
                )
                return snapshot_id
            if status in ("deleted", "error"):
                raise RuntimeError(f"import-snapshot failed: {task}")

            now = time.time()
            if now - last_log_t >= 60 or progress != last_pct:
                logger.info(
                    "_import_snapshot: [%dm%02ds] %s%%",
                    elapsed // 60,
                    elapsed % 60,
                    progress,
                )
                last_log_t = now
                last_pct = progress
            time.sleep(15)

    def _register_ami(self, snapshot_id: str, name: str) -> str:
        """Register an EBS snapshot as a bootable HVM AMI. Returns ami_id. Idempotent.

        EnaSupport=True is required for t3+ instance families (enhanced networking).
        Omitting it causes InvalidParameterCombination when launching t3+ instances.
        """
        ec2 = self._ec2()

        resp = ec2.describe_images(
            Owners=["self"],
            Filters=[{"Name": "name", "Values": [name]}],
        )
        if resp["Images"]:
            ami_id = resp["Images"][0]["ImageId"]
            logger.info("_register_ami: already registered: %s (%s) — skipping", ami_id, name)
            return ami_id

        logger.info("_register_ami: registering %s", name)
        resp = ec2.register_image(
            Name=name,
            Description=f"CUBE benchmark image: {name}",
            Architecture="x86_64",
            RootDeviceName="/dev/sda1",
            VirtualizationType="hvm",
            EnaSupport=True,  # required for t3+ instance families
            BlockDeviceMappings=[
                {
                    "DeviceName": "/dev/sda1",
                    "Ebs": {
                        "SnapshotId": snapshot_id,
                        "VolumeType": "gp3",
                        "DeleteOnTermination": True,
                    },
                }
            ],
        )
        ami_id = resp["ImageId"]
        ec2.create_tags(Resources=[ami_id], Tags=_dict_to_ec2_tags(self.tags))
        logger.info("_register_ami: registered %s", ami_id)
        return ami_id

    def _launch_bootstrap_ec2(self, script: str) -> dict:
        """Launch an EC2 instance with the bootstrap script as user-data.

        Uses the latest Ubuntu 22.04 AMI.  The instance profile grants S3 write access.
        Returns {instance_id, public_ip}.
        """
        ec2 = self._ec2()
        ami_id = self._latest_ubuntu_ami()
        sg_id = self._ensure_security_group()
        profile_name = self._ensure_bootstrap_instance_profile()

        user_data = base64.b64encode(script.encode()).decode()
        uid = uuid.uuid4().hex[:6]

        logger.info(
            "_launch_bootstrap_ec2: launching (%s, %d GB root)",
            self.bootstrap_instance_type,
            self.bootstrap_root_volume_gb,
        )
        t0 = time.time()

        # NetworkInterfaces with AssociatePublicIpAddress=True to get a routable IP.
        resp = ec2.run_instances(
            ImageId=ami_id,
            InstanceType=self.bootstrap_instance_type,
            MinCount=1,
            MaxCount=1,
            KeyName=self._ensure_key_pair(),
            UserData=user_data,
            IamInstanceProfile={"Name": profile_name},
            NetworkInterfaces=[
                {
                    "DeviceIndex": 0,
                    "SubnetId": self.subnet_id,
                    "Groups": [sg_id],
                    "AssociatePublicIpAddress": True,
                }
            ],
            TagSpecifications=[
                {
                    "ResourceType": "instance",
                    "Tags": _dict_to_ec2_tags(
                        {
                            **self.tags,
                            "Name": f"cube-bootstrap-{uid}",
                            "role": "bootstrap",
                        }
                    ),
                }
            ],
            BlockDeviceMappings=[
                {
                    "DeviceName": "/dev/sda1",
                    "Ebs": {
                        "VolumeSize": self.bootstrap_root_volume_gb,
                        "VolumeType": "gp3",
                        "DeleteOnTermination": True,
                    },
                }
            ],
        )
        instance_id = resp["Instances"][0]["InstanceId"]
        logger.info("_launch_bootstrap_ec2: instance %s", instance_id)

        ec2.get_waiter("instance_running").wait(InstanceIds=[instance_id])

        desc = ec2.describe_instances(InstanceIds=[instance_id])
        public_ip = desc["Reservations"][0]["Instances"][0].get("PublicIpAddress", "")
        logger.info(
            "_launch_bootstrap_ec2: running in %ds: %s @ %s",
            int(time.time() - t0),
            instance_id,
            public_ip,
        )
        logger.info(
            "_launch_bootstrap_ec2: SSH: ssh -i %s -o IdentitiesOnly=yes ubuntu@%s",
            self.ssh_privkey_path,
            public_ip,
        )
        return {"instance_id": instance_id, "public_ip": public_ip}

    def _provision_docker_service(self, resource: DockerServiceConfig, image_name: str) -> str:
        """Bootstrap a Docker-host AMI from a DockerServiceConfig.

        Pipeline (idempotent):
          1. Ensure S3 bucket, key pair, bootstrap profile.
          2. If sentinel not in S3: launch bootstrap EC2, run Docker pull, wait for sentinel, terminate.
          3. Create AMI from the stopped instance's snapshot.
          4. Return ami_id.

        Unlike the VM bootstrap, the Docker images are pulled into the instance at
        bootstrap time and baked into the AMI — no qcow2 conversion or S3 VHD needed.
        """
        sentinel_key = image_name + ".docker_bootstrap_done"
        failed_key = image_name + ".docker_bootstrap_failed"

        logger.info("_provision_docker_service: %s  images=%s", image_name, resource.docker_images)

        self._ensure_s3_bucket()
        self._ensure_key_pair()

        # Determine if we need to run the bootstrap EC2 phase.
        bootstrap_instance_id: str | None = None
        if not self._s3_object_exists(sentinel_key):
            pull_cmds = "\n".join(
                f"echo '[bootstrap] Pulling {img}...'\ndocker pull {img}" for img in resource.docker_images
            )
            volume_cmds = build_volume_setup_script(resource.volumes)
            script = _AWS_DOCKER_BOOTSTRAP_SCRIPT.format(
                s3_bucket=self.s3_bucket,
                sentinel_key=sentinel_key,
                failed_key=failed_key,
                region=self.region,
                docker_pull_commands=pull_cmds,
                volume_setup_commands=volume_cmds,
            )
            vm_info = self._launch_bootstrap_ec2(script)
            bootstrap_instance_id = vm_info["instance_id"]
            t0 = time.time()
            try:
                logger.info("_provision_docker_service: EC2 running, streaming logs from %s", vm_info["public_ip"])
                logger.info(
                    "_provision_docker_service: SSH: ssh -i %s -o IdentitiesOnly=yes ubuntu@%s",
                    self.ssh_privkey_path,
                    vm_info["public_ip"],
                )
                with BootstrapMonitor(
                    public_ip=vm_info["public_ip"],
                    ssh_privkey=self.ssh_privkey_path,  # type: ignore[arg-type]
                    ssh_user="ubuntu",
                    sentinel_fn=lambda: self._s3_object_exists(sentinel_key),
                ) as monitor:
                    monitor.wait(timeout=3600)
            except Exception:
                logger.warning("_provision_docker_service: bootstrap failed — terminating %s", bootstrap_instance_id)
                self._terminate_instance(bootstrap_instance_id)
                raise
            logger.info("_provision_docker_service: Docker images ready in %.1f min", (time.time() - t0) / 60)
        else:
            logger.info("_provision_docker_service: sentinel exists — checking for existing AMI")

        # Check if AMI already exists (idempotent if previously created but ProvisionStore was lost).
        ec2 = self._ec2()
        resp = ec2.describe_images(
            Owners=["self"],
            Filters=[{"Name": "name", "Values": [image_name]}],
        )
        if resp["Images"]:
            ami_id = resp["Images"][0]["ImageId"]
            logger.info("_provision_docker_service: AMI already exists: %s (%s)", ami_id, image_name)
            if bootstrap_instance_id:
                self._terminate_instance(bootstrap_instance_id)
            return ami_id

        if bootstrap_instance_id is None:
            raise RuntimeError(
                f"Sentinel exists but no AMI found for {image_name!r} and no bootstrap instance running. "
                "Delete the sentinel key and re-run provision()."
            )

        # Stop the instance so the root volume snapshot is consistent.
        try:
            logger.info("_provision_docker_service: stopping %s to create AMI …", bootstrap_instance_id)
            ec2.stop_instances(InstanceIds=[bootstrap_instance_id])
            ec2.get_waiter("instance_stopped").wait(InstanceIds=[bootstrap_instance_id])

            logger.info("_provision_docker_service: creating AMI %r …", image_name)
            resp = ec2.create_image(
                InstanceId=bootstrap_instance_id,
                Name=image_name,
                Description=f"CUBE Docker-host image: {image_name}",
                NoReboot=True,
            )
            ami_id = resp["ImageId"]
            # Default waiter: 40 × 15s = 10 min — too short for large disks.
            # Extend to 120 × 15s = 30 min.
            waiter = ec2.get_waiter("image_available")
            waiter.wait(ImageIds=[ami_id], WaiterConfig={"MaxAttempts": 120})
            ec2.create_tags(Resources=[ami_id], Tags=_dict_to_ec2_tags({**self.tags, "role": "docker-host"}))
            logger.info("_provision_docker_service: AMI ready: %s", ami_id)
        finally:
            self._terminate_instance(bootstrap_instance_id)
        return ami_id

    def _ssh_run(self, public_ip: str, ssh_user: str, script: str) -> None:
        """Run a bash script on the remote host via SSH. Raises on non-zero exit."""
        ssh_run(public_ip, ssh_user, self.ssh_privkey_path, script)  # type: ignore[arg-type]

    def _terminate_instance(self, instance_id: str) -> None:
        """Terminate an EC2 instance and wait for it to be fully terminated."""
        logger.info("_terminate_instance: terminating %s", instance_id)
        try:
            ec2 = self._ec2()
            ec2.terminate_instances(InstanceIds=[instance_id])
            ec2.get_waiter("instance_terminated").wait(InstanceIds=[instance_id])
            logger.info("_terminate_instance: %s terminated", instance_id)
        except Exception as e:
            logger.warning("_terminate_instance: %s: %s", instance_id, e)

    def _bootstrap(self, url: str, image_name: str) -> str:
        """In-cloud bootstrap: spin up EC2 to download, convert, and upload the image.

        Pipeline (idempotent):
          1. Ensure S3 bucket, vmimport role, key pair.
          2. If sentinel not in S3: launch bootstrap EC2, wait for sentinel, terminate.
          3. Import VHD from S3 as EBS snapshot.
          4. Register snapshot as AMI.
          5. Return ami_id.

        Returns ami_id.
        """
        vhd_key = image_name + ".vhd"
        sentinel_key = vhd_key + ".bootstrap_done"
        failed_key = vhd_key + ".bootstrap_failed"

        logger.info("_bootstrap (AWS): %s  source=%s", image_name, url)
        logger.info("_bootstrap: vhd key: s3://%s/%s", self.s3_bucket, vhd_key)

        self._ensure_vmimport_role()
        self._ensure_s3_bucket()
        self._ensure_key_pair()

        if not self._s3_object_exists(sentinel_key):
            script = _AWS_BOOTSTRAP_SCRIPT.format(
                hf_url=url,
                s3_bucket=self.s3_bucket,
                s3_key=vhd_key,
                sentinel_key=sentinel_key,
                failed_key=failed_key,
                region=self.region,
                ssh_pubkey=Path(self.ssh_pubkey_path).read_text().strip(),  # type: ignore[arg-type]
            )
            vm_info = self._launch_bootstrap_ec2(script)
            t0 = time.time()
            try:
                logger.info("_bootstrap: EC2 running, streaming logs from %s", vm_info["public_ip"])
                logger.info(
                    "_bootstrap: SSH: ssh -i %s -o IdentitiesOnly=yes ubuntu@%s",
                    self.ssh_privkey_path,
                    vm_info["public_ip"],
                )
                with BootstrapMonitor(
                    public_ip=vm_info["public_ip"],
                    ssh_privkey=self.ssh_privkey_path,  # type: ignore[arg-type]
                    ssh_user="ubuntu",
                    sentinel_fn=lambda: self._s3_object_exists(sentinel_key),
                ) as monitor:
                    monitor.wait(timeout=7200)
            finally:
                self._terminate_instance(vm_info["instance_id"])
            logger.info("_bootstrap: VHD in S3 (%.1f min)", (time.time() - t0) / 60)
        else:
            logger.info("_bootstrap: sentinel exists — skipping EC2 phase")

        s3_uri = f"s3://{self.s3_bucket}/{vhd_key}"
        snapshot_id = self._import_snapshot(s3_uri, description=image_name, disk_format="VHD")
        return self._register_ami(snapshot_id, image_name)

    # ── Informational ─────────────────────────────────────────────────────────

    def list_images(self) -> list[dict]:
        """Return all CUBE AMIs owned by this account (informational)."""
        ec2 = self._ec2()
        resp = ec2.describe_images(
            Owners=["self"],
            Filters=[{"Name": "tag:project", "Values": ["cube"]}],
        )
        return [
            {"name": img["Name"], "ami_id": img["ImageId"], "state": img["State"]}
            for img in sorted(resp.get("Images", []), key=lambda x: x["Name"])
        ]
