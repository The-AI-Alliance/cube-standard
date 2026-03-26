"""AWS VM backend for CUBE experiments."""
from __future__ import annotations

import base64
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import boto3
import botocore.exceptions
from _common import BootstrapMonitor, convert_image, free_port, open_tunnel as ssh_tunnel, probe, wait_for_ssh

log = logging.getLogger(__name__)

# ── Bootstrap script ──────────────────────────────────────────────────────────
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

# ── inject SSH into VHD (OSWorld has no openssh-server; cloud-init won't run) ─
echo "[bootstrap] Injecting SSH into VHD..."
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
# This is exactly what 'systemctl enable ssh' does internally.
SSH_SVC=/mnt/guest/lib/systemd/system/ssh.service
SSH_SVC_ALT=/mnt/guest/usr/lib/systemd/system/ssh.service
for svc in "$SSH_SVC" "$SSH_SVC_ALT"; do
    [ -f "$svc" ] && \
        mkdir -p /mnt/guest/etc/systemd/system/multi-user.target.wants && \
        ln -sf "${{svc#/mnt/guest}}" \
            /mnt/guest/etc/systemd/system/multi-user.target.wants/ssh.service && \
        echo "[bootstrap] Enabled sshd via $svc" && break
done
# Create xorg.conf for dummy display so Xorg/GDM work headless in EC2 (no GPU).
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


# ── Backend class ─────────────────────────────────────────────────────────────

@dataclass
class AWSBackend:
    """AWS VM backend: full lifecycle from image import to running EC2 instance."""

    region:                    str  = "us-east-2"
    s3_bucket:                 str  = "cube-vm-images-664283147550"
    instance_type:             str  = "t3.xlarge"
    guest_port:                int  = 5000
    key_pair_name:             str  = "cube-key"
    tags:                      list = field(default_factory=lambda: [{"Key": "project", "Value": "cube-experiment"}])
    ssh_privkey:               str  = field(default_factory=lambda: str(Path.home() / ".ssh" / "id_ed25519"))
    ssh_pubkey:                str  = field(default_factory=lambda: str(Path.home() / ".ssh" / "id_ed25519.pub"))
    bootstrap_instance_type:   str  = "t3.medium"
    bootstrap_root_volume_gb:  int  = 128
    bootstrap_profile_name:    str  = "cube-bootstrap"
    bootstrap_role_name:       str  = "cube-bootstrap-role"

    # ── Private AWS clients ───────────────────────────────────────────────────

    def _ec2(self):
        return boto3.client("ec2", region_name=self.region)

    def _s3(self):
        return boto3.client("s3", region_name=self.region)

    def _iam(self):
        return boto3.client("iam", region_name=self.region)

    # ── One-time account setup ────────────────────────────────────────────────

    def ensure_vmimport_role(self) -> None:
        """Create the 'vmimport' IAM role required by ec2 import-snapshot. Idempotent.

        AWS's VM import service (vmie.amazonaws.com) needs permission to read the
        VHD/VMDK from S3 and create EBS snapshots. This role must be named exactly
        'vmimport' — it's a well-known role name that AWS VM Import looks for by
        convention, not configurable.
        """
        import json as _json
        iam = self._iam()

        trust_policy = _json.dumps({
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "vmie.amazonaws.com"},
                "Action": "sts:AssumeRole",
                "Condition": {"StringEquals": {"sts:ExternalId": "vmimport"}},
            }],
        })

        role_policy = _json.dumps({
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
        })

        try:
            iam.create_role(
                RoleName="vmimport",
                AssumeRolePolicyDocument=trust_policy,
                Description="Allows EC2 VM Import service to access S3 and create snapshots",
            )
            log.info("ensure_vmimport_role: created vmimport IAM role")
        except iam.exceptions.EntityAlreadyExistsException:
            pass

        iam.put_role_policy(
            RoleName="vmimport",
            PolicyName="vmimport-s3-ec2",
            PolicyDocument=role_policy,
        )

    def ensure_s3_bucket(self) -> None:
        """Create the S3 bucket for VHD uploads if it doesn't exist."""
        s3 = self._s3()
        try:
            s3.head_bucket(Bucket=self.s3_bucket)
        except botocore.exceptions.ClientError:
            log.info("ensure_s3_bucket: creating %s in %s", self.s3_bucket, self.region)
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
            log.info("ensure_s3_bucket: created s3://%s", self.s3_bucket)

    def ensure_key_pair(self) -> None:
        """Import our local SSH public key as an EC2 key pair. Idempotent."""
        ec2 = self._ec2()
        try:
            ec2.describe_key_pairs(KeyNames=[self.key_pair_name])
        except botocore.exceptions.ClientError:
            pubkey = Path(self.ssh_pubkey).read_bytes()
            ec2.import_key_pair(KeyName=self.key_pair_name, PublicKeyMaterial=pubkey)
            log.info("ensure_key_pair: imported SSH key pair: %s", self.key_pair_name)

    def ensure_security_group(self) -> str:
        """Create a security group that allows SSH inbound. Returns group ID."""
        ec2 = self._ec2()

        try:
            resp = ec2.describe_security_groups(
                Filters=[{"Name": "group-name", "Values": ["cube-sg"]}]
            )
            if resp["SecurityGroups"]:
                return resp["SecurityGroups"][0]["GroupId"]
        except botocore.exceptions.ClientError:
            pass

        sg = ec2.create_security_group(
            GroupName="cube-sg",
            Description="CUBE experiment - SSH inbound only",
        )
        sg_id = sg["GroupId"]
        ec2.authorize_security_group_ingress(
            GroupId=sg_id,
            IpPermissions=[{
                "IpProtocol": "tcp",
                "FromPort": 22,
                "ToPort": 22,
                "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "SSH"}],
            }],
        )
        ec2.create_tags(Resources=[sg_id], Tags=self.tags)
        log.info("ensure_security_group: created %s", sg_id)
        return sg_id

    def ensure_bootstrap_instance_profile(self) -> str:
        """Create an IAM instance profile that grants bootstrap VMs S3 write access. Idempotent.

        Bootstrap EC2 instances need to write the converted VHD to S3 without
        embedding credentials in the user-data script. An instance profile attached
        at launch gives the VM temporary credentials via the EC2 metadata service
        (IMDSv2), scoped to PutObject on our S3 bucket only.
        Returns the instance profile name.
        """
        import json as _json
        iam = self._iam()

        trust_policy = _json.dumps({
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "ec2.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }],
        })
        s3_policy = _json.dumps({
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Action": ["s3:PutObject", "s3:GetObject", "s3:HeadObject"],
                "Resource": f"arn:aws:s3:::{self.s3_bucket}/*",
            }],
        })

        try:
            iam.create_role(
                RoleName=self.bootstrap_role_name,
                AssumeRolePolicyDocument=trust_policy,
                Description="Bootstrap VMs: S3 write for image conversion",
            )
            log.info("ensure_bootstrap_instance_profile: created IAM role %s", self.bootstrap_role_name)
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
            log.info("ensure_bootstrap_instance_profile: created profile %s", self.bootstrap_profile_name)
            time.sleep(10)  # IAM propagation delay
        except iam.exceptions.EntityAlreadyExistsException:
            pass

        return self.bootstrap_profile_name

    def _latest_ubuntu_ami(self) -> str:
        """Return the latest Ubuntu 22.04 LTS AMI ID for the configured region."""
        ec2 = self._ec2()
        resp = ec2.describe_images(
            Owners=["099720109477"],  # Canonical
            Filters=[
                {"Name": "name", "Values": ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]},
                {"Name": "state", "Values": ["available"]},
                {"Name": "architecture", "Values": ["x86_64"]},
            ],
        )
        images = sorted(resp["Images"], key=lambda x: x["CreationDate"], reverse=True)
        if not images:
            raise RuntimeError("No Ubuntu 22.04 AMI found in region")
        ami_id = images[0]["ImageId"]
        log.info("_latest_ubuntu_ami: Ubuntu 22.04 AMI: %s", ami_id)
        return ami_id

    # ── S3 helpers ────────────────────────────────────────────────────────────

    def s3_object_exists(self, key: str) -> bool:
        """Return True if an S3 object exists in the bootstrap bucket."""
        try:
            self._s3().head_object(Bucket=self.s3_bucket, Key=key)
            return True
        except botocore.exceptions.ClientError:
            return False

    # ── Image conversion ──────────────────────────────────────────────────────

    def convert_to_vmdk(self, image_path: Path, output_path: Path | None = None) -> Path:
        """Convert qcow2/vhd to sparse VMDK for S3 upload."""
        src = image_path.resolve()
        dst = output_path.resolve() if output_path else src.with_suffix(".vmdk")
        convert_image(src, dst, "vmdk", "subformat=monolithicSparse", log)
        return dst

    def convert_to_vhd(self, image_path: Path, output_path: Path | None = None) -> Path:
        """Convert a disk image to a fixed-size VHD."""
        src = image_path.resolve()
        dst = output_path.resolve() if output_path else src.with_suffix(".vhd")
        convert_image(src, dst, "vpc", "subformat=fixed,force_size", log)
        return dst

    # ── S3 upload ─────────────────────────────────────────────────────────────

    def upload_to_s3(self, file_path: Path, s3_key: str | None = None) -> str:
        """Upload a disk image to S3 using multipart upload with progress logging. Idempotent.

        Skips upload if an object with the same key and byte-length already exists.
        Progress is logged at DEBUG level (rate + ETA) during upload.
        Returns the s3://bucket/key URI.
        """
        vhd = file_path.resolve()
        s3_key = s3_key or vhd.name
        size_gb = vhd.stat().st_size / 1024**3

        s3 = self._s3()
        try:
            head = s3.head_object(Bucket=self.s3_bucket, Key=s3_key)
            if head["ContentLength"] == vhd.stat().st_size:
                uri = f"s3://{self.s3_bucket}/{s3_key}"
                log.info("upload_to_s3: already in S3: %s — skipping", uri)
                return uri
        except botocore.exceptions.ClientError:
            pass

        log.info("upload_to_s3: %s (%.1f GB) → s3://%s/%s", vhd.name, size_gb, self.s3_bucket, s3_key)
        t0 = time.time()
        uploaded = [0]
        total = vhd.stat().st_size

        def progress(bytes_transferred: int) -> None:
            uploaded[0] += bytes_transferred
            pct = uploaded[0] / total * 100
            mb = uploaded[0] / 1024**2
            total_mb = total / 1024**2
            elapsed = time.time() - t0
            rate_mb = mb / elapsed if elapsed > 0 else 0
            eta = int((total - uploaded[0]) / (rate_mb * 1024**2)) if rate_mb > 0 else 0
            log.debug(
                "upload_to_s3: %d%%  %.0f/%.0f MB  %.1f MB/s  ETA %dm%02ds",
                pct, mb, total_mb, rate_mb, eta // 60, eta % 60,
            )

        s3.upload_file(
            str(vhd),
            self.s3_bucket,
            s3_key,
            Callback=progress,
            ExtraArgs={"StorageClass": "STANDARD"},
        )
        elapsed = time.time() - t0
        log.info("upload_to_s3: done in %.1f min", elapsed / 60)
        return f"s3://{self.s3_bucket}/{s3_key}"

    # ── Snapshot / AMI ────────────────────────────────────────────────────────

    def import_snapshot(self, s3_uri: str, description: str, disk_format: str = "VMDK") -> str:
        """Import a disk image from S3 as an EBS snapshot via ec2 import-snapshot. Returns snapshot_id.

        Submits an import task and polls until complete (~15-30 min for a 23 GB VHD).
        disk_format must match the actual file: 'VHD' or 'VMDK'.
        Note: ec2 import-snapshot rejects monolithicSparse VMDK — use fixed VHD or
        stream-optimized VMDK. The local ensure_resource() path uses sparse VMDK
        (smaller file), while bootstrap() uses fixed VHD (produced by qemu-img on
        the bootstrap EC2).
        """
        parts = s3_uri.removeprefix("s3://").split("/", 1)
        bucket, key = parts[0], parts[1]

        ec2 = self._ec2()
        log.info("import_snapshot: %s (format=%s)", key, disk_format)
        resp = ec2.import_snapshot(
            Description=description,
            DiskContainer={
                "Description": description,
                "Format": disk_format,
                "UserBucket": {"S3Bucket": bucket, "S3Key": key},
            },
        )
        task_id = resp["ImportTaskId"]
        log.info("import_snapshot: ImportTaskId=%s", task_id)

        log.info("import_snapshot: polling until complete (~5-15 min)...")
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
                log.info("import_snapshot: done in %dm%02ds: %s", elapsed // 60, elapsed % 60, snapshot_id)
                ec2.create_tags(Resources=[snapshot_id], Tags=self.tags)
                return snapshot_id
            if status in ("deleted", "error"):
                raise RuntimeError(f"Import failed: {task}")

            # Log progress at INFO every 60s or when percentage changes
            now = time.time()
            if now - last_log_t >= 60 or progress != last_pct:
                log.info("import_snapshot: [%dm%02ds] %s%%", elapsed // 60, elapsed % 60, progress)
                last_log_t = now
                last_pct = progress
            time.sleep(15)

    def register_ami(self, snapshot_id: str, name: str) -> str:
        """Register an EBS snapshot as a bootable HVM AMI. Returns ami_id. Idempotent.

        EnaSupport=True is required for t3+ instance families (enhanced networking).
        ec2 import-snapshot does not set this automatically — omitting it causes
        InvalidParameterCombination when launching t3+ instances from the AMI.
        """
        ec2 = self._ec2()

        resp = ec2.describe_images(
            Owners=["self"],
            Filters=[{"Name": "name", "Values": [name]}],
        )
        if resp["Images"]:
            ami_id = resp["Images"][0]["ImageId"]
            log.info("register_ami: already registered: %s (%s) — skipping", ami_id, name)
            return ami_id

        log.info("register_ami: registering %s", name)
        resp = ec2.register_image(
            Name=name,
            Description=f"CUBE experiment: {name}",
            Architecture="x86_64",
            RootDeviceName="/dev/sda1",
            VirtualizationType="hvm",
            EnaSupport=True,   # required for t3+ instance families
            BlockDeviceMappings=[{
                "DeviceName": "/dev/sda1",
                "Ebs": {
                    "SnapshotId": snapshot_id,
                    "VolumeType": "gp3",
                    "DeleteOnTermination": True,
                },
            }],
        )
        ami_id = resp["ImageId"]
        ec2.create_tags(Resources=[ami_id], Tags=self.tags)
        log.info("register_ami: registered %s", ami_id)
        return ami_id

    # ── ensure_resource: all steps ────────────────────────────────────────────

    def ensure_resource(self, image_path: Path, name: str) -> str:
        """Full one-time setup: local image file → AMI.

        image_path : local path to .qcow2, .vmdk, or .vhd
        name       : AMI name
        Returns ami_id.
        """
        log.info("ensure_resource: %s  source=%s", name, image_path)
        t_total = time.time()

        self.ensure_vmimport_role()
        self.ensure_s3_bucket()
        self.ensure_key_pair()

        # Convert to sparse VMDK (~23 GB for OSWorld vs 50 GB fixed VHD)
        vmdk_path = self.convert_to_vmdk(image_path)
        s3_uri = self.upload_to_s3(vmdk_path)
        snapshot_id = self.import_snapshot(s3_uri, description=name, disk_format="VMDK")
        ami_id = self.register_ami(snapshot_id, name)

        log.info("ensure_resource: done in %.1f min  ami=%s", (time.time() - t_total) / 60, ami_id)
        return ami_id

    # ── Bootstrap EC2 ─────────────────────────────────────────────────────────

    def launch_bootstrap_ec2(self, script: str) -> dict:
        """Launch an EC2 instance with a bootstrap script and a large root volume.

        Uses the latest Ubuntu 22.04 AMI. The instance profile grants S3 write access.
        Returns {instance_id, public_ip}.
        """
        ec2 = self._ec2()
        ami_id = self._latest_ubuntu_ami()
        sg_id = self.ensure_security_group()
        profile_name = self.ensure_bootstrap_instance_profile()

        user_data = base64.b64encode(script.encode()).decode()

        log.info(
            "launch_bootstrap_ec2: launching (%s, %d GB root)",
            self.bootstrap_instance_type, self.bootstrap_root_volume_gb,
        )
        t0 = time.time()

        resp = ec2.run_instances(
            ImageId=ami_id,
            InstanceType=self.bootstrap_instance_type,
            MinCount=1,
            MaxCount=1,
            KeyName=self.key_pair_name,
            SecurityGroupIds=[sg_id],
            UserData=user_data,
            IamInstanceProfile={"Name": profile_name},
            TagSpecifications=[{
                "ResourceType": "instance",
                "Tags": self.tags + [
                    {"Key": "Name",  "Value": f"cube-bootstrap-{uuid.uuid4().hex[:6]}"},
                    {"Key": "role",  "Value": "bootstrap"},
                ],
            }],
            BlockDeviceMappings=[{
                "DeviceName": "/dev/sda1",
                "Ebs": {
                    "VolumeSize": self.bootstrap_root_volume_gb,
                    "VolumeType": "gp3",
                    "DeleteOnTermination": True,
                },
            }],
        )
        instance_id = resp["Instances"][0]["InstanceId"]
        log.info("launch_bootstrap_ec2: instance %s", instance_id)

        waiter = ec2.get_waiter("instance_running")
        waiter.wait(InstanceIds=[instance_id])

        desc = ec2.describe_instances(InstanceIds=[instance_id])
        public_ip = desc["Reservations"][0]["Instances"][0].get("PublicIpAddress", "")
        log.info("launch_bootstrap_ec2: running in %ds: %s @ %s", int(time.time() - t0), instance_id, public_ip)
        log.info("launch_bootstrap_ec2: SSH: ssh -i %s -o IdentitiesOnly=yes ubuntu@%s", self.ssh_privkey, public_ip)
        log.info("launch_bootstrap_ec2: Logs: ssh ... 'sudo tail -f /var/log/cube-bootstrap.log'")

        return {"instance_id": instance_id, "public_ip": public_ip}

    def cleanup_bootstrap_ec2(self, instance_id: str) -> None:
        """Terminate bootstrap EC2 instance. Root EBS volume auto-deletes."""
        log.info("cleanup_bootstrap_ec2: terminating %s", instance_id)
        ec2 = self._ec2()
        ec2.terminate_instances(InstanceIds=[instance_id])
        waiter = ec2.get_waiter("instance_terminated")
        waiter.wait(InstanceIds=[instance_id])
        log.info("cleanup_bootstrap_ec2: terminated")

    def bootstrap(self, url: str, image_name: str, vhd_key: str | None = None) -> str:
        """In-cloud bootstrap: spin up an EC2 instance to download, convert, and upload the image.

        Faster than ensure_resource() for large remote images:
        - Bootstrap EC2 downloads from HuggingFace at ~120 MB/s (datacenter speed)
        - Converts to fixed VHD with qemu-img
        - Uploads to S3 via boto3
        - Writes a sentinel S3 object when done; local process polls every 30s
        - Bootstrap EC2 terminates automatically after upload

        Uses BootstrapMonitor to SSH-tail /var/log/cube-bootstrap.log in real time.
        Idempotent: skips the EC2 phase if the sentinel S3 object already exists.

        After VHD is in S3, calls import_snapshot() + register_ami() to create the AMI.
        Returns ami_id.
        """
        src_filename = url.rstrip("/").split("/")[-1]
        base_name = src_filename.split(".")[0]
        vhd_key = vhd_key or (base_name + ".vhd")
        sentinel_key = vhd_key + ".bootstrap_done"
        failed_key = vhd_key + ".bootstrap_failed"

        log.info("bootstrap (AWS): %s  source=%s", image_name, url)
        log.info("bootstrap: vhd key: s3://%s/%s", self.s3_bucket, vhd_key)

        self.ensure_vmimport_role()
        self.ensure_s3_bucket()
        self.ensure_key_pair()

        if not self.s3_object_exists(sentinel_key):
            script = _AWS_BOOTSTRAP_SCRIPT.format(
                hf_url=url,
                s3_bucket=self.s3_bucket,
                s3_key=vhd_key,
                sentinel_key=sentinel_key,
                failed_key=failed_key,
                region=self.region,
                ssh_pubkey=Path(self.ssh_pubkey).read_text().strip(),
            )
            vm_info = self.launch_bootstrap_ec2(script)
            t0 = time.time()
            try:
                log.info("bootstrap: EC2 running, streaming logs from %s", vm_info["public_ip"])
                log.info(
                    "bootstrap: SSH: ssh -i %s -o IdentitiesOnly=yes ubuntu@%s",
                    self.ssh_privkey, vm_info["public_ip"],
                )
                with BootstrapMonitor(
                    public_ip=vm_info["public_ip"],
                    ssh_privkey=self.ssh_privkey,
                    ssh_user="ubuntu",
                    sentinel_fn=lambda: self.s3_object_exists(sentinel_key),
                ) as monitor:
                    monitor.wait(timeout=7200)
            finally:
                self.cleanup_bootstrap_ec2(vm_info["instance_id"])
            log.info("bootstrap: VHD in S3 (%.1f min)", (time.time() - t0) / 60)
        else:
            log.info("bootstrap: sentinel exists — skipping EC2 phase")

        s3_uri = f"s3://{self.s3_bucket}/{vhd_key}"
        snapshot_id = self.import_snapshot(s3_uri, description=image_name, disk_format="VHD")
        return self.register_ami(snapshot_id, image_name)

    # ── VM Lifecycle ──────────────────────────────────────────────────────────

    def _make_user_data(self, pubkey: str) -> str:
        """Build cloud-init user-data that injects our SSH public key into the launched VM.

        Injects into 'user' (OSWorld default), 'ubuntu', and 'root' — we don't know
        which account the image has, so we try all three. wait_for_ssh() then probes
        with fallback_users to find which one actually works.
        Returns base64-encoded user-data string (as EC2 run_instances expects).
        """
        script = f"""#!/bin/bash
set -e
export DEBIAN_FRONTEND=noninteractive
# Ensure SSH server is installed and running.
# OSWorld is a desktop image — openssh-server is not pre-installed.
if ! command -v sshd &>/dev/null; then
    apt-get update -qq && apt-get install -y -qq openssh-server
fi
systemctl enable ssh
systemctl start ssh || true
ufw allow ssh 2>/dev/null || true
# Inject SSH key into 'user' account (OSWorld default)
if id user &>/dev/null; then
    mkdir -p /home/user/.ssh
    echo '{pubkey}' >> /home/user/.ssh/authorized_keys
    chmod 700 /home/user/.ssh
    chmod 600 /home/user/.ssh/authorized_keys
    chown -R user:user /home/user/.ssh
fi
# Also inject into ubuntu account if it exists
if id ubuntu &>/dev/null; then
    mkdir -p /home/ubuntu/.ssh
    echo '{pubkey}' >> /home/ubuntu/.ssh/authorized_keys
    chmod 700 /home/ubuntu/.ssh
    chmod 600 /home/ubuntu/.ssh/authorized_keys
    chown -R ubuntu:ubuntu /home/ubuntu/.ssh
fi
# And root
mkdir -p /root/.ssh
echo '{pubkey}' >> /root/.ssh/authorized_keys
chmod 700 /root/.ssh
chmod 600 /root/.ssh/authorized_keys
"""
        return base64.b64encode(script.encode()).decode()

    def launch(
        self,
        name: str,
        ssh_user: str = "user",
        open_tunnel: bool = True,
    ) -> dict:
        """Launch an EC2 instance from a named AMI. Returns a dict with connection info.

        Looks up the AMI by name (must have been created by ensure_resource or bootstrap).
        Injects SSH key via user-data cloud-init script into multiple accounts
        (user/ubuntu/root) since we don't know which account the image has.

        If open_tunnel=True: waits for SSH (trying fallback users), opens
        localhost:{port} → instance:5000 tunnel.

        Returns {instance_id, public_ip, endpoint, tunnel, local_port, ssh_user}.
        endpoint is None if open_tunnel=False.
        """
        ec2 = self._ec2()

        resp = ec2.describe_images(
            Owners=["self"],
            Filters=[{"Name": "name", "Values": [name]}],
        )
        if not resp["Images"]:
            raise RuntimeError(f"AMI '{name}' not found — run ensure_resource or bootstrap first.")
        ami_id = resp["Images"][0]["ImageId"]

        sg_id = self.ensure_security_group()
        pubkey = Path(self.ssh_pubkey).read_text().strip()
        user_data = self._make_user_data(pubkey)

        log.info("launch: starting instance from AMI %s (%s)  type=%s", ami_id, name, self.instance_type)
        t0 = time.time()

        resp = ec2.run_instances(
            ImageId=ami_id,
            InstanceType=self.instance_type,
            MinCount=1,
            MaxCount=1,
            KeyName=self.key_pair_name,
            SecurityGroupIds=[sg_id],
            UserData=user_data,
            TagSpecifications=[{
                "ResourceType": "instance",
                "Tags": self.tags + [{"Key": "Name", "Value": f"cube-vm-{uuid.uuid4().hex[:6]}"}],
            }],
            BlockDeviceMappings=[{
                "DeviceName": "/dev/sda1",
                "Ebs": {"VolumeType": "gp3", "DeleteOnTermination": True},
            }],
        )
        instance_id = resp["Instances"][0]["InstanceId"]
        log.info("launch: instance %s", instance_id)

        waiter = ec2.get_waiter("instance_running")
        waiter.wait(InstanceIds=[instance_id])

        desc = ec2.describe_instances(InstanceIds=[instance_id])
        instance = desc["Reservations"][0]["Instances"][0]
        public_ip = instance.get("PublicIpAddress", "")
        log.info("launch: running in %ds: %s @ %s", int(time.time() - t0), instance_id, public_ip)
        log.info("launch: SSH: ssh -i %s -o IdentitiesOnly=yes %s@%s", self.ssh_privkey, ssh_user, public_ip)

        result: dict = {
            "instance_id": instance_id,
            "public_ip": public_ip,
            "endpoint": None,
            "tunnel": None,
            "local_port": None,
            "ssh_user": ssh_user,
        }

        if not open_tunnel:
            return result

        log.info("launch: waiting for SSH (OSWorld first boot can take up to 10 min)...")
        ssh_user_actual = wait_for_ssh(
            public_ip, ssh_user, self.ssh_privkey,
            fallback_users=["ubuntu", "root"],
            timeout=600,
        )
        result["ssh_user"] = ssh_user_actual

        local_port = free_port()
        log.info("launch: opening tunnel localhost:%d → %s:%d", local_port, public_ip, self.guest_port)
        tunnel = ssh_tunnel(public_ip, ssh_user_actual, self.ssh_privkey, local_port, self.guest_port)
        result.update({
            "endpoint": f"http://localhost:{local_port}",
            "tunnel": tunnel,
            "local_port": local_port,
        })
        return result

    def stop(self, instance_id: str) -> None:
        """Terminate an EC2 instance. The root EBS volume is auto-deleted."""
        log.info("stop: terminating %s", instance_id)
        ec2 = self._ec2()
        ec2.terminate_instances(InstanceIds=[instance_id])
        waiter = ec2.get_waiter("instance_terminated")
        waiter.wait(InstanceIds=[instance_id])
        log.info("stop: terminated")

    def restore_snapshot(self, instance_id: str, name: str, ssh_user: str = "user") -> dict:
        """Reset an EC2 instance to clean state: terminate it and launch a fresh one from the AMI.

        For cloud VMs there is no snapshot mechanism — 'restore' means terminate + re-provision.
        This takes ~3-5 min on AWS (vs ~30s for local QEMU savestate).
        """
        self.stop(instance_id)
        return self.launch(name, ssh_user=ssh_user)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def probe(self, endpoint: str, timeout: int = 300) -> dict:
        """Poll the guest agent at endpoint. See _common.probe for details."""
        return probe(endpoint, timeout=timeout)

    def list_images(self) -> list[dict]:
        """Return all CUBE AMIs in the account."""
        ec2 = self._ec2()
        images = ec2.describe_images(
            Owners=["self"],
            Filters=[{"Name": "tag:project", "Values": ["cube-experiment"]}],
        )["Images"]
        return [
            {"name": img["Name"], "ami_id": img["ImageId"], "state": img["State"]}
            for img in sorted(images, key=lambda x: x["Name"])
        ]
