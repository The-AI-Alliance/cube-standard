"""
CUBE AWS VM Pipeline
====================
AWS equivalent of cube_azure_pipeline.py. Full path from a VM image (qcow2 or
HuggingFace URL) to a running EC2 instance the CUBE harness can talk to.

Two ensure_resource approaches
--------------------------------
1. Local pipeline  — convert + upload from your machine (ensure_resource)
   Steps: convert qcow2 → sparse VMDK → upload to S3 → ec2 import-snapshot → AMI
   Note: sparse VMDK (~23 GB for OSWorld) is faster to upload than fixed VHD (50 GB).

2. Bootstrap VM    — spin up a cheap EC2 to do the heavy lifting (bootstrap_ensure_resource)
   Best for: images on HuggingFace / public URLs, slow local upload.
   Steps: launch t3.medium → download from HF at ~120 MB/s → convert to fixed VHD
          → upload to S3 via boto3 at datacenter speed → signal sentinel → instance terminated
          → ec2 import-snapshot (VHD format) → register AMI
   Timing: ~30-35 min total for 12 GB zip / 50 GB VHD
   Cost:   ~$0.02 (t3.medium @ $0.047/hr × ~30 min + 128 GB gp3 volume)

Key design decisions
---------------------
- IAM instance profile for bootstrap EC2: grants S3 write access without embedding
  credentials. Created automatically and idempotently by ensure_bootstrap_instance_profile().
- Bootstrap uses fixed VHD (not sparse VMDK): ec2 import-snapshot rejects
  monolithicSparse VMDK with "unsupported vmdk file format". Fixed VHD works reliably.
- Local ensure_resource uses sparse VMDK for faster upload; only bootstrap uses VHD.
- S3 sentinel pattern: bootstrap EC2 writes a zero-byte .bootstrap_done object when done;
  caller polls every 30s. A .bootstrap_failed object triggers immediate error propagation.
- boto3 for S3 ops in bootstrap script: Ubuntu 22.04 AMIs don't have the AWS CLI installed.
  We install boto3 via pip3 and use a heredoc Python script instead.
- No Golden Image Policy on AWS (personal account) — AMIs launch directly without gallery.
- SSH tunnel: same pattern as Azure — localhost:{port} → vm:5000.
- OSWorld image user: 'user' (not 'ubuntu') — user-data injects our SSH key.

USAGE
-----
    python aws_pipeline.py bootstrap --url https://huggingface.co/.../Ubuntu.qcow2.zip --name cube-osworld
    python aws_pipeline.py ensure --image path/to/image.qcow2 --name cube-osworld
    python aws_pipeline.py launch --name cube-osworld
    python aws_pipeline.py probe --ip <public-ip>
    python aws_pipeline.py stop --instance i-0abc123
    python aws_pipeline.py list

Tested end-to-end
------------------
- Ubuntu 22.04 cloud image: local pipeline ✓ (2026-03-24)
- OSWorld Ubuntu image (50 GB qcow2): bootstrap VM pipeline ✓ (2026-03-25)
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import boto3
import botocore.exceptions

# ── Configuration ─────────────────────────────────────────────────────────────

REGION         = "us-east-2"
S3_BUCKET      = "cube-vm-images-664283147550"   # account-id suffix avoids global collision
INSTANCE_TYPE  = "t3.xlarge"   # 4 vCPU, 16 GB RAM — sufficient for OSWorld desktop
GUEST_PORT     = 5000
TAGS           = [{"Key": "project", "Value": "cube-experiment"}]

SSH_PRIVKEY    = str(Path.home() / ".ssh" / "id_ed25519")
SSH_PUBKEY     = str(Path.home() / ".ssh" / "id_ed25519.pub")
KEY_PAIR_NAME  = "cube-key"

# ── AWS clients ───────────────────────────────────────────────────────────────

def _ec2():
    return boto3.client("ec2", region_name=REGION)

def _s3():
    return boto3.client("s3", region_name=REGION)

def _iam():
    return boto3.client("iam", region_name=REGION)


# ── One-time account setup ────────────────────────────────────────────────────

def ensure_vmimport_role() -> None:
    """Create the vmimport IAM role required for ec2 import-snapshot.

    Idempotent — safe to call multiple times.
    AWS requires this role to exist before any snapshot import can run.
    """
    iam = _iam()

    trust_policy = json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "vmie.amazonaws.com"},
            "Action": "sts:AssumeRole",
            "Condition": {"StringEquals": {"sts:ExternalId": "vmimport"}},
        }],
    })

    role_policy = json.dumps({
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["s3:GetBucketLocation", "s3:GetObject", "s3:ListBucket"],
                "Resource": [
                    f"arn:aws:s3:::{S3_BUCKET}",
                    f"arn:aws:s3:::{S3_BUCKET}/*",
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

    # Create role (idempotent)
    try:
        iam.create_role(
            RoleName="vmimport",
            AssumeRolePolicyDocument=trust_policy,
            Description="Allows EC2 VM Import service to access S3 and create snapshots",
        )
        print("[setup] Created vmimport IAM role.")
    except iam.exceptions.EntityAlreadyExistsException:
        pass

    # Attach inline policy (idempotent)
    iam.put_role_policy(
        RoleName="vmimport",
        PolicyName="vmimport-s3-ec2",
        PolicyDocument=role_policy,
    )


def ensure_s3_bucket() -> None:
    """Create the S3 bucket for VHD uploads if it doesn't exist."""
    s3 = _s3()
    try:
        s3.head_bucket(Bucket=S3_BUCKET)
    except botocore.exceptions.ClientError:
        print(f"[setup] Creating S3 bucket: {S3_BUCKET} in {REGION}")
        s3.create_bucket(
            Bucket=S3_BUCKET,
            CreateBucketConfiguration={"LocationConstraint": REGION},
        )
        # Block all public access
        s3.put_public_access_block(
            Bucket=S3_BUCKET,
            PublicAccessBlockConfiguration={
                "BlockPublicAcls": True,
                "IgnorePublicAcls": True,
                "BlockPublicPolicy": True,
                "RestrictPublicBuckets": True,
            },
        )
        print(f"  Bucket created: s3://{S3_BUCKET}")


def ensure_key_pair() -> None:
    """Import our local SSH public key as an EC2 key pair. Idempotent."""
    ec2 = _ec2()
    try:
        ec2.describe_key_pairs(KeyNames=[KEY_PAIR_NAME])
    except botocore.exceptions.ClientError:
        pubkey = Path(SSH_PUBKEY).read_bytes()
        ec2.import_key_pair(KeyName=KEY_PAIR_NAME, PublicKeyMaterial=pubkey)
        print(f"[setup] Imported SSH key pair: {KEY_PAIR_NAME}")


def ensure_security_group() -> str:
    """Create a security group that allows SSH inbound. Returns group ID."""
    ec2 = _ec2()

    # Check if already exists
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
    ec2.create_tags(Resources=[sg_id], Tags=TAGS)
    print(f"[setup] Created security group: {sg_id}")
    return sg_id


# ── Step 1: Convert image ─────────────────────────────────────────────────────
# AWS ec2 import-snapshot accepts VMDK (sparse) — much smaller than fixed VHD.
# A 50 GB qcow2 becomes ~50 GB fixed VHD but only ~23 GB sparse VMDK.

def convert_to_vmdk(image_path: str, output_path: str | None = None) -> str:
    """Convert qcow2/vhd to sparse VMDK for S3 upload.

    Sparse VMDK contains only the written sectors (~23 GB for OSWorld Ubuntu)
    vs a fixed VHD which is always the full virtual size (50 GB).
    """
    src = Path(image_path).resolve()
    if output_path is None:
        output_path = str(src.with_suffix(".vmdk"))
    dst = Path(output_path).resolve()

    if dst.exists():
        size_gb = dst.stat().st_size / 1024**3
        print(f"[convert] VMDK already exists: {dst.name} ({size_gb:.1f} GB), skipping.")
        return str(dst)

    result = subprocess.run(
        ["qemu-img", "info", "--output=json", str(src)],
        capture_output=True, text=True, check=True,
    )
    import json as _json
    info = _json.loads(result.stdout)
    fmt = info["format"]
    vsize_gb = info["virtual-size"] / 1024**3
    dsize_gb = info.get("disk-size", info["virtual-size"]) / 1024**3
    print(f"[convert] {src.name}")
    print(f"  format: {fmt}  virtual: {vsize_gb:.1f} GB  on-disk: {dsize_gb:.1f} GB")
    print(f"  → {dst.name} (sparse VMDK, ~{dsize_gb:.1f} GB expected)")

    t0 = time.time()
    subprocess.run(
        ["qemu-img", "convert", "-f", fmt, "-O", "vmdk",
         "-o", "subformat=monolithicSparse", str(src), str(dst)],
        check=True,
    )
    elapsed = time.time() - t0
    actual_gb = dst.stat().st_size / 1024**3
    print(f"  Done in {elapsed:.0f}s ({actual_gb:.1f} GB on disk)")
    return str(dst)


# ── Step 2: Upload VHD to S3 ──────────────────────────────────────────────────

def upload_to_s3(file_path: str) -> str:
    """Upload a disk image to S3 using multipart upload with progress. Idempotent.

    Returns s3://bucket/key URI.
    """
    vhd = Path(file_path).resolve()
    s3_key = vhd.name
    size_gb = vhd.stat().st_size / 1024**3

    # Check if already uploaded (same size)
    s3 = _s3()
    try:
        head = s3.head_object(Bucket=S3_BUCKET, Key=s3_key)
        if head["ContentLength"] == vhd.stat().st_size:
            uri = f"s3://{S3_BUCKET}/{s3_key}"
            print(f"[upload] Already in S3: {uri} — skipping.")
            return uri
    except botocore.exceptions.ClientError:
        pass

    print(f"[upload] {vhd.name} ({size_gb:.1f} GB) → s3://{S3_BUCKET}/{s3_key}")
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
        print(f"\r  {pct:.0f}%  {mb:.0f}/{total_mb:.0f} MB  {rate_mb:.1f} MB/s  ETA {eta//60}m{eta%60:02d}s",
              end="", flush=True)

    s3.upload_file(
        str(vhd),
        S3_BUCKET,
        s3_key,
        Callback=progress,
        ExtraArgs={"StorageClass": "STANDARD"},
    )
    elapsed = time.time() - t0
    print(f"\n  Uploaded in {elapsed/60:.1f} min")
    return f"s3://{S3_BUCKET}/{s3_key}"


# ── Step 3: Import S3 object → EBS snapshot ───────────────────────────────────

def import_snapshot(s3_uri: str, description: str = "cube-snapshot", disk_format: str = "VMDK") -> str:
    """Import a disk image from S3 as an EBS snapshot. Returns snapshot_id.

    disk_format : "VMDK" (default, sparse) or "VHD" (fixed)
    This is async — polls until complete (~15-30 min for a 23 GB VMDK).
    """
    # Parse s3://bucket/key
    parts = s3_uri.removeprefix("s3://").split("/", 1)
    bucket, key = parts[0], parts[1]

    ec2 = _ec2()
    print(f"[import] S3 → EBS snapshot: {key} (format={disk_format})")
    resp = ec2.import_snapshot(
        Description=description,
        DiskContainer={
            "Description": description,
            "Format": disk_format,
            "UserBucket": {"S3Bucket": bucket, "S3Key": key},
        },
    )
    task_id = resp["ImportTaskId"]
    print(f"  ImportTaskId: {task_id}")

    t0 = time.time()
    while True:
        tasks = ec2.describe_import_snapshot_tasks(ImportTaskIds=[task_id])
        task = tasks["ImportSnapshotTasks"][0]["SnapshotTaskDetail"]
        status = task["Status"]
        progress = task.get("Progress", "0")
        description_msg = task.get("StatusMessage", "")
        elapsed = int(time.time() - t0)
        print(f"\r  [{elapsed}s] {status} {progress}%  {description_msg}    ", end="", flush=True)

        if status == "completed":
            snapshot_id = task["SnapshotId"]
            print(f"\n  Done in {elapsed//60}m{elapsed%60:02d}s: {snapshot_id}")
            ec2.create_tags(Resources=[snapshot_id], Tags=TAGS)
            return snapshot_id
        if status == "deleted" or status == "error":
            raise RuntimeError(f"Import failed: {task}")
        time.sleep(15)


# ── Step 4: Register snapshot as AMI ─────────────────────────────────────────

def register_ami(snapshot_id: str, name: str) -> str:
    """Register an EBS snapshot as a bootable HVM AMI. Returns ami_id.

    Idempotent — returns existing AMI if name already registered.
    """
    ec2 = _ec2()

    # Check if already registered
    resp = ec2.describe_images(
        Owners=["self"],
        Filters=[{"Name": "name", "Values": [name]}],
    )
    if resp["Images"]:
        ami_id = resp["Images"][0]["ImageId"]
        print(f"[ami] Already registered: {ami_id} ({name}) — skipping.")
        return ami_id

    print(f"[ami] Registering AMI: {name}")
    resp = ec2.register_image(
        Name=name,
        Description=f"CUBE experiment: {name}",
        Architecture="x86_64",
        RootDeviceName="/dev/sda1",
        VirtualizationType="hvm",
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
    ec2.create_tags(Resources=[ami_id], Tags=TAGS)
    print(f"  AMI registered: {ami_id}")
    return ami_id


# ── ensure_resource: all steps in one call ────────────────────────────────────

def ensure_resource(image_path: str, name: str) -> str:
    """Full one-time setup: image file → AMI.

    image_path : local path to .qcow2, .vmdk, or .vhd
    name       : AMI name (used as gallery image name equivalent)

    Returns ami_id.
    """
    print(f"\n{'='*60}")
    print(f"ensure_resource: {name}")
    print(f"{'='*60}")

    t_total = time.time()
    timings: dict[str, float] = {}

    # One-time account setup
    ensure_vmimport_role()
    ensure_s3_bucket()
    ensure_key_pair()

    # 1. Convert to sparse VMDK (~23 GB for OSWorld vs 50 GB fixed VHD)
    t = time.time()
    vmdk_path = convert_to_vmdk(image_path)
    timings["convert"] = time.time() - t

    # 2. Upload
    t = time.time()
    s3_uri = upload_to_s3(vmdk_path)
    timings["upload"] = time.time() - t

    # 3. Import snapshot
    t = time.time()
    snapshot_id = import_snapshot(s3_uri, description=name, disk_format="VMDK")
    timings["import"] = time.time() - t

    # 4. Register AMI
    t = time.time()
    ami_id = register_ami(snapshot_id, name)
    timings["register"] = time.time() - t

    print(f"\n--- ensure_resource timings ---")
    for step, secs in timings.items():
        print(f"  {step:<10}: {secs/60:.1f} min")
    print(f"  TOTAL      : {(time.time() - t_total)/60:.1f} min")
    print(f"Ready to launch: python aws_pipeline.py launch --name {name}")
    return ami_id


# ── Step 5: Launch instance ───────────────────────────────────────────────────

def _free_port(start: int = 16000) -> int:
    for port in range(start, start + 100):
        try:
            with socket.socket() as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    raise RuntimeError("No free port in 16000-16099")


def _open_tunnel(vm_ip: str, local_port: int, ssh_user: str = "user") -> subprocess.Popen:
    """Open SSH tunnel. Returns process handle — caller must .terminate()."""
    proc = subprocess.Popen(
        [
            "ssh", "-N",
            "-L", f"127.0.0.1:{local_port}:localhost:{GUEST_PORT}",
            "-i", SSH_PRIVKEY,
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=30",
            "-o", "IdentitiesOnly=yes",
            f"{ssh_user}@{vm_ip}",
        ],
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)
    return proc


def _make_user_data(pubkey: str) -> str:
    """cloud-init user-data that injects our SSH public key.

    The OSWorld image has a 'user' account. This script adds our key to
    authorized_keys so we can SSH in without the image's default password.
    Also enables the 'ubuntu' account as fallback.
    """
    script = f"""#!/bin/bash
set -e
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


def launch(name: str, ssh_user: str = "user", open_tunnel: bool = True) -> dict:
    """Launch an EC2 instance from an AMI.

    Returns {{
        "instance_id": str,
        "public_ip": str,
        "endpoint": "http://localhost:{{port}}",
        "tunnel": subprocess.Popen,
        "local_port": int,
        "ssh_user": str,
    }}
    """
    ec2 = _ec2()

    # Find AMI by name
    resp = ec2.describe_images(
        Owners=["self"],
        Filters=[{"Name": "name", "Values": [name]}],
    )
    if not resp["Images"]:
        raise RuntimeError(f"AMI '{name}' not found — run ensure_resource first.")
    ami_id = resp["Images"][0]["ImageId"]

    sg_id = ensure_security_group()
    pubkey = Path(SSH_PUBKEY).read_text().strip()
    user_data = _make_user_data(pubkey)

    print(f"[launch] Starting instance from AMI {ami_id} ({name})")
    print(f"  instance type: {INSTANCE_TYPE}")
    t0 = time.time()

    resp = ec2.run_instances(
        ImageId=ami_id,
        InstanceType=INSTANCE_TYPE,
        MinCount=1,
        MaxCount=1,
        KeyName=KEY_PAIR_NAME,
        SecurityGroupIds=[sg_id],
        UserData=user_data,
        TagSpecifications=[{
            "ResourceType": "instance",
            "Tags": TAGS + [{"Key": "Name", "Value": f"cube-vm-{uuid.uuid4().hex[:6]}"}],
        }],
        BlockDeviceMappings=[{
            "DeviceName": "/dev/sda1",
            "Ebs": {"VolumeType": "gp3", "DeleteOnTermination": True},
        }],
    )
    instance_id = resp["Instances"][0]["InstanceId"]
    print(f"  Instance: {instance_id}")

    # Wait for running + public IP
    print("  Waiting for instance to be running...")
    waiter = ec2.get_waiter("instance_running")
    waiter.wait(InstanceIds=[instance_id])

    desc = ec2.describe_instances(InstanceIds=[instance_id])
    instance = desc["Reservations"][0]["Instances"][0]
    public_ip = instance.get("PublicIpAddress", "")
    print(f"  Running in {int(time.time()-t0)}s: {instance_id} @ {public_ip}")
    print(f"  SSH: ssh -i {SSH_PRIVKEY} -o IdentitiesOnly=yes {ssh_user}@{public_ip}")

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

    # Wait for SSH (user-data runs early, key should be injected quickly)
    print("[launch] Waiting for SSH...")
    deadline = time.time() + 300
    ssh_user_actual = ssh_user
    while time.time() < deadline:
        for try_user in [ssh_user, "ubuntu", "root"]:
            r = subprocess.run(
                [
                    "ssh", "-i", SSH_PRIVKEY,
                    "-o", "IdentitiesOnly=yes",
                    "-o", "StrictHostKeyChecking=no",
                    "-o", "UserKnownHostsFile=/dev/null",
                    "-o", "ConnectTimeout=5",
                    "-o", "BatchMode=yes",
                    f"{try_user}@{public_ip}",
                    "echo OK",
                ],
                capture_output=True, text=True,
            )
            if "OK" in r.stdout:
                ssh_user_actual = try_user
                print(f"  SSH available as {try_user}!")
                break
        else:
            time.sleep(10)
            continue
        break

    result["ssh_user"] = ssh_user_actual
    local_port = _free_port()
    print(f"[launch] Opening tunnel: localhost:{local_port} → {public_ip}:{GUEST_PORT}")
    tunnel = _open_tunnel(public_ip, local_port, ssh_user=ssh_user_actual)
    result.update({
        "endpoint": f"http://localhost:{local_port}",
        "tunnel": tunnel,
        "local_port": local_port,
    })
    return result


# ── Probe ─────────────────────────────────────────────────────────────────────

def probe(endpoint: str, timeout: int = 300) -> dict:
    """Wait for the HTTP server to be ready, then check all endpoints."""
    import requests

    print(f"[probe] Polling {endpoint}/screenshot ...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{endpoint}/screenshot", timeout=5)
            if r.status_code == 200 and len(r.content) > 0:
                print(f"  ✅ /screenshot → HTTP 200, {len(r.content)} bytes")
                r2 = requests.post(f"{endpoint}/execute", json={"command": ["uname", "-a"]}, timeout=10)
                print(f"  ✅ /execute → {r2.json().get('stdout','').strip()}")
                return {"screenshot_bytes": len(r.content), "execute_ok": r2.status_code == 200}
        except Exception:
            pass
        remaining = int(deadline - time.time())
        print(f"  Waiting... ({remaining}s left)")
        time.sleep(10)
    raise TimeoutError(f"HTTP server not ready after {timeout}s")


# ── Stop ──────────────────────────────────────────────────────────────────────

def stop(instance_id: str) -> None:
    """Terminate an EC2 instance. The root EBS volume is auto-deleted."""
    ec2 = _ec2()
    print(f"[stop] Terminating: {instance_id}")
    ec2.terminate_instances(InstanceIds=[instance_id])
    waiter = ec2.get_waiter("instance_terminated")
    waiter.wait(InstanceIds=[instance_id])
    print(f"  Terminated.")


# ── Bootstrap VM: remote ensure_resource ─────────────────────────────────────
#
# Spin up a cheap EC2 instance in the same region as the S3 bucket, download
# the qcow2 from HuggingFace, convert to sparse VMDK, upload to S3, terminate.
# Uses an IAM instance profile so the VM can write to S3 without embedded creds.

BOOTSTRAP_INSTANCE_TYPE = "t3.medium"   # 2 vCPU, 4 GB — sufficient for qemu-img
BOOTSTRAP_ROOT_VOLUME_GB = 128          # holds ~23 GB qcow2 + ~50 GB VHD
BOOTSTRAP_PROFILE_NAME = "cube-bootstrap"
BOOTSTRAP_ROLE_NAME    = "cube-bootstrap-role"

# Script injected via EC2 user_data (base64).
# Placeholders: {hf_url}, {s3_bucket}, {s3_key}, {sentinel_key}, {failed_key}, {region}
_BOOTSTRAP_SCRIPT = """\
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


def ensure_bootstrap_instance_profile() -> str:
    """Create IAM instance profile that grants S3 write access to bootstrap VMs.

    Idempotent — safe to call on every bootstrap_ensure_resource() call.
    Returns the instance profile name.
    """
    iam = _iam()

    trust_policy = json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ec2.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }],
    })
    s3_policy = json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Action": ["s3:PutObject", "s3:GetObject", "s3:HeadObject"],
            "Resource": f"arn:aws:s3:::{S3_BUCKET}/*",
        }],
    })

    try:
        iam.create_role(
            RoleName=BOOTSTRAP_ROLE_NAME,
            AssumeRolePolicyDocument=trust_policy,
            Description="Bootstrap VMs: S3 write for image conversion",
        )
        print(f"[setup] Created IAM role: {BOOTSTRAP_ROLE_NAME}")
    except iam.exceptions.EntityAlreadyExistsException:
        pass

    iam.put_role_policy(
        RoleName=BOOTSTRAP_ROLE_NAME,
        PolicyName="s3-bootstrap-write",
        PolicyDocument=s3_policy,
    )

    try:
        iam.create_instance_profile(InstanceProfileName=BOOTSTRAP_PROFILE_NAME)
        iam.add_role_to_instance_profile(
            InstanceProfileName=BOOTSTRAP_PROFILE_NAME,
            RoleName=BOOTSTRAP_ROLE_NAME,
        )
        print(f"[setup] Created instance profile: {BOOTSTRAP_PROFILE_NAME}")
        time.sleep(10)  # IAM propagation delay
    except iam.exceptions.EntityAlreadyExistsException:
        pass

    return BOOTSTRAP_PROFILE_NAME


def _latest_ubuntu_ami() -> str:
    """Return the latest Ubuntu 22.04 LTS AMI ID for the configured region."""
    ec2 = _ec2()
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
    print(f"[bootstrap-vm] Ubuntu 22.04 AMI: {ami_id}")
    return ami_id


def _s3_object_exists(key: str) -> bool:
    """Return True if an S3 object exists in the bootstrap bucket."""
    try:
        _s3().head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except botocore.exceptions.ClientError:
        return False


def poll_s3_sentinel(
    sentinel_key: str,
    failed_key: str | None = None,
    timeout: int = 7200,
    interval: int = 30,
) -> None:
    """Poll S3 until sentinel object appears (bootstrap done) or failed object appears."""
    s3 = _s3()
    deadline = time.time() + timeout
    t0 = time.time()

    while time.time() < deadline:
        if failed_key:
            try:
                data = s3.get_object(Bucket=S3_BUCKET, Key=failed_key)["Body"].read()
                raise RuntimeError(f"Bootstrap VM reported failure: {data.decode()}")
            except RuntimeError:
                raise
            except Exception:
                pass

        if _s3_object_exists(sentinel_key):
            print(f"\n  Bootstrap complete after {int(time.time()-t0)}s")
            return

        elapsed = int(time.time() - t0)
        remaining = int(deadline - time.time())
        print(f"\r  [{elapsed}s elapsed, {remaining}s remaining] waiting for bootstrap...", end="", flush=True)
        time.sleep(interval)

    raise TimeoutError(f"Bootstrap did not complete within {timeout}s")


def launch_bootstrap_ec2(script: str, root_volume_gb: int = BOOTSTRAP_ROOT_VOLUME_GB) -> dict:
    """Launch an EC2 instance with a bootstrap script and a large root volume.

    Uses the latest Ubuntu 22.04 AMI. The instance profile grants S3 write access.
    Returns {instance_id, public_ip}.
    """
    ec2 = _ec2()
    ami_id = _latest_ubuntu_ami()
    sg_id  = ensure_security_group()
    profile_name = ensure_bootstrap_instance_profile()

    # Encode script as base64 for user_data
    user_data = base64.b64encode(script.encode()).decode()

    print(f"[bootstrap-vm] Launching EC2 ({BOOTSTRAP_INSTANCE_TYPE}, {root_volume_gb} GB root)")
    t0 = time.time()

    resp = ec2.run_instances(
        ImageId=ami_id,
        InstanceType=BOOTSTRAP_INSTANCE_TYPE,
        MinCount=1,
        MaxCount=1,
        KeyName=KEY_PAIR_NAME,
        SecurityGroupIds=[sg_id],
        UserData=user_data,
        IamInstanceProfile={"Name": profile_name},
        TagSpecifications=[{
            "ResourceType": "instance",
            "Tags": TAGS + [
                {"Key": "Name",  "Value": f"cube-bootstrap-{uuid.uuid4().hex[:6]}"},
                {"Key": "role",  "Value": "bootstrap"},
            ],
        }],
        BlockDeviceMappings=[{
            "DeviceName": "/dev/sda1",
            "Ebs": {
                "VolumeSize": root_volume_gb,
                "VolumeType": "gp3",
                "DeleteOnTermination": True,
            },
        }],
    )
    instance_id = resp["Instances"][0]["InstanceId"]
    print(f"  Instance: {instance_id}")

    waiter = ec2.get_waiter("instance_running")
    waiter.wait(InstanceIds=[instance_id])

    desc = ec2.describe_instances(InstanceIds=[instance_id])
    public_ip = desc["Reservations"][0]["Instances"][0].get("PublicIpAddress", "")
    print(f"  Running in {int(time.time()-t0)}s: {instance_id} @ {public_ip}")
    print(f"  SSH (for debugging): ssh -i {SSH_PRIVKEY} -o IdentitiesOnly=yes ubuntu@{public_ip}")
    print(f"  Logs: ssh ... 'sudo tail -f /var/log/cube-bootstrap.log'")

    return {"instance_id": instance_id, "public_ip": public_ip}


def cleanup_bootstrap_ec2(instance_id: str) -> None:
    """Terminate bootstrap EC2 instance. Root EBS volume auto-deletes."""
    print(f"[bootstrap-vm] Terminating {instance_id}...")
    ec2 = _ec2()
    ec2.terminate_instances(InstanceIds=[instance_id])
    waiter = ec2.get_waiter("instance_terminated")
    waiter.wait(InstanceIds=[instance_id])
    print("  Bootstrap instance terminated.")


def bootstrap_ensure_resource(hf_url: str, name: str, vhd_key: str | None = None) -> str:
    """Remote bootstrap: spin up an EC2 instance to download + convert + upload.

    Replaces the local upload steps with an in-cloud operation that runs at
    datacenter speed (~15-20 min vs hours from home broadband).
    After this returns, the VHD is in S3 and the downstream steps
    (import_snapshot, register_ami) run as usual.

    hf_url  : HTTPS URL to the source .qcow2 (HuggingFace public repo)
    name    : AMI name
    Returns ami_id.
    """
    # Strip all extensions: "Ubuntu.qcow2.zip" → "Ubuntu"
    src_filename = hf_url.rstrip("/").split("/")[-1]
    base_name    = src_filename.split(".")[0]
    vhd_key      = vhd_key if vhd_key else (base_name + ".vhd")
    sentinel_key = vhd_key + ".bootstrap_done"
    failed_key   = vhd_key + ".bootstrap_failed"

    print(f"\n{'='*60}")
    print(f"bootstrap_ensure_resource (AWS): {name}")
    print(f"  source:   {hf_url}")
    print(f"  vhd key:  s3://{S3_BUCKET}/{vhd_key}")
    print(f"{'='*60}")

    # One-time account setup
    ensure_vmimport_role()
    ensure_s3_bucket()
    ensure_key_pair()

    # Idempotent: skip bootstrap if VHD already in S3
    if _s3_object_exists(sentinel_key):
        print("[bootstrap] Sentinel exists — VHD already bootstrapped.")
    else:
        t_bootstrap = time.time()

        script = _BOOTSTRAP_SCRIPT.format(
            hf_url=hf_url,
            s3_bucket=S3_BUCKET,
            s3_key=vhd_key,
            sentinel_key=sentinel_key,
            failed_key=failed_key,
            region=REGION,
        )

        vm_info = launch_bootstrap_ec2(script)

        try:
            print("\n[bootstrap] EC2 is running. Polling for completion every 30s...")
            print(f"  (watch logs: ssh -i {SSH_PRIVKEY} ubuntu@{vm_info['public_ip']}"
                  f" 'sudo tail -f /var/log/cube-bootstrap.log')")
            poll_s3_sentinel(sentinel_key, failed_key=failed_key)
        finally:
            cleanup_bootstrap_ec2(vm_info["instance_id"])

        print(f"[bootstrap] VHD in S3. Bootstrap took {(time.time()-t_bootstrap)/60:.1f} min")

    # Downstream steps: import snapshot → register AMI (unchanged)
    s3_uri      = f"s3://{S3_BUCKET}/{vhd_key}"
    snapshot_id = import_snapshot(s3_uri, description=name, disk_format="VHD")
    ami_id      = register_ami(snapshot_id, name)
    return ami_id


# ── List AMIs ─────────────────────────────────────────────────────────────────

def list_images() -> None:
    ec2 = _ec2()
    images = ec2.describe_images(
        Owners=["self"],
        Filters=[{"Name": "tag:project", "Values": ["cube-experiment"]}],
    )["Images"]
    if not images:
        print("No CUBE AMIs found.")
        return
    print(f"\n{'Name':<35} {'AMI ID':<25} {'State'}")
    print("-" * 70)
    for img in sorted(images, key=lambda x: x["Name"]):
        print(f"{img['Name']:<35} {img['ImageId']:<25} {img['State']}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CUBE AWS VM Pipeline")
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("ensure"); p.add_argument("--image", required=True); p.add_argument("--name", required=True)
    p = sub.add_parser("launch"); p.add_argument("--name", required=True); p.add_argument("--user", default="user")
    p = sub.add_parser("probe");  p.add_argument("--endpoint", required=True)
    p = sub.add_parser("stop");   p.add_argument("--instance", required=True)
    sub.add_parser("list")

    args = parser.parse_args()

    if args.cmd == "ensure":
        ensure_resource(args.image, args.name)
    elif args.cmd == "launch":
        result = launch(args.name, ssh_user=args.user)
        print(json.dumps({k: v for k, v in result.items() if k != "tunnel"}, indent=2))
        if result.get("endpoint"):
            probe(result["endpoint"])
            input("\nPress Enter to stop instance...")
            stop(result["instance_id"])
            result["tunnel"].terminate()
    elif args.cmd == "probe":
        probe(args.endpoint)
    elif args.cmd == "stop":
        stop(args.instance)
    elif args.cmd == "list":
        list_images()
    else:
        parser.print_help()
        sys.exit(1)
