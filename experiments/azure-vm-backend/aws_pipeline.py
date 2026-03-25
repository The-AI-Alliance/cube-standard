"""
CUBE AWS VM Pipeline
====================
AWS equivalent of cube_azure_pipeline.py.

ensure_resource():
    1. Convert qcow2/vmdk → fixed VHD  (shared with Azure pipeline)
    2. Upload VHD to S3 (multipart, with progress)
    3. ec2 import-snapshot → EBS snapshot
    4. Register snapshot as AMI

launch():
    5. RunInstances from AMI + key pair injection
    6. Wait for SSH, open tunnel: localhost:{port} → vm:5000
    → returns endpoint URL

restore_snapshot():
    stop() + launch()

stop():
    terminate instance + release elastic IP (if any)

USAGE
-----
    python aws_pipeline.py ensure --image path/to/image.qcow2 --name cube-osworld
    python aws_pipeline.py launch --name cube-osworld
    python aws_pipeline.py probe --ip <public-ip>
    python aws_pipeline.py stop --instance i-0abc123
    python aws_pipeline.py list

NOTES
-----
- No Golden Image Policy on AWS (personal account) — AMI launch works directly
- SSH key injected via key pair (os_profile equivalent) + user-data fallback
- SSH tunnel same pattern as Azure (bypasses any proxy)
- vmimport IAM role created automatically on first run (one-time)
- OSWorld image user: 'user' (not 'ubuntu') — user-data injects our SSH key
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
        Description="CUBE experiment — SSH inbound only",
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
# Reuses cube_azure_pipeline.convert_to_vhd — same VHD works for both clouds.

def convert_to_vhd(image_path: str, output_path: str | None = None) -> str:
    """Convert qcow2/vmdk to fixed VHD. Shared with Azure pipeline."""
    import cube_azure_pipeline as az
    return az.convert_to_vhd(image_path, output_path)


# ── Step 2: Upload VHD to S3 ──────────────────────────────────────────────────

def upload_to_s3(vhd_path: str) -> str:
    """Upload VHD to S3 using multipart upload with progress. Idempotent.

    Returns s3://bucket/key URI.
    """
    vhd = Path(vhd_path).resolve()
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

def import_snapshot(s3_uri: str, description: str = "cube-snapshot") -> str:
    """Import VHD from S3 as an EBS snapshot. Returns snapshot_id.

    This is async — polls until complete (~8-15 min for a 2 GB VHD).
    """
    # Parse s3://bucket/key
    parts = s3_uri.removeprefix("s3://").split("/", 1)
    bucket, key = parts[0], parts[1]

    ec2 = _ec2()
    print(f"[import] S3 → EBS snapshot: {key}")
    resp = ec2.import_snapshot(
        Description=description,
        DiskContainer={
            "Description": description,
            "Format": "VHD",
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

    # 1. Convert
    t = time.time()
    vhd_path = convert_to_vhd(image_path)
    timings["convert"] = time.time() - t

    # 2. Upload
    t = time.time()
    s3_uri = upload_to_s3(vhd_path)
    timings["upload"] = time.time() - t

    # 3. Import snapshot
    t = time.time()
    snapshot_id = import_snapshot(s3_uri, description=name)
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
