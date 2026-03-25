"""
Bootstrap VM test — Azure + AWS.

Validates the remote bootstrap approach: a cheap ephemeral VM in each cloud
downloads the OSWorld qcow2 from HuggingFace, converts it, uploads it at
datacenter speed, then terminates. No large local upload needed.

Timeline (estimated):
  Azure bootstrap VM:  ~15-20 min (Standard_B2ms, 128 GB data disk)
  AWS bootstrap VM:    ~15-20 min (t3.medium, 128 GB root volume)
  Both run sequentially to avoid saturating HuggingFace bandwidth.

  Azure post-bootstrap (import disk + gallery): ~10 min
  AWS post-bootstrap (import snapshot + AMI):   ~10 min

  Total wall-clock: ~45-60 min

Cost per run:
  Azure bootstrap VM:  ~$0.04  (Standard_B2ms @ $0.087/hr × 20 min + disk)
  AWS bootstrap VM:    ~$0.02  (t3.medium @ $0.047/hr × 20 min)

USAGE
-----
    cd experiments/azure-vm-backend
    .venv/bin/python test_bootstrap.py
"""

from __future__ import annotations

import sys
import time

import aws_pipeline as aws
import cube_azure_pipeline as az

# ── Config ────────────────────────────────────────────────────────────────────

# Public HuggingFace URL for OSWorld Ubuntu image (no auth required)
# Note: the dataset ships Ubuntu.qcow2.zip (not a bare .qcow2)
HF_URL = "https://huggingface.co/datasets/xlangai/ubuntu_osworld/resolve/main/Ubuntu.qcow2.zip"

# Distinct names — no overlap with test_osworld_parallel.py artifacts
AZURE_IMAGE_NAME  = "cube-bootstrap-test"
AWS_IMAGE_NAME    = "cube-bootstrap-test"
AZURE_BLOB_NAME   = "Ubuntu-bootstrap-test.vhd"   # isolated from Ubuntu.vhd
AWS_VHD_KEY       = "Ubuntu-bootstrap-test.vhd"   # isolated from Ubuntu.vhd


# ── Helpers ───────────────────────────────────────────────────────────────────

def _delete_azure_image(name: str) -> None:
    """Delete all versions of an Azure gallery image + the definition itself."""
    from azure.identity import AzureCliCredential
    from azure.mgmt.compute import ComputeManagementClient
    compute = ComputeManagementClient(AzureCliCredential(), az.SUBSCRIPTION)
    try:
        versions = list(compute.gallery_image_versions.list_by_gallery_image(
            az.RESOURCE_GROUP, az.GALLERY_NAME, name,
        ))
        for v in versions:
            print(f"  Deleting gallery version {v.name}...")
            compute.gallery_image_versions.begin_delete(
                az.RESOURCE_GROUP, az.GALLERY_NAME, name, v.name,
            ).result()
        compute.gallery_images.begin_delete(
            az.RESOURCE_GROUP, az.GALLERY_NAME, name,
        ).result()
        print(f"  Gallery image '{name}' deleted.")
    except Exception as e:
        print(f"  (gallery cleanup: {e})")


def _delete_azure_bootstrap_blobs(vhd_blob: str) -> None:
    """Delete the VHD blob and its sentinel/failed markers."""
    svc = az._blob_service_client()
    for suffix in ["", ".bootstrap_done", ".bootstrap_failed"]:
        try:
            svc.get_blob_client(az.CONTAINER_NAME, vhd_blob + suffix).delete_blob()
            print(f"  Deleted blob: {vhd_blob + suffix}")
        except Exception:
            pass


def _deregister_aws_ami(name: str) -> None:
    """Deregister an AMI and delete its snapshot."""
    ec2 = aws._ec2()
    resp = ec2.describe_images(Owners=["self"], Filters=[{"Name": "name", "Values": [name]}])
    for img in resp.get("Images", []):
        ami_id = img["ImageId"]
        snap_ids = [m["Ebs"]["SnapshotId"] for m in img.get("BlockDeviceMappings", []) if "Ebs" in m]
        ec2.deregister_image(ImageId=ami_id)
        print(f"  Deregistered AMI: {ami_id}")
        for snap_id in snap_ids:
            try:
                ec2.delete_snapshot(SnapshotId=snap_id)
                print(f"  Deleted snapshot: {snap_id}")
            except Exception as e:
                print(f"  (snapshot cleanup: {e})")


def _delete_aws_bootstrap_s3(vhd_key: str) -> None:
    """Delete the VHD and its sentinel/failed markers from S3."""
    s3 = aws._s3()
    for suffix in ["", ".bootstrap_done", ".bootstrap_failed"]:
        key = vhd_key + suffix
        try:
            s3.delete_object(Bucket=aws.S3_BUCKET, Key=key)
            print(f"  Deleted s3://{aws.S3_BUCKET}/{key}")
        except Exception:
            pass


# ── Per-cloud test ────────────────────────────────────────────────────────────

def test_azure() -> bool:
    print("\n" + "=" * 60)
    print("  Azure bootstrap test")
    print("=" * 60)

    # Reset state so bootstrap runs end-to-end
    print("\n[reset] Removing existing test artifacts...")
    _delete_azure_image(AZURE_IMAGE_NAME)
    _delete_azure_bootstrap_blobs(AZURE_BLOB_NAME)

    t0 = time.time()
    try:
        # bootstrap_ensure_resource: spin up VM → download → convert → upload → gallery
        az.bootstrap_ensure_resource(HF_URL, AZURE_IMAGE_NAME, blob_name=AZURE_BLOB_NAME)

        # Launch a real VM from the gallery image and probe it
        print(f"\n[launch] Launching VM from bootstrapped gallery image...")
        t_launch = time.time()
        info = az.launch(AZURE_IMAGE_NAME, open_tunnel=True)
        print(f"  Launch ready in {(time.time()-t_launch)/60:.1f} min")

        print(f"\n[probe] {info['endpoint']}")
        probe = az.probe(info["endpoint"])
        print(f"  /screenshot: {probe.get('screenshot_bytes', 0)} bytes")
        print(f"  /execute:    {'ok' if probe.get('execute_ok') else 'failed'}")

        elapsed = (time.time() - t0) / 60
        print(f"\n[AZURE ✅] Done in {elapsed:.1f} min — endpoint: {info['endpoint']}")

        # Cleanup runtime VM (gallery image kept)
        if info.get("tunnel"):
            info["tunnel"].terminate()
        az.stop(info["vm_name"], pip_name=info["pip_name"], nic_name=info["nic_name"])
        return True

    except Exception as e:
        print(f"\n[AZURE ❌] {e}")
        import traceback; traceback.print_exc()
        return False


def test_aws() -> bool:
    print("\n" + "=" * 60)
    print("  AWS bootstrap test")
    print("=" * 60)

    # Reset state
    print("\n[reset] Removing existing test artifacts...")
    _deregister_aws_ami(AWS_IMAGE_NAME)
    _delete_aws_bootstrap_s3(AWS_VHD_KEY)

    t0 = time.time()
    try:
        # bootstrap_ensure_resource: spin up EC2 → download → convert → upload → AMI
        ami_id = aws.bootstrap_ensure_resource(HF_URL, AWS_IMAGE_NAME, vhd_key=AWS_VHD_KEY)
        print(f"  AMI: {ami_id}")

        # Launch a real VM from the AMI and probe it
        print(f"\n[launch] Launching VM from bootstrapped AMI...")
        t_launch = time.time()
        info = aws.launch(AWS_IMAGE_NAME, ssh_user="user", open_tunnel=True)
        print(f"  Launch ready in {(time.time()-t_launch)/60:.1f} min")

        print(f"\n[probe] {info['endpoint']}")
        probe = aws.probe(info["endpoint"])
        print(f"  /screenshot: {probe.get('screenshot_bytes', 0)} bytes")
        print(f"  /execute:    {'ok' if probe.get('execute_ok') else 'failed'}")

        elapsed = (time.time() - t0) / 60
        print(f"\n[AWS ✅] Done in {elapsed:.1f} min — instance: {info['instance_id']}")

        # Cleanup runtime instance (AMI kept)
        if info.get("tunnel"):
            info["tunnel"].terminate()
        aws.stop(info["instance_id"])
        return True

    except Exception as e:
        print(f"\n[AWS ❌] {e}")
        import traceback; traceback.print_exc()
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  CUBE Bootstrap VM test — Azure + AWS")
    print("=" * 60)
    print(f"\nSource: {HF_URL}")
    print(f"Azure image name: {AZURE_IMAGE_NAME}")
    print(f"AWS AMI name:     {AWS_IMAGE_NAME}")
    print("\nExpected total: ~45-60 min, ~$0.06 in cloud costs")

    t_total = time.time()

    azure_ok = test_azure()
    aws_ok   = test_aws()

    print(f"\n{'='*60}")
    print(f"  Results  (total: {(time.time()-t_total)/60:.1f} min)")
    print(f"{'='*60}")
    print(f"  Azure: {'✅ PASSED' if azure_ok else '❌ FAILED'}")
    print(f"  AWS:   {'✅ PASSED' if aws_ok   else '❌ FAILED'}")

    if not azure_ok or not aws_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
