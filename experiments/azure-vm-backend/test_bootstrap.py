"""
Bootstrap VM test — Azure + AWS.

Validates the remote bootstrap approach: a cheap ephemeral VM in each cloud
downloads the OSWorld qcow2 from HuggingFace, converts it, uploads it at
datacenter speed, then terminates. No large local upload needed.

Timeline (estimated):
  Azure bootstrap VM:  ~15-20 min (Standard_B2ms, 128 GB OS disk)
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

import logging
import sys
import time

from _common import configure_logging
from aws_backend import AWSBackend
from azure_backend import AzureBackend
from azure.identity import AzureCliCredential
from azure.mgmt.compute import ComputeManagementClient

configure_logging()
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

HF_URL = "https://huggingface.co/datasets/xlangai/ubuntu_osworld/resolve/main/Ubuntu.qcow2.zip"

AZURE_IMAGE_NAME = "cube-bootstrap-test"
AWS_IMAGE_NAME   = "cube-bootstrap-test"
AZURE_BLOB_NAME  = "Ubuntu-bootstrap-test.vhd"
AWS_VHD_KEY      = "Ubuntu-bootstrap-test.vhd"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _delete_azure_image(backend: AzureBackend, name: str) -> None:
    """Delete all versions of an Azure gallery image + the definition itself."""
    compute = ComputeManagementClient(AzureCliCredential(), backend.subscription)
    try:
        versions = list(compute.gallery_image_versions.list_by_gallery_image(
            backend.resource_group, backend.gallery_name, name,
        ))
        for v in versions:
            log.info("  Deleting gallery version %s...", v.name)
            compute.gallery_image_versions.begin_delete(
                backend.resource_group, backend.gallery_name, name, v.name,
            ).result()
        compute.gallery_images.begin_delete(
            backend.resource_group, backend.gallery_name, name,
        ).result()
        log.info("  Gallery image '%s' deleted.", name)
    except Exception as e:
        log.debug("  gallery cleanup: %s", e)


def _delete_azure_bootstrap_blobs(backend: AzureBackend, vhd_blob: str) -> None:
    """Delete the VHD blob and its sentinel/failed markers."""
    svc = backend._blob_service_client()
    for suffix in ["", ".bootstrap_done", ".bootstrap_failed"]:
        try:
            svc.get_blob_client(backend.container_name, vhd_blob + suffix).delete_blob()
            log.info("  Deleted blob: %s", vhd_blob + suffix)
        except Exception:
            pass


def _deregister_aws_ami(backend: AWSBackend, name: str) -> None:
    """Deregister an AMI and delete its snapshot."""
    ec2 = backend._ec2()
    resp = ec2.describe_images(Owners=["self"], Filters=[{"Name": "name", "Values": [name]}])
    for img in resp.get("Images", []):
        ami_id = img["ImageId"]
        snap_ids = [m["Ebs"]["SnapshotId"] for m in img.get("BlockDeviceMappings", []) if "Ebs" in m]
        ec2.deregister_image(ImageId=ami_id)
        log.info("  Deregistered AMI: %s", ami_id)
        for snap_id in snap_ids:
            try:
                ec2.delete_snapshot(SnapshotId=snap_id)
                log.info("  Deleted snapshot: %s", snap_id)
            except Exception as e:
                log.debug("  snapshot cleanup: %s", e)


def _delete_aws_bootstrap_s3(backend: AWSBackend, vhd_key: str) -> None:
    """Delete the VHD and its sentinel/failed markers from S3."""
    s3 = backend._s3()
    for suffix in ["", ".bootstrap_done", ".bootstrap_failed"]:
        key = vhd_key + suffix
        try:
            s3.delete_object(Bucket=backend.s3_bucket, Key=key)
            log.info("  Deleted s3://%s/%s", backend.s3_bucket, key)
        except Exception:
            pass


# ── Per-cloud test ────────────────────────────────────────────────────────────

def test_azure() -> bool:
    log.info("")
    log.info("=" * 60)
    log.info("  Azure bootstrap test")
    log.info("=" * 60)

    backend = AzureBackend()

    log.info("[reset] Removing existing test artifacts...")
    _delete_azure_image(backend, AZURE_IMAGE_NAME)
    _delete_azure_bootstrap_blobs(backend, AZURE_BLOB_NAME)

    t0 = time.time()
    try:
        backend.bootstrap(HF_URL, AZURE_IMAGE_NAME, blob_name=AZURE_BLOB_NAME)

        log.info("[launch] Launching VM from bootstrapped gallery image...")
        t_launch = time.time()
        info = backend.launch(AZURE_IMAGE_NAME, open_tunnel=True)
        log.info("  Launch ready in %.1f min", (time.time() - t_launch) / 60)

        log.info("[probe] %s", info["endpoint"])
        result = backend.probe(info["endpoint"])
        log.info("  /screenshot: %d bytes", result.get("screenshot_bytes", 0))
        log.info("  /execute:    %s", "ok" if result.get("execute_ok") else "failed")

        log.info("[AZURE ✅] Done in %.1f min — endpoint: %s",
                 (time.time() - t0) / 60, info["endpoint"])

        if info.get("tunnel"):
            info["tunnel"].terminate()
        backend.stop(info["vm_name"], pip_name=info["pip_name"], nic_name=info["nic_name"])
        return True

    except Exception as e:
        log.error("[AZURE ❌] %s", e, exc_info=True)
        return False


def test_aws() -> bool:
    log.info("")
    log.info("=" * 60)
    log.info("  AWS bootstrap test")
    log.info("=" * 60)

    backend = AWSBackend()

    log.info("[reset] Removing existing test artifacts...")
    _deregister_aws_ami(backend, AWS_IMAGE_NAME)
    _delete_aws_bootstrap_s3(backend, AWS_VHD_KEY)

    t0 = time.time()
    try:
        ami_id = backend.bootstrap(HF_URL, AWS_IMAGE_NAME, vhd_key=AWS_VHD_KEY)
        log.info("  AMI: %s", ami_id)

        log.info("[launch] Launching VM from bootstrapped AMI...")
        t_launch = time.time()
        info = backend.launch(AWS_IMAGE_NAME, ssh_user="user", open_tunnel=True)
        log.info("  Launch ready in %.1f min", (time.time() - t_launch) / 60)

        log.info("[probe] %s", info["endpoint"])
        result = backend.probe(info["endpoint"])
        log.info("  /screenshot: %d bytes", result.get("screenshot_bytes", 0))
        log.info("  /execute:    %s", "ok" if result.get("execute_ok") else "failed")

        log.info("[AWS ✅] Done in %.1f min — instance: %s",
                 (time.time() - t0) / 60, info["instance_id"])

        if info.get("tunnel"):
            info["tunnel"].terminate()
        backend.stop(info["instance_id"])
        return True

    except Exception as e:
        log.error("[AWS ❌] %s", e, exc_info=True)
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=" * 60)
    log.info("  CUBE Bootstrap VM test — Azure + AWS")
    log.info("=" * 60)
    log.info("Source: %s", HF_URL)
    log.info("Azure image: %s", AZURE_IMAGE_NAME)
    log.info("AWS AMI:     %s", AWS_IMAGE_NAME)
    log.info("Expected total: ~45-60 min, ~$0.06 in cloud costs")

    t_total = time.time()
    azure_ok = test_azure()
    aws_ok   = test_aws()

    log.info("")
    log.info("=" * 60)
    log.info("  Results  (total: %.1f min)", (time.time() - t_total) / 60)
    log.info("=" * 60)
    log.info("  Azure: %s", "✅ PASSED" if azure_ok else "❌ FAILED")
    log.info("  AWS:   %s", "✅ PASSED" if aws_ok   else "❌ FAILED")

    if not azure_ok or not aws_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
