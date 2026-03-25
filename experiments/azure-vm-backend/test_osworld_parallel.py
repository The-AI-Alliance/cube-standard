"""
Full OSWorld pipeline test — Azure + AWS.

Uses the real OSWorld Ubuntu.qcow2 from ~/.cube/osworld/Ubuntu.qcow2.

Timeline (estimated):
  VHD conversion (50 GB):  ~15-20 min  (sequential, shared)
  Azure upload (50 GB):    ~60-90 min
  AWS upload   (~23 GB VMDK): ~30-60 min
  Azure gallery + launch:  ~15 min
  AWS import + launch:     ~20 min

  Total wall-clock: ~90-120 min (dominated by uploads)

USAGE
-----
    cd experiments/azure-vm-backend
    .venv/bin/python test_osworld_parallel.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from _common import configure_logging
from aws_backend import AWSBackend
from azure_backend import AzureBackend

configure_logging()
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

OSWORLD_QCOW2    = Path.home() / ".cube" / "osworld" / "Ubuntu.qcow2"
AZURE_IMAGE_NAME = "cube-osworld-ubuntu"
AWS_IMAGE_NAME   = "cube-osworld-ubuntu"
OSWORLD_SSH_USER = "user"


# ── Per-cloud pipeline ────────────────────────────────────────────────────────

class CloudResult:
    def __init__(self, cloud: str):
        self.cloud = cloud
        self.success = False
        self.error: str | None = None
        self.timings: dict[str, float] = {}
        self.endpoint: str | None = None
        self.vm_id: str | None = None
        self.tunnel = None
        self.probe_results: dict = {}


def run_azure(vhd_path: Path, result: CloudResult) -> None:
    backend = AzureBackend()
    try:
        t0 = time.time()

        t = time.time()
        backend.ensure_resource(vhd_path, AZURE_IMAGE_NAME)
        result.timings["ensure_resource"] = time.time() - t

        t = time.time()
        info = backend.launch(AZURE_IMAGE_NAME, open_tunnel=True)
        result.timings["launch"] = time.time() - t
        result.vm_id    = info["vm_name"]
        result.endpoint = info["endpoint"]
        result.tunnel   = info.get("tunnel")

        result.probe_results = backend.probe(info["endpoint"])
        result.timings["total"] = time.time() - t0
        result.success = True
        log.info("[AZURE ✅] Done — endpoint: %s  vm: %s", info["endpoint"], info["vm_name"])

    except Exception as e:
        result.error = str(e)
        log.error("[AZURE ❌] %s", e, exc_info=True)


def run_aws(result: CloudResult) -> None:
    backend = AWSBackend()
    try:
        t0 = time.time()

        backend.ensure_vmimport_role()
        backend.ensure_s3_bucket()
        backend.ensure_key_pair()

        # Use sparse VMDK (~23 GB) for faster upload vs fixed VHD (~50 GB)
        t = time.time()
        vmdk_path   = backend.convert_to_vmdk(OSWORLD_QCOW2)
        s3_uri      = backend.upload_to_s3(vmdk_path)
        snap_id     = backend.import_snapshot(s3_uri, description=AWS_IMAGE_NAME, disk_format="VMDK")
        backend.register_ami(snap_id, AWS_IMAGE_NAME)
        result.timings["ensure_resource"] = time.time() - t

        t = time.time()
        info = backend.launch(AWS_IMAGE_NAME, ssh_user=OSWORLD_SSH_USER, open_tunnel=True)
        result.timings["launch"] = time.time() - t
        result.vm_id    = info["instance_id"]
        result.endpoint = info["endpoint"]
        result.tunnel   = info.get("tunnel")

        result.probe_results = backend.probe(info["endpoint"])
        result.timings["total"] = time.time() - t0
        result.success = True
        log.info("[AWS ✅] Done — endpoint: %s  instance: %s", info["endpoint"], info["instance_id"])

    except Exception as e:
        result.error = str(e)
        log.error("[AWS ❌] %s", e, exc_info=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not OSWORLD_QCOW2.exists():
        log.error("OSWorld image not found: %s", OSWORLD_QCOW2)
        log.error("Download from https://huggingface.co/datasets/xlangai/ubuntu_osworld")
        sys.exit(1)

    size_gb = OSWORLD_QCOW2.stat().st_size / 1024**3
    log.info("=" * 60)
    log.info("  CUBE OSWorld pipeline test — Azure + AWS")
    log.info("=" * 60)
    log.info("Image: %s  (%.1f GB)", OSWORLD_QCOW2, size_gb)
    log.info("Azure gallery image: %s", AZURE_IMAGE_NAME)
    log.info("AWS AMI name:        %s", AWS_IMAGE_NAME)
    log.info("Expected total: ~90-120 min")

    t_total = time.time()

    # Step 1: Convert to VHD (shared)
    log.info("[step 1] Convert OSWorld qcow2 → fixed VHD")
    azure_backend = AzureBackend()
    t = time.time()
    vhd_path = azure_backend.convert_to_vhd(OSWORLD_QCOW2)
    log.info("  VHD ready in %.1f min: %s", (time.time() - t) / 60, vhd_path)

    # Step 2: Run both pipelines
    azure_result = CloudResult("azure")
    aws_result   = CloudResult("aws")

    log.info("[step 2] Azure pipeline...")
    run_azure(vhd_path, azure_result)

    log.info("[step 3] AWS pipeline...")
    run_aws(aws_result)

    # Summary
    log.info("")
    log.info("=" * 60)
    log.info("  Results")
    log.info("=" * 60)
    total_min = (time.time() - t_total) / 60

    for r in [azure_result, aws_result]:
        log.info("")
        log.info("%s: %s", r.cloud.upper(), "✅ SUCCESS" if r.success else "❌ FAILED")
        if r.error:
            log.info("  Error: %s", r.error)
        for k, v in r.timings.items():
            log.info("  %-20s: %.1f min", k, v / 60)
        if r.probe_results:
            log.info("  /screenshot: %d bytes", r.probe_results.get("screenshot_bytes", 0))
            log.info("  /execute:    %s", "ok" if r.probe_results.get("execute_ok") else "failed")
        if r.endpoint:
            log.info("  endpoint: %s", r.endpoint)
        if r.vm_id:
            log.info("  vm: %s", r.vm_id)

    log.info("Total wall-clock: %.1f min", total_min)

    # Cleanup runtime VMs only
    log.info("[cleanup] Stopping VMs (AMI and gallery image kept)")
    if azure_result.success and azure_result.vm_id:
        if azure_result.tunnel:
            azure_result.tunnel.terminate()
        try:
            AzureBackend().stop(azure_result.vm_id)
        except Exception as e:
            log.warning("Azure stop: %s", e)

    if aws_result.success and aws_result.vm_id:
        if aws_result.tunnel:
            aws_result.tunnel.terminate()
        try:
            AWSBackend().stop(aws_result.vm_id)
        except Exception as e:
            log.warning("AWS stop: %s", e)

    log.info("Done. Images retained for future launches.")

    if not azure_result.success or not aws_result.success:
        sys.exit(1)


if __name__ == "__main__":
    main()
