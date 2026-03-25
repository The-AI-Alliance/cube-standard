"""
Full OSWorld pipeline test — Azure + AWS in parallel.

Uses the real OSWorld Ubuntu.qcow2 from ~/.cube/osworld/Ubuntu.qcow2.
Both clouds share the same VHD conversion artifact.

Timeline (estimated):
  VHD conversion (50 GB):  ~15-20 min  (sequential, shared)
  Azure upload (50 GB):    ~60-90 min  ┐
  AWS upload   (50 GB):    ~60-90 min  ┘ parallel
  Azure gallery + launch:  ~15 min
  AWS import + launch:     ~20 min

  Total wall-clock: ~90-120 min (dominated by uploads)

USAGE
-----
    uv run --extra cube python test_osworld_parallel.py
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import requests

import aws_pipeline as aws
import cube_azure_pipeline as az

# ── Config ────────────────────────────────────────────────────────────────────

OSWORLD_QCOW2  = Path.home() / ".cube" / "osworld" / "Ubuntu.qcow2"
AZURE_IMAGE_NAME = "cube-osworld-ubuntu"   # new entry, won't touch cube-osworld-linux
AWS_IMAGE_NAME   = "cube-osworld-ubuntu"
OSWORLD_SSH_USER = "user"                  # OSWorld default user


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


def run_azure(vhd_path: str, result: CloudResult) -> None:
    try:
        t0 = time.time()

        # ensure_resource: upload → disk → gallery
        t = time.time()
        az.ensure_resource(vhd_path, AZURE_IMAGE_NAME)
        result.timings["ensure_resource"] = time.time() - t

        # launch
        t = time.time()
        info = az.launch(AZURE_IMAGE_NAME, open_tunnel=True)
        result.timings["launch"] = time.time() - t
        result.vm_id = info["vm_name"]
        result.endpoint = info["endpoint"]
        result.tunnel = info.get("tunnel")

        # probe
        probe = az.probe(info["endpoint"])
        result.probe_results = probe

        result.timings["total"] = time.time() - t0
        result.success = True
        print(f"\n[AZURE ✅] Done — endpoint: {info['endpoint']}  vm: {info['vm_name']}")

    except Exception as e:
        result.error = str(e)
        print(f"\n[AZURE ❌] {e}")


def run_aws(vhd_path: str, result: CloudResult) -> None:
    try:
        t0 = time.time()

        # One-time account setup
        aws.ensure_vmimport_role()
        aws.ensure_s3_bucket()
        aws.ensure_key_pair()

        # upload → snapshot → AMI
        t = time.time()
        s3_uri  = aws.upload_to_s3(vhd_path)
        snap_id = aws.import_snapshot(s3_uri, description=AWS_IMAGE_NAME)
        ami_id  = aws.register_ami(snap_id, AWS_IMAGE_NAME)
        result.timings["ensure_resource"] = time.time() - t

        # launch
        t = time.time()
        info = aws.launch(AWS_IMAGE_NAME, ssh_user=OSWORLD_SSH_USER, open_tunnel=True)
        result.timings["launch"] = time.time() - t
        result.vm_id = info["instance_id"]
        result.endpoint = info["endpoint"]
        result.tunnel = info.get("tunnel")

        # probe
        probe = aws.probe(info["endpoint"])
        result.probe_results = probe

        result.timings["total"] = time.time() - t0
        result.success = True
        print(f"\n[AWS   ✅] Done — endpoint: {info['endpoint']}  instance: {info['instance_id']}")

    except Exception as e:
        result.error = str(e)
        print(f"\n[AWS   ❌] {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not OSWORLD_QCOW2.exists():
        print(f"[abort] OSWorld image not found: {OSWORLD_QCOW2}")
        print("  Download it first from https://huggingface.co/datasets/xlangai/ubuntu_osworld")
        sys.exit(1)

    size_gb = OSWORLD_QCOW2.stat().st_size / 1024**3
    print("=" * 60)
    print("  CUBE OSWorld pipeline test — Azure + AWS parallel")
    print("=" * 60)
    print(f"\nImage: {OSWORLD_QCOW2}  ({size_gb:.1f} GB on disk)")
    print(f"Azure gallery image: {AZURE_IMAGE_NAME}")
    print(f"AWS AMI name:        {AWS_IMAGE_NAME}")
    print(f"\nExpected total time: ~90-120 min (uploads dominate)")

    t_total = time.time()

    # Step 1: Convert qcow2 → VHD (shared, sequential — only needs to happen once)
    print(f"\n[step 1] Convert OSWorld qcow2 → fixed VHD (50 GB output)")
    t = time.time()
    vhd_path = az.convert_to_vhd(str(OSWORLD_QCOW2))
    print(f"  VHD ready in {(time.time()-t)/60:.1f} min: {vhd_path}")

    # Step 2: Upload + provision on both clouds in parallel
    print(f"\n[step 2] Upload + provision — Azure and AWS running in parallel")
    azure_result = CloudResult("azure")
    aws_result   = CloudResult("aws")

    t_azure = threading.Thread(target=run_azure, args=(vhd_path, azure_result), name="Azure")
    t_aws   = threading.Thread(target=run_aws,   args=(vhd_path, aws_result),   name="AWS")

    t_azure.start()
    t_aws.start()
    t_azure.join()
    t_aws.join()

    # Summary
    print(f"\n{'='*60}")
    print(f"  Results")
    print(f"{'='*60}")
    total_min = (time.time() - t_total) / 60

    for r in [azure_result, aws_result]:
        print(f"\n{r.cloud.upper()}: {'✅ SUCCESS' if r.success else '❌ FAILED'}")
        if r.error:
            print(f"  Error: {r.error}")
        if r.timings:
            for k, v in r.timings.items():
                print(f"  {k:<20}: {v/60:.1f} min")
        if r.probe_results:
            print(f"  /screenshot: {r.probe_results.get('screenshot_bytes', 0)} bytes")
            print(f"  /execute:    {'ok' if r.probe_results.get('execute_ok') else 'failed'}")
        if r.endpoint:
            print(f"  endpoint: {r.endpoint}")
        if r.vm_id:
            print(f"  vm: {r.vm_id}")

    print(f"\nTotal wall-clock: {total_min:.1f} min")

    # ── Cleanup: only stop runtime VMs, keep AMI/gallery image ───────────────
    print(f"\n[cleanup] Stopping VMs (AMI and gallery image kept for future use)")

    if azure_result.success and azure_result.vm_id:
        print(f"  Azure: stopping {azure_result.vm_id}")
        if azure_result.tunnel:
            azure_result.tunnel.terminate()
        try:
            az.stop(azure_result.vm_id)
        except Exception as e:
            print(f"  Warning: {e}")

    if aws_result.success and aws_result.vm_id:
        print(f"  AWS: terminating {aws_result.vm_id}")
        if aws_result.tunnel:
            aws_result.tunnel.terminate()
        try:
            aws.stop(aws_result.vm_id)
        except Exception as e:
            print(f"  Warning: {e}")

    print("\nDone. AMI and gallery image retained for future launches (~4 min each).")

    if not azure_result.success or not aws_result.success:
        sys.exit(1)


if __name__ == "__main__":
    main()
