"""
New-user simulation: full ensure_resource → launch → probe → stop pipeline.

Simulates what happens the first time a researcher runs CUBE with AzureVMBackend:
  1. No gallery image exists yet → ensure_resource runs the full pipeline
  2. Downloads image from URL (simulating hf:// download)
  3. Converts qcow2 → fixed VHD
  4. Uploads VHD to Azure Blob Storage
  5. Imports blob → Managed Disk
  6. Publishes Managed Disk → Compute Gallery
  7. Launches VM from gallery, opens SSH tunnel, waits for HTTP
  8. Probes /screenshot and /execute
  9. Restores snapshot (stop + relaunch)
 10. Stops VM (cleanup runtime resources only)

Gallery image and blob are kept after the test (permanent artifacts).
Only the runtime VM (cube-vm-*, cube-nic-*, cube-ip-*) is deleted.

IMAGE CHOICE
------------
Ubuntu 22.04 Server cloud image (~660 MB download, ~2.2 GB VHD).
Same code path as OSWorld Ubuntu.qcow2 but 25x smaller → ~15 min total
vs ~90 min for the full 50 GB OSWorld image.

For a real OSWorld deployment, replace IMAGE_URL with:
    hf://xlangai/ubuntu_osworld/Ubuntu.qcow2
and pass it to AzureVMBackend(hf_qcow2=...).

USAGE
-----
    uv run --extra cube python test_new_user.py
"""

import logging
import sys
import time
import urllib.request
from pathlib import Path

import requests

from azure_vm_backend import AzureVMBackend, _gallery_image_exists
from cube.vm import VMConfig

import cube_azure_pipeline as pipeline

# ── Config ────────────────────────────────────────────────────────────────────

# Ubuntu 22.04 "Jammy" server cloud image — cloud-init enabled, Hyper-V drivers
# pre-installed, ~660 MB download → ~2.2 GB fixed VHD.
IMAGE_URL = "https://cloud-images.ubuntu.com/jammy/current/jammy-server-cloudimg-amd64.img"
IMAGE_CACHE = Path.home() / ".cube" / "cache" / "jammy-server-cloudimg-amd64.img"

# Fresh gallery name — doesn't exist yet, so ensure_resource runs end-to-end.
GALLERY_IMAGE_NAME = "cube-new-user-test"


# ── Helpers ───────────────────────────────────────────────────────────────────

def download_image(url: str, dest: Path) -> str:
    """Download image to dest, showing progress. Skips if already cached."""
    if dest.exists():
        size_gb = dest.stat().st_size / 1024**3
        print(f"[download] Already cached: {dest.name} ({size_gb:.1f} GB) — skipping.")
        return str(dest)

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url}")
    print(f"  → {dest}")
    t0 = time.time()

    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = downloaded / total_size * 100
            mb = downloaded / 1024**2
            total_mb = total_size / 1024**2
            print(f"\r  {pct:.0f}%  {mb:.0f} / {total_mb:.0f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, dest, progress)
    elapsed = time.time() - t0
    size_mb = dest.stat().st_size / 1024**2
    print(f"\n  Done in {elapsed:.0f}s ({size_mb:.0f} MB)")
    return str(dest)


def delete_gallery_image(name: str) -> None:
    """Delete all versions of a gallery image, then the image definition itself.

    Only call this for images created by the test — never for pre-existing ones.
    """
    from azure.identity import AzureCliCredential
    from azure.mgmt.compute import ComputeManagementClient

    compute = ComputeManagementClient(AzureCliCredential(), pipeline.SUBSCRIPTION)
    print(f"[cleanup] Deleting gallery image '{name}' and all its versions...")

    try:
        versions = list(compute.gallery_image_versions.list_by_gallery_image(
            pipeline.RESOURCE_GROUP, pipeline.GALLERY_NAME, name
        ))
        for v in versions:
            print(f"  Deleting version {v.name}...")
            compute.gallery_image_versions.begin_delete(
                pipeline.RESOURCE_GROUP, pipeline.GALLERY_NAME, name, v.name
            ).result()
        compute.gallery_images.begin_delete(
            pipeline.RESOURCE_GROUP, pipeline.GALLERY_NAME, name
        ).result()
        print(f"  Gallery image '{name}' deleted.")
    except Exception as e:
        print(f"  Warning: could not delete gallery image: {e}")


# ── Main test ─────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.WARNING,   # suppress SDK HTTP noise; key steps print() directly
        format="%(asctime)s %(levelname)s %(message)s",
    )

    print("=" * 60)
    print("  CUBE AzureVMBackend — new user simulation")
    print("=" * 60)

    # Safety check: refuse to run if gallery image already exists
    if _gallery_image_exists(pipeline.SUBSCRIPTION, pipeline.RESOURCE_GROUP, pipeline.GALLERY_NAME, GALLERY_IMAGE_NAME):
        print(f"\n[abort] Gallery image '{GALLERY_IMAGE_NAME}' already exists.")
        print("  Delete it first or change GALLERY_IMAGE_NAME to test from scratch.")
        sys.exit(1)

    t_total = time.time()

    # ── Step 1: Download image ────────────────────────────────────────────────
    print(f"\n[step 1/6] Download image")
    image_path = download_image(IMAGE_URL, IMAGE_CACHE)

    # ── Step 2–5: ensure_resource (convert → upload → disk → gallery) ─────────
    print(f"\n[step 2-5/6] ensure_resource (convert → upload → managed disk → gallery)")
    print(f"  image:   {image_path}")
    print(f"  gallery: {GALLERY_IMAGE_NAME}")

    config = VMConfig(snapshot_name=GALLERY_IMAGE_NAME)
    backend = AzureVMBackend(hf_qcow2=image_path)

    t_ensure = time.time()
    backend.ensure_resource(config)
    print(f"  ensure_resource done in {(time.time() - t_ensure)/60:.1f} min")

    # ── Step 6: Launch VM ─────────────────────────────────────────────────────
    print(f"\n[step 6/6] Launch VM from new gallery image")
    t_launch = time.time()
    vm = backend.launch(config)
    print(f"  {vm}")
    print(f"  Launch ready in {(time.time() - t_launch)/60:.1f} min")

    # ── Probe ─────────────────────────────────────────────────────────────────
    print(f"\n[probe] Checking HTTP endpoints on {vm.endpoint}")
    for path, method, body in [
        ("/screenshot", "GET",  None),
        ("/execute",    "POST", {"command": ["uname", "-a"]}),
        ("/execute",    "POST", {"command": ["cat", "/etc/os-release"]}),
    ]:
        r = (requests.get(f"{vm.endpoint}{path}", timeout=10) if method == "GET"
             else requests.post(f"{vm.endpoint}{path}", json=body, timeout=10))
        if r.headers.get("content-type", "").startswith("application/json"):
            detail = r.json().get("stdout", "").strip().split("\n")[0]
            print(f"  {method} {path} → HTTP {r.status_code}  {detail}")
        else:
            print(f"  {method} {path} → HTTP {r.status_code}  {len(r.content)} bytes")

    # ── restore_snapshot ──────────────────────────────────────────────────────
    print(f"\n[restore_snapshot]")
    t_restore = time.time()
    vm.restore_snapshot("init_state")
    print(f"  Done in {(time.time() - t_restore)/60:.1f} min — new endpoint: {vm.endpoint}")

    r = requests.get(f"{vm.endpoint}/screenshot", timeout=10)
    print(f"  GET /screenshot after restore → HTTP {r.status_code}  {len(r.content)} bytes")

    # ── Stop VM ───────────────────────────────────────────────────────────────
    print(f"\n[stop] Deleting runtime VM (gallery image kept)")
    vm.stop()
    print(f"  cube-vm-*, cube-nic-*, cube-ip-* deleted")
    print(f"  Gallery image '{GALLERY_IMAGE_NAME}' retained for future launches")

    print(f"\n{'='*60}")
    print(f"  Total time: {(time.time() - t_total)/60:.1f} min")
    print(f"  Gallery image ready for reuse: {GALLERY_IMAGE_NAME}")
    print(f"  Next launch will skip ensure_resource (~4 min instead of ~{(time.time() - t_total)/60:.0f} min)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
