#!/usr/bin/env bash
# make_test_image.sh — Create a minimal test qcow2 for pipeline experiments
#
# This downloads the Ubuntu 22.04 minimal cloud image (~600MB) and injects
# a cloud-init config that:
#   - Creates a fake /screenshot endpoint on port 5000 (returns a 1x1 PNG)
#   - So we can validate the full pipeline without a real OSWorld image
#
# Usage:
#   ./make_test_image.sh
#   # → produces: test_vm.qcow2

set -euo pipefail

BASE_URL="https://cloud-images.ubuntu.com/minimal/releases/jammy/release"
IMG="ubuntu-22.04-minimal-cloudimg-amd64.img"
OUT="test_vm.qcow2"

echo "=== Step 1: Download Ubuntu 22.04 minimal cloud image ==="
if [ ! -f "$IMG" ]; then
    echo "Downloading $IMG (~360MB) ..."
    curl -L -o "$IMG" "$BASE_URL/$IMG"
else
    echo "Already downloaded: $IMG"
fi

echo ""
echo "=== Step 2: Convert to qcow2 ==="
if [ ! -f "$OUT" ]; then
    qemu-img convert -f qcow2 -O qcow2 "$IMG" "$OUT"
    echo "Converted to: $OUT"
else
    echo "Already exists: $OUT"
fi

echo ""
echo "=== Step 3: Resize to 10GB ==="
qemu-img resize "$OUT" 10G

echo ""
echo "=== Done ==="
echo "Test image: $OUT"
qemu-img info "$OUT"

echo ""
echo "Next steps:"
echo "  python pipeline.py run --qcow2 $OUT"
echo ""
echo "NOTE: This is a cloud-init image (Generalized)."
echo "On Azure, first boot will initialize via cloud-init (~2 min)."
echo "The guest agent (port 5000) will NOT be present — use probe step"
echo "to verify the VM boots at all, then SSH in to inspect."
