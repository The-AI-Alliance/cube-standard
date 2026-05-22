#!/usr/bin/env bash
# Install system dependencies for LocalInfraConfig (qemu, docker).
# Called automatically by LocalInfraConfig.install() — safe to run multiple times.
set -euo pipefail

case "$(uname)" in
  Linux)
    if ! command -v qemu-system-x86_64 &>/dev/null; then
      sudo apt-get update -qq
      sudo apt-get install -y qemu-system-x86 qemu-utils
    fi
    ;;
  Darwin)
    if ! command -v qemu-system-x86_64 &>/dev/null; then
      brew install qemu
    fi
    ;;
  *)
    echo "Unsupported platform: $(uname). Install qemu-system-x86_64 manually." >&2
    exit 1
    ;;
esac

if command -v qemu-system-x86_64 &>/dev/null; then
  echo "QEMU: $(qemu-system-x86_64 --version | head -1)"
fi

if [ -e /dev/kvm ]; then
  echo "KVM: available"
else
  echo "KVM: not available (VMs will use software emulation)"
fi

if ! command -v docker &>/dev/null; then
  echo "WARNING: docker not found — container-based tasks will not work." >&2
fi
