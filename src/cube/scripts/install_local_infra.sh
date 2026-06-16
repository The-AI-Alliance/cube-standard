#!/usr/bin/env bash
# Install system dependencies for LocalInfraConfig (qemu, docker).
# Called automatically by LocalInfraConfig.install() — safe to run multiple times.
set -euo pipefail

# QEMU is only needed by VM-backed cubes (osworld, windows-agent-arena). Its install
# is therefore BEST-EFFORT: a failure (e.g. Homebrew's qemu bottle not building on
# Apple Silicon) must NOT abort local-infra setup, or offline / Docker / browser cubes
# — which need no qemu at all — become unrunnable on `local` infra. A VM cube then
# fails later with a clear "qemu-system-x86_64 not found" when it actually boots a VM.
# (Proper fix: scope provisioning to declared task capabilities — cube-standard #191.)
qemu_warn() {
  echo "WARNING: could not install qemu — VM-backed cubes (osworld, windows-agent-arena) " \
       "won't run until you install it manually. Other cubes (offline / Docker / browser) " \
       "are unaffected." >&2
}

case "$(uname)" in
  Linux)
    if ! command -v qemu-system-x86_64 &>/dev/null; then
      { sudo apt-get update -qq && sudo apt-get install -y qemu-system-x86 qemu-utils; } || qemu_warn
    fi
    ;;
  Darwin)
    if ! command -v qemu-system-x86_64 &>/dev/null; then
      brew install qemu || qemu_warn
    fi
    ;;
  *)
    echo "Unsupported platform: $(uname). Install qemu-system-x86_64 manually if you need VM cubes." >&2
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
