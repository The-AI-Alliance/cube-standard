#!/usr/bin/env bash
# Best-effort install of qemu (qemu-img + qemu-system-x86_64), needed only by
# VM-backed cubes (osworld, windows-agent-arena). Invoked lazily by
# LocalInfraConfig._ensure_qemu() when it provisions/launches a VMResourceConfig —
# never for Docker/browser cubes. Safe to run multiple times.
#
# BEST-EFFORT: a failure (e.g. Homebrew's qemu bottle not building on Apple
# Silicon) warns but does not abort — the VM cube then fails later with a clear
# "qemu-system-x86_64 not found" when it actually boots a VM.
set -euo pipefail

qemu_warn() {
  echo "WARNING: could not install qemu — VM-backed cubes (osworld, " \
       "windows-agent-arena) won't run until you install it manually." >&2
}

case "$(uname)" in
  Linux)
    { sudo apt-get update -qq && sudo apt-get install -y qemu-system-x86 qemu-utils; } || qemu_warn
    ;;
  Darwin)
    brew install qemu || qemu_warn
    ;;
  *)
    echo "Unsupported platform: $(uname). Install qemu-system-x86_64 manually for VM cubes." >&2
    ;;
esac
