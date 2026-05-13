#!/usr/bin/env bash
# Publish the cube-sidecar Go binary (and optionally a uv asset blob) to EAI data.
#
# Today we publish under the caller's personal account (snow.<user>) because
# snow.shared is admin-locked. Anyone with toolkit access can read their own
# account's data but not other users' — so each developer who wants to drive
# minimal images via ToolkitContainer needs to run this once per profile.
#
# Usage:
#   ./publish-cube-sidecar.sh [<profile>]   # default: yul101
#
# Idempotent: re-running overwrites the existing data blob with the freshly-
# built binary. Tag-and-rebuild before re-publishing if you bumped the Go code.

set -euo pipefail

PROFILE="${1:-yul101}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN_DIR="$REPO_DIR/src/cube_infra_toolkit/_bin/linux-amd64"

if ! command -v eai >/dev/null 2>&1; then
    echo "FAIL: 'eai' CLI not on PATH. Install it from https://docs.console.elementai.com/" >&2
    exit 1
fi

# Resolve the caller's account from `eai user get` so we publish to
# snow.<their_user> regardless of who runs this.
USER_ACCOUNT=$(EAI_PROFILE="$PROFILE" eai user get --fields account --no-header 2>/dev/null | tr -d '[:space:]')
if [ -z "$USER_ACCOUNT" ]; then
    echo "FAIL: could not resolve EAI account on profile $PROFILE. Are you logged in? Try 'eai login'." >&2
    exit 1
fi
echo "Publishing under $USER_ACCOUNT on profile $PROFILE"

# ── 1. cube-sidecar ───────────────────────────────────────────────────────────
echo "── building cube-sidecar Go binary ──"
make -C "$REPO_DIR/sidecar-go" linux-amd64

if [ ! -f "$BIN_DIR/cube-sidecar" ]; then
    echo "FAIL: build did not produce $BIN_DIR/cube-sidecar" >&2
    exit 1
fi

STAGING=$(mktemp -d)
trap 'rm -rf "$STAGING"' EXIT
cp "$BIN_DIR/cube-sidecar" "$STAGING/cube-sidecar"
chmod 0644 "$STAGING/cube-sidecar"   # EAI strips exec bit on mount anyway

SIDECAR_NAME="${USER_ACCOUNT}.cube_sidecar"
echo "── pushing $SIDECAR_NAME ──"
if EAI_PROFILE="$PROFILE" eai data get "$SIDECAR_NAME" >/dev/null 2>&1; then
    # Already exists — push a new version. `eai data push` adds a new commit.
    EAI_PROFILE="$PROFILE" eai data push "$SIDECAR_NAME" "$STAGING"
else
    EAI_PROFILE="$PROFILE" eai data new "$SIDECAR_NAME" "$STAGING"
fi

# ── 2. cube_uv (optional) ─────────────────────────────────────────────────────
# Mounted at /opt/cube-assets/ for images that lack python3/curl/apt, so the
# evaluator's uv-bootstrap path can copy a static uv out of the mount.
UV_NAME="${USER_ACCOUNT}.cube_uv"
echo "── fetching uv binaries from astral-sh release ──"
UV_VERSION="0.5.18"
UV_URL="https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz"
UV_STAGING="$STAGING/uv"
mkdir -p "$UV_STAGING"
if ! curl -fsSL "$UV_URL" | tar -xz -C "$UV_STAGING" --strip-components=1; then
    echo "WARN: could not fetch uv from $UV_URL — skipping cube_uv push." >&2
    echo "── done (cube-sidecar only) ──"
    exit 0
fi

echo "── pushing $UV_NAME ──"
if EAI_PROFILE="$PROFILE" eai data get "$UV_NAME" >/dev/null 2>&1; then
    EAI_PROFILE="$PROFILE" eai data push "$UV_NAME" "$UV_STAGING"
else
    EAI_PROFILE="$PROFILE" eai data new "$UV_NAME" "$UV_STAGING"
fi

echo "── done ──"
echo "Mount with: ToolkitInfraConfig(sidecar_data='$SIDECAR_NAME', assets_data='$UV_NAME')"
