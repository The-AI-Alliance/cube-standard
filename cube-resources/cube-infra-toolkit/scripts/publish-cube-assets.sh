#!/usr/bin/env bash
# Publish the cube-assets bundle (cube-sidecar Go binary + uv + uvx) to the
# caller's EAI personal account.
#
# Normally you don't need to run this — ``ToolkitInfraConfig.cube_data="auto"``
# (the default) auto-publishes on first ``launch()`` per user.  Use this script
# to force a re-push after rebuilding the Go binary or bumping the pinned uv
# version (see _UV_VERSION in ../src/cube_infra_toolkit/toolkit.py).
#
# Usage:
#   ./publish-cube-assets.sh [<profile>]   # default: yul101

set -euo pipefail

PROFILE="${1:-yul101}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SIDECAR_BIN="$REPO_DIR/src/cube_infra_toolkit/_bin/linux-amd64/cube-sidecar"
UV_VERSION="0.5.18"
UV_URL="https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz"

if ! command -v eai >/dev/null 2>&1; then
    echo "FAIL: 'eai' CLI not on PATH. Install it from https://docs.console.elementai.com/" >&2
    exit 1
fi

USER_ACCOUNT=$(EAI_PROFILE="$PROFILE" eai user get --fields account --no-header 2>/dev/null | tr -d '[:space:]')
if [ -z "$USER_ACCOUNT" ]; then
    echo "FAIL: could not resolve EAI account on profile $PROFILE. Are you logged in?" >&2
    exit 1
fi
DATA_NAME="${USER_ACCOUNT}.cube_assets"
echo "Publishing $DATA_NAME on profile $PROFILE"

echo "── building cube-sidecar Go binary ──"
make -C "$REPO_DIR/sidecar-go" linux-amd64
if [ ! -f "$SIDECAR_BIN" ]; then
    echo "FAIL: build did not produce $SIDECAR_BIN" >&2
    exit 1
fi

STAGING=$(mktemp -d)
trap 'rm -rf "$STAGING"' EXIT

# 1. cube-sidecar
cp "$SIDECAR_BIN" "$STAGING/cube-sidecar"
chmod 0644 "$STAGING/cube-sidecar"  # EAI strips exec bit on mount anyway

# 2. uv + uvx from astral-sh release
echo "── fetching uv ${UV_VERSION} from astral-sh ──"
if ! curl -fsSL "$UV_URL" | tar -xz -C "$STAGING" --strip-components=1; then
    echo "FAIL: could not fetch uv from $UV_URL" >&2
    exit 1
fi

# Sanity check
for f in cube-sidecar uv uvx; do
    if [ ! -f "$STAGING/$f" ]; then
        echo "FAIL: bundle missing $f after staging" >&2
        exit 1
    fi
done
echo "── staged: $(ls "$STAGING" | tr '\n' ' ')──"

echo "── pushing $DATA_NAME ──"
if EAI_PROFILE="$PROFILE" eai data get "$DATA_NAME" >/dev/null 2>&1; then
    EAI_PROFILE="$PROFILE" eai data push "$DATA_NAME" "$STAGING"
else
    EAI_PROFILE="$PROFILE" eai data new "$DATA_NAME" "$STAGING"
fi

echo "── done ──"
echo "Mount with: ToolkitInfraConfig()  # cube_data='auto' resolves to $DATA_NAME"
