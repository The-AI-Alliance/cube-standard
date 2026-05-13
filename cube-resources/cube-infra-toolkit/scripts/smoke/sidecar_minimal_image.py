#!/usr/bin/env python3
"""Smoke: exec-relay sidecar boots on a python3-less image (minimal_image).

Mounts ``snow.allac.cube_sidecar`` at /opt/cube-sidecar and verifies an
``exec("echo hi")`` round-trips through the Go relay.  This is the path
that's impossible to exercise on python:3-slim because that image already
has python3 — the sidecar branch never fires.

Run from cube-standard repo root:
    uv run cube-resources/cube-infra-toolkit/scripts/smoke/sidecar_minimal_image.py [--profile yul101]

Skip rules:
  - eai CLI not on PATH or not authed → SMOKE SKIP
  - snow.allac.cube_sidecar not readable → SMOKE SKIP (per-user publish caveat)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

from cube_infra_toolkit.toolkit import ToolkitInfraConfig

from cube.resource import DockerServiceConfig

NAME = "sidecar_minimal_image"


def banner(status: str, reason: str = "") -> int:
    line = f"SMOKE {status}: {NAME}"
    if reason:
        line += f": {reason}"
    print(line)
    return {"OK": 0, "FAIL": 1, "SKIP": 2}[status]


def preflight(profile: str, sidecar_data: str) -> str | None:
    if shutil.which("eai") is None:
        return "eai CLI not on PATH"
    env = {**os.environ, "EAI_PROFILE": profile}
    r = subprocess.run(["eai", "user", "get"], env=env, capture_output=True, timeout=15)
    if r.returncode != 0:
        return f"eai user get failed on {profile}: {r.stderr.decode()[:200]}"
    r = subprocess.run(["eai", "data", "get", sidecar_data], env=env, capture_output=True, timeout=15)
    if r.returncode != 0:
        return f"data {sidecar_data} not accessible on {profile}"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="yul101")
    ap.add_argument("--sidecar-data", default="snow.allac.cube_sidecar")
    # alpine:3.19 ships no python3 by default → sidecar branch is the only one
    # that can fire. alpine:3.19 doesn't work because its /bin/sh shim is
    # dynamically linked against libdl which the EAI runtime doesn't provide.
    ap.add_argument("--image", default="alpine:3.19")
    args = ap.parse_args()

    skip = preflight(args.profile, args.sidecar_data)
    if skip:
        return banner("SKIP", skip)

    infra = ToolkitInfraConfig(
        profile=args.profile,
        preemptable=True,
        sidecar_data=args.sidecar_data,
        assets_data=None,
    )
    resource = DockerServiceConfig(name="cube-sidecar-smoke", docker_images=[args.image])

    print(f"Launching {args.image} on {args.profile} with sidecar={args.sidecar_data}…")
    container = None
    try:
        container = infra.launch(resource)
        print(f"Job {container.id[:8]} running; exec'ing 'echo hi'…")
        t0 = time.monotonic()
        result = container.exec("echo hi", timeout=30)
        dt = time.monotonic() - t0
        print(f"exec returned in {dt:.2f}s: rc={result.exit_code} stdout={result.stdout!r}")
        if result.exit_code != 0 or result.stdout.strip() != "hi":
            return banner("FAIL", f"unexpected result: rc={result.exit_code} stdout={result.stdout!r}")
        return banner("OK")
    except Exception as exc:
        return banner("FAIL", f"{type(exc).__name__}: {exc}")
    finally:
        if container is not None:
            try:
                container.stop()
            except Exception as exc:
                print(f"WARN: stop() raised: {exc}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
