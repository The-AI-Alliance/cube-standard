#!/usr/bin/env python3
"""Smoke: ``assets_data`` mounts and the bundled uv binary is usable.

Mounts ``snow.allac.cube_uv`` at ``/opt/cube-assets/`` and verifies that
``cp /opt/cube-assets/uv /tmp && chmod +x /tmp/uv && /tmp/uv --version``
succeeds — the same dance terminalbench-cube's evaluator setup performs to
get a working ``uv`` on images that ship without python/curl/apt.

Run from cube-standard repo root:
    uv run cube-resources/cube-infra-toolkit/scripts/smoke/assets_uv.py [--profile yul101]

Skip rules:
  - eai CLI not on PATH or not authed → SMOKE SKIP
  - snow.allac.cube_sidecar OR snow.allac.cube_uv not readable → SMOKE SKIP
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

NAME = "assets_uv"


def banner(status: str, reason: str = "") -> int:
    line = f"SMOKE {status}: {NAME}"
    if reason:
        line += f": {reason}"
    print(line)
    return {"OK": 0, "FAIL": 1, "SKIP": 2}[status]


def preflight(profile: str, *data_names: str) -> str | None:
    if shutil.which("eai") is None:
        return "eai CLI not on PATH"
    env = {**os.environ, "EAI_PROFILE": profile}
    r = subprocess.run(["eai", "user", "get"], env=env, capture_output=True, timeout=15)
    if r.returncode != 0:
        return f"eai user get failed on {profile}: {r.stderr.decode()[:200]}"
    for name in data_names:
        r = subprocess.run(["eai", "data", "get", name], env=env, capture_output=True, timeout=15)
        if r.returncode != 0:
            return f"data {name} not accessible on {profile}"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="yul101")
    ap.add_argument("--sidecar-data", default="snow.allac.cube_sidecar")
    ap.add_argument("--assets-data", default="snow.allac.cube_uv")
    # debian:12-slim matches the libc family (glibc) of real Terminal-Bench task
    # images. alpine ships musl, which the gnu-flavoured uv binary published by
    # publish-cube-sidecar.sh won't run on. The Go sidecar itself is CGO-free
    # and works on either family.
    ap.add_argument("--image", default="debian:12-slim")
    args = ap.parse_args()

    skip = preflight(args.profile, args.sidecar_data, args.assets_data)
    if skip:
        return banner("SKIP", skip)

    infra = ToolkitInfraConfig(
        profile=args.profile,
        preemptable=True,
        sidecar_data=args.sidecar_data,
        assets_data=args.assets_data,
    )
    resource = DockerServiceConfig(name="cube-assets-smoke", docker_images=[args.image])

    print(f"Launching {args.image} on {args.profile} with sidecar+assets…")
    container = None
    try:
        container = infra.launch(resource)
        print(f"Job {container.id[:8]} running; copying + running uv…")
        t0 = time.monotonic()
        result = container.exec(
            "cp /opt/cube-assets/uv /tmp/uv && chmod +x /tmp/uv && /tmp/uv --version",
            timeout=30,
        )
        dt = time.monotonic() - t0
        print(f"exec returned in {dt:.2f}s: rc={result.exit_code} stdout={result.stdout!r}")
        if result.exit_code != 0:
            return banner("FAIL", f"rc={result.exit_code} stderr={result.stderr!r}")
        if not result.stdout.startswith("uv "):
            return banner("FAIL", f"unexpected stdout: {result.stdout!r}")
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
