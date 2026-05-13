#!/usr/bin/env python3
"""Smoke: clean error when the relay can't come up.

Launches alpine:3.19 (no python3) with ``sidecar_data=None`` so neither the
sidecar nor the python3 branch of ``relay_startup_args`` can start the
relay.  The first ``.exec()`` should raise ``ExecRelayUnavailable`` with a
clear message — *not* hang silently the way the deleted direct-eai-exec
fallback used to.

Run from cube-standard repo root:
    uv run cube-resources/cube-infra-toolkit/scripts/smoke/missing_relay_error.py [--profile yul101]

Skip rules:
  - eai CLI not on PATH or not authed → SMOKE SKIP
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

from cube_infra_toolkit.container import ExecRelayUnavailable
from cube_infra_toolkit.toolkit import ToolkitInfraConfig

from cube.resource import DockerServiceConfig

NAME = "missing_relay_error"


def banner(status: str, reason: str = "") -> int:
    line = f"SMOKE {status}: {NAME}"
    if reason:
        line += f": {reason}"
    print(line)
    return {"OK": 0, "FAIL": 1, "SKIP": 2}[status]


def preflight(profile: str) -> str | None:
    if shutil.which("eai") is None:
        return "eai CLI not on PATH"
    env = {**os.environ, "EAI_PROFILE": profile}
    r = subprocess.run(["eai", "user", "get"], env=env, capture_output=True, timeout=15)
    if r.returncode != 0:
        return f"eai user get failed on {profile}: {r.stderr.decode()[:200]}"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="yul101")
    ap.add_argument("--image", default="alpine:3.19")
    args = ap.parse_args()

    skip = preflight(args.profile)
    if skip:
        return banner("SKIP", skip)

    # No sidecar_data → no /opt/cube-sidecar mount, alpine has no python3, so
    # relay_startup_args's if/elif chain fires neither branch → no relay → first
    # .exec() should raise ExecRelayUnavailable, not hang.
    infra = ToolkitInfraConfig(
        profile=args.profile,
        preemptable=True,
        sidecar_data=None,
        assets_data=None,
    )
    resource = DockerServiceConfig(name="cube-no-relay-smoke", docker_images=[args.image])

    print(f"Launching {args.image} on {args.profile} with NO sidecar (expect ExecRelayUnavailable)…")
    container = None
    try:
        container = infra.launch(resource)
        print(f"Job {container.id[:8]} running; expecting .exec() to fail loudly…")
        try:
            container.exec("echo this_should_never_print", timeout=30)
        except ExecRelayUnavailable as exc:
            msg = str(exc)
            if "lacks both /opt/cube-sidecar" not in msg and "without an exec-relay" not in msg:
                return banner("FAIL", f"raised ExecRelayUnavailable but message lacks the expected guidance: {msg!r}")
            print(f"Got expected ExecRelayUnavailable: {msg}")
            return banner("OK")
        return banner("FAIL", "exec() returned where ExecRelayUnavailable was expected")
    except Exception as exc:
        return banner("FAIL", f"unexpected {type(exc).__name__}: {exc}")
    finally:
        if container is not None:
            try:
                container.stop()
            except Exception as exc:
                print(f"WARN: stop() raised: {exc}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
