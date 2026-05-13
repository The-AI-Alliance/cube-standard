#!/usr/bin/env python3
"""Smoke: the whole story on a real Terminal-Bench task image.

Launches ``alexgshaw/overfull-hbox:20251031`` — the LaTeX task image from
Terminal-Bench that surfaced every problem this PR fixes:

  * no ``python3`` in the image → the in-container Python relay can't boot;
    only the mounted Go ``cube-sidecar`` binary can serve ``.exec()``.
  * no ``curl`` / no ``apt`` → the evaluator's ``uv`` bootstrap fails; the
    only working source for ``uv`` is the ``/opt/cube/`` mount.

The smoke launches the image with both ``sidecar_data`` and ``assets_data``
mounted, then runs a sequence of ``.exec()`` calls that exercise every code
path the production harness depends on:

  1. ``echo hi``                              — relay bootstrap + round-trip
  2. ``command -v python3 || echo NONE``      — confirms python3 absent
                                                  (so step 1 went through the
                                                   Go sidecar, not the Python
                                                   relay)
  3. ``pdflatex --version | head -1``         — confirms the LaTeX toolchain
                                                  is what we think it is
  4. ``cp + chmod + /tmp/uv --version``       — assets mount works,
                                                  uv binary runs
  5. 20 × ``echo iter_$i``                    — stress: relay stays healthy
                                                  across many calls (this is
                                                  the path that used to hang
                                                  ~6 % of the time via direct
                                                  eai-exec; we expect 0/20)

Run from cube-standard repo root:
    uv run cube-resources/cube-infra-toolkit/scripts/smoke/terminalbench_latex.py [--profile yul101]

Skip rules:
  - ``eai`` CLI not on PATH or not authed → SMOKE SKIP

The cube-assets bundle (sidecar + uv) is auto-published to the caller's EAI
account on first launch via ``ToolkitInfraConfig.cube_data="auto"`` — no
maintainer-specific data needed.
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

NAME = "terminalbench_latex"
_LATEX_IMAGE = "alexgshaw/overfull-hbox:20251031"
_STRESS_ITERATIONS = 20


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
    ap.add_argument("--image", default=_LATEX_IMAGE)
    args = ap.parse_args()

    skip = preflight(args.profile)
    if skip:
        return banner("SKIP", skip)

    # cube_data defaults to "auto" — bundle is published to the caller's
    # personal account on first call this process, then mounted at /opt/cube.
    infra = ToolkitInfraConfig(profile=args.profile, preemptable=True)
    resource = DockerServiceConfig(name="cube-tbench-smoke", docker_images=[args.image])

    print(f"Launching {args.image} on {args.profile} (cube_data='auto')…")
    container = None
    try:
        container = infra.launch(resource)
        print(f"Job {container.id[:8]} running.\n")

        # 1. relay round-trip
        r = container.exec("echo hi", timeout=30)
        if r.exit_code != 0 or r.stdout.strip() != "hi":
            return banner("FAIL", f"step 1 (echo): rc={r.exit_code} stdout={r.stdout!r}")
        print("1. echo round-trip OK")

        # 2. confirm no python3 (so step 1 *had* to use the Go sidecar)
        r = container.exec("command -v python3 || echo NONE", timeout=30)
        if r.exit_code != 0 or r.stdout.strip() != "NONE":
            return banner(
                "FAIL",
                f"step 2: expected NONE for python3 (else the test isn't exercising the Go "
                f"sidecar path); got rc={r.exit_code} stdout={r.stdout!r}",
            )
        print("2. no python3 in image — confirmed the Go sidecar served step 1")

        # 3. LaTeX toolchain sanity (this is the whole reason this image exists)
        r = container.exec("pdflatex --version 2>&1 | head -1", timeout=30)
        if r.exit_code != 0 or "TeX Live" not in r.stdout and "pdfTeX" not in r.stdout:
            return banner("FAIL", f"step 3 (pdflatex): rc={r.exit_code} stdout={r.stdout!r}")
        print(f"3. pdflatex OK ({r.stdout.strip()})")

        # 4. assets mount + uv runs (the cluster-B fast path)
        r = container.exec(
            "cp /opt/cube/uv /tmp/uv && chmod +x /tmp/uv && /tmp/uv --version",
            timeout=30,
        )
        if r.exit_code != 0 or not r.stdout.startswith("uv "):
            return banner("FAIL", f"step 4 (uv): rc={r.exit_code} stdout={r.stdout!r} stderr={r.stderr!r}")
        print(f"4. assets mount + uv OK ({r.stdout.strip()})")

        # 5. stress: N sequential execs. Original bug was ~6 % hang on eai-exec;
        # we expect 0/N hangs through the relay.
        t0 = time.monotonic()
        hangs = 0
        wrong = 0
        for i in range(_STRESS_ITERATIONS):
            try:
                r = container.exec(f"echo iter_{i}", timeout=15)
            except Exception as exc:
                hangs += 1
                print(f"   iter {i}: HANG/ERROR {type(exc).__name__}: {exc}")
                continue
            if r.exit_code != 0 or r.stdout.strip() != f"iter_{i}":
                wrong += 1
                print(f"   iter {i}: WRONG rc={r.exit_code} stdout={r.stdout!r}")
        dt = time.monotonic() - t0
        print(
            f"5. stress: {_STRESS_ITERATIONS} sequential exec in {dt:.1f}s "
            f"({_STRESS_ITERATIONS / dt:.1f}/s) — {hangs} hangs, {wrong} wrong results"
        )
        if hangs or wrong:
            return banner("FAIL", f"stress: {hangs} hangs + {wrong} wrong over {_STRESS_ITERATIONS}")

        return banner("OK")
    except Exception as exc:
        return banner("FAIL", f"{type(exc).__name__}: {exc}")
    finally:
        if container is not None:
            try:
                container.stop()
                print(f"\nJob {container.id[:8]} stopped.")
            except Exception as exc:
                print(f"WARN: stop() raised: {exc}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
