#!/usr/bin/env python3
"""Canary: measure the ``eai job exec`` hang rate on a fresh job.

This smoke deliberately *does not* use the sidecar — it goes through the
``eai job exec`` RPC path directly so we can monitor whether the upstream
bug that motivated this whole package is still present.

The original bug report (``toolkit-hang-bugreport.md``, 2026-04-22) measured
~6-9 % of ``eai job exec`` calls hanging indefinitely against a long-lived
``sleep infinity`` job on yul101.  This canary launches the same shape of
job, fires ``_STRESS_ITERATIONS`` short execs with a tight timeout, and
prints the observed hang rate.

How to read the output
======================
- **Hang rate ≥ 3 %** (expected at time of writing) → bug still present;
  the sidecar bypass remains necessary; SMOKE OK.
- **Hang rate 0 %** over a meaningful sample → upstream likely fixed it;
  open a follow-up to evaluate dropping the sidecar binary and reverting
  to direct exec.  Still SMOKE OK — measurement succeeded, just print the
  result loudly so a reader notices.
- **Hang rate > 30 %** → something is much more broken than the original
  report; SMOKE FAIL so the reader investigates.

This is intentionally a *probe*, not a verdict.  The banner just confirms
the measurement completed; the printed hang rate is the actual signal.

Run from cube-standard repo root:
    uv run cube-resources/cube-infra-toolkit/scripts/smoke/eai_exec_hang_canary.py [--profile yul101]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

NAME = "eai_exec_hang_canary"
_IMAGE = "python:3.13-slim"
_STRESS_ITERATIONS = 30
_PER_CALL_TIMEOUT = 10  # seconds; matches bugreport methodology


def banner(status: str, reason: str = "") -> int:
    line = f"SMOKE {status}: {NAME}"
    if reason:
        line += f": {reason}"
    print(line)
    return {"OK": 0, "FAIL": 1, "SKIP": 2}[status]


def _eai(*args: str, profile: str, timeout: float) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["eai", "--profile", profile, *args], capture_output=True, timeout=timeout
    )


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
    ap.add_argument("--image", default=_IMAGE)
    ap.add_argument("--iterations", type=int, default=_STRESS_ITERATIONS)
    args = ap.parse_args()

    skip = preflight(args.profile)
    if skip:
        return banner("SKIP", skip)

    print(f"Launching minimal sleep-infinity job on {args.profile} ({args.image})…")
    submit = _eai(
        "job",
        "new",
        "--preemptable",
        "--format",
        "json",
        "--no-header",
        "-i",
        args.image,
        "--cpu",
        "1",
        "--mem",
        "2",
        "--",
        "sleep",
        "infinity",
        profile=args.profile,
        timeout=120,
    )
    if submit.returncode != 0:
        return banner("FAIL", f"job submit: {submit.stderr.decode()[:300]}")
    # Output may be a JSON object or a bare ID line — accept either.
    out = submit.stdout.decode().strip()
    if out.startswith("{"):
        import json

        try:
            job_id = json.loads(out)["id"]
        except (ValueError, KeyError) as exc:
            return banner("FAIL", f"could not parse job submit output: {exc} / {out!r}")
    else:
        job_id = out.split("\n", 1)[0]

    print(f"Job {job_id[:8]} submitted; waiting for RUNNING…")
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        r = _eai(
            "job",
            "get",
            job_id,
            "--field",
            "state",
            "--no-header",
            profile=args.profile,
            timeout=30,
        )
        state = r.stdout.decode().strip().upper()
        if state == "RUNNING":
            break
        if state in ("FAILED", "CANCELLED", "KILLED"):
            return banner("FAIL", f"job entered terminal state before running: {state}")
        time.sleep(5)
    else:
        return banner("FAIL", "job did not reach RUNNING within 300s")

    print(f"Job {job_id[:8]} running; firing {args.iterations} sequential `eai job exec` calls…")
    hangs: list[int] = []
    non_zero: list[int] = []
    durations: list[float] = []
    try:
        for i in range(args.iterations):
            t0 = time.monotonic()
            try:
                r = _eai(
                    "job",
                    "exec",
                    job_id,
                    "--",
                    "bash",
                    "-c",
                    f"echo iter_{i}",
                    profile=args.profile,
                    timeout=_PER_CALL_TIMEOUT,
                )
            except subprocess.TimeoutExpired:
                dt = time.monotonic() - t0
                durations.append(dt)
                hangs.append(i)
                print(f"  iter {i:>3}: HANG ({dt:.1f}s)")
                continue
            dt = time.monotonic() - t0
            durations.append(dt)
            if r.returncode != 0:
                non_zero.append(i)
                print(f"  iter {i:>3}: rc={r.returncode} ({dt:.1f}s) stderr={r.stderr.decode()[:120]!r}")
    finally:
        print(f"\nKilling job {job_id[:8]}…")
        try:
            _eai("job", "kill", job_id, profile=args.profile, timeout=60)
        except Exception as exc:  # noqa: BLE001
            print(f"WARN: job kill raised: {exc}", file=sys.stderr)

    rate = len(hangs) / args.iterations
    rate_pct = rate * 100
    avg_dt = sum(durations) / len(durations) if durations else 0.0
    print(
        f"\nMEASURED: {len(hangs)}/{args.iterations} hangs ({rate_pct:.1f} %), "
        f"{len(non_zero)} non-zero, avg per-call duration {avg_dt:.2f}s"
    )

    # Interpretation guidance (see module docstring).
    if rate > 0.30:
        return banner("FAIL", f"hang rate {rate_pct:.1f} % > 30 % — investigate")
    if rate == 0.0 and args.iterations >= 20:
        print(
            "NOTE: 0 hangs measured. If reproducible across runs, EAI may have fixed the "
            "underlying bug — open a follow-up to evaluate dropping the sidecar bypass."
        )
    else:
        print(
            "NOTE: bug still present; sidecar bypass remains necessary."
            if rate > 0
            else "NOTE: 0 hangs this run; bug may be intermittent — re-run a few times before drawing conclusions."
        )
    return banner("OK")


if __name__ == "__main__":
    sys.exit(main())
