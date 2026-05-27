#!/usr/bin/env python3
"""Smoke: CUBE_RUN_ID scopes the reaping tag to a run, so cleanup(run_id) reaps the
whole run — and only that run (auto-fix #206).

Launches 3 jobs: two with `CUBE_RUN_ID=run_a`, one with `CUBE_RUN_ID=run_b`, then
calls `cleanup(run_a)`. Asserts:
  1. both run_a jobs are reaped (cleanup(run_id) reaps the WHOLE run, not one job);
  2. the run_b job is untouched (identity-scoped — no cross-run/cross-session kill).

This is the toolkit half of #206; the cube-harness half exports CUBE_RUN_ID per
experiment and reaps on exit / via a heartbeat startup GC.

SKIP if `eai` is absent / not authed (safe in any CI).

Run from cube-standard repo root with the toolkit venv:
    EAI_PROFILE=yul101 .venv/bin/python \
      cube-resources/cube-infra-toolkit/scripts/smoke/experiment_scoped_reap.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import uuid

from cube_infra_toolkit.toolkit import ToolkitInfraConfig

from cube.resource import DockerServiceConfig

_NAME = "experiment_scoped_reap"
_ACTIVE = {"queued", "queuing", "running"}


def banner(status: str, reason: str = "") -> int:
    print(f"\nSMOKE {status}: {_NAME}" + (f" — {reason}" if reason else ""))
    return {"OK": 0, "FAIL": 1, "SKIP": 2}[status]


def _state(eai: str, profile: str, job_id: str) -> str:
    r = subprocess.run(
        [eai, "--profile", profile, "job", "get", job_id, "--fields", "state", "--format", "json"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    for ln in r.stdout.splitlines():
        ln = ln.strip()
        if ln:
            try:
                return str(json.loads(ln).get("state", "")).lower()
            except json.JSONDecodeError:
                continue
    return ""


def _released(eai: str, profile: str, job_id: str, *, deadline_s: int = 30) -> bool:
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        st = _state(eai, profile, job_id)
        if st and st not in _ACTIVE:
            return True
        time.sleep(2)
    return False


def main() -> int:
    eai_path = "eai" if shutil.which("eai") else os.path.expanduser("~/bin/eai")
    if shutil.which("eai") is None and not os.path.exists(eai_path):
        return banner("SKIP", "eai CLI not found")
    profile = os.environ.get("EAI_PROFILE", "yul101")

    run_a = f"smoke-206-{uuid.uuid4().hex[:8]}"
    run_b = f"smoke-206-{uuid.uuid4().hex[:8]}"
    infra = ToolkitInfraConfig(profile=profile, eai_path=eai_path, cube_data=None, default_ttl_seconds=600)
    resource = DockerServiceConfig(name="cube-smoke-206", scope="task", docker_images=["python:3.12-slim"])
    infra.provision(resource)

    handles = []
    try:
        os.environ["CUBE_RUN_ID"] = run_a
        a1 = infra.launch(resource)
        a2 = infra.launch(resource)
        os.environ["CUBE_RUN_ID"] = run_b
        b1 = infra.launch(resource)
        handles = [a1, a2, b1]

        # Reap only run_a.
        infra.cleanup(run_a)

        if not (_released(eai_path, profile, a1.id) and _released(eai_path, profile, a2.id)):
            return banner("FAIL", f"cleanup({run_a}) did not reap BOTH its jobs ({a1.id[:8]}, {a2.id[:8]})")
        if _state(eai_path, profile, b1.id) not in _ACTIVE:
            return banner("FAIL", f"cleanup({run_a}) wrongly killed run_b's job {b1.id[:8]} — not identity-scoped")
        return banner("OK", f"cleanup(run_a) reaped both run_a jobs; run_b job {b1.id[:8]} untouched")
    except Exception as exc:  # noqa: BLE001
        return banner("FAIL", f"{type(exc).__name__}: {exc}")
    finally:
        os.environ.pop("CUBE_RUN_ID", None)
        for h in handles:
            try:
                h.close()
            except Exception:  # noqa: BLE001, S110
                pass
        # Belt-and-suspenders: reap both smoke runs no matter what.
        for rid in (run_a, run_b):
            try:
                infra.cleanup(rid)
            except Exception:  # noqa: BLE001, S110
                pass


if __name__ == "__main__":
    raise SystemExit(main())
