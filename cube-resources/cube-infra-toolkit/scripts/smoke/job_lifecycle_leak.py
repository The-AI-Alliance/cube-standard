#!/usr/bin/env python3
"""Smoke: ToolkitInfraConfig releases every resource it acquires (#182).

Regression guard for the auto-fix(182) lifecycle hardening. On a real
cluster it asserts the two leak surfaces are closed:

  1. normal teardown — ``handle.close()`` kills the job AND the local
     ``eai job port-forward`` process group (no orphaned ``eai`` child);
  2. crash teardown — a handle dropped WITHOUT close (simulated SIGKILL)
     is still reaped by ``cleanup_stale()`` reading cloud tags only.

SKIP if ``eai`` is absent / not authed (so it is safe in any CI).

Run from cube-standard repo root:
    PATH="$HOME/bin:$PATH" EAI_PROFILE=yul101 \
      uv run cube-resources/cube-infra-toolkit/scripts/smoke/job_lifecycle_leak.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

from cube_infra_toolkit.toolkit import ToolkitInfraConfig

from cube.resource import DockerServiceConfig

_NAME = "job_lifecycle_leak"
# A job still consuming/awaiting cluster resources. CANCELLING/terminal
# means the kill was accepted and the async reap is in flight -> released.
_ACTIVE = {"queued", "queuing", "running"}


def banner(status: str, reason: str = "") -> int:
    print(f"\nSMOKE {status}: {_NAME}" + (f" — {reason}" if reason else ""))
    return {"OK": 0, "FAIL": 1, "SKIP": 2}[status]


def _job_released(eai: str, profile: str, job_id: str, *, deadline_s: int = 25) -> bool:
    """Poll until *job_id* leaves the active states (kill accepted)."""
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        r = subprocess.run(
            [eai, "--profile", profile, "job", "get", job_id, "--fields", "state", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        state = ""
        for ln in r.stdout.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                state = str(json.loads(ln).get("state", "")).lower()
            except json.JSONDecodeError:
                continue
        if state and state not in _ACTIVE:
            return True
        time.sleep(2)
    return False


def _pgroup_alive(pid: int) -> bool:
    """True if the port-forward process group still exists."""
    try:
        os.killpg(os.getpgid(pid), 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def main() -> int:
    if shutil.which("eai") is None and not os.path.exists(os.path.expanduser("~/bin/eai")):
        return banner("SKIP", "eai CLI not found")
    eai = "eai" if shutil.which("eai") else os.path.expanduser("~/bin/eai")
    profile = os.environ.get("EAI_PROFILE", "yul101")

    infra = ToolkitInfraConfig(profile=profile, eai_path=eai, cube_data=None, default_ttl_seconds=600)
    resource = DockerServiceConfig(name="cube-smoke-182", scope="task", docker_images=["python:3.12-slim"])
    infra.provision(resource)

    h_close = h_crash = None
    try:
        h_close = infra.launch(resource)
        h_crash = infra.launch(resource)
        # First exec spins up the relay port-forward child (the historical
        # local leak surface) and registers it on the handle.
        assert "ok" in h_close.exec("echo ok", timeout=30).stdout
        pf_procs = list(h_close._port_forwards.values())  # noqa: SLF001
        if not pf_procs:
            return banner("FAIL", "port-forward proc never spawned — test invalid")
        pf_pid = pf_procs[0].pid
        if not _pgroup_alive(pf_pid):
            return banner("FAIL", "port-forward group dead before close — test invalid")

        # 1) normal teardown must release job + port-forward group.
        close_job = h_close.id
        h_close.close()
        h_close = None
        if not _job_released(eai, profile, close_job):
            return banner("FAIL", f"close() left job {close_job[:8]} in an active state")
        if _pgroup_alive(pf_pid):
            return banner("FAIL", "close() leaked the eai port-forward process group")

        # 2) crash teardown: drop the handle, GC purely from cloud tags.
        crash_id = h_crash.id
        h_crash = None  # simulate SIGKILL: no close(), handle lost
        reaped = ToolkitInfraConfig(profile=profile, eai_path=eai, cube_data=None).cleanup_stale(max_age_seconds=0)
        if crash_id not in reaped:
            return banner("FAIL", f"cleanup_stale did not reap orphaned job {crash_id[:8]}")

        return banner("OK", "close() + cleanup_stale() leak-free (job + proc)")
    except Exception as exc:  # noqa: BLE001
        return banner("FAIL", f"{type(exc).__name__}: {exc}")
    finally:
        for h in (h_close, h_crash):
            if h is not None:
                try:
                    h.close()
                except Exception:  # noqa: BLE001, S110
                    pass
        # Belt-and-suspenders: never let the smoke itself leak.
        try:
            ToolkitInfraConfig(profile=profile, eai_path=eai, cube_data=None).cleanup_stale(max_age_seconds=0)
        except Exception:  # noqa: BLE001, S110
            pass
        subprocess.run(["pkill", "-9", "-f", "/bin/eai .* job port-forward"], capture_output=True)


if __name__ == "__main__":
    sys.exit(main())
