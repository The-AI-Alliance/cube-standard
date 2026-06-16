#!/usr/bin/env python3
"""Smoke: DaytonaContainer.exec (async-submit + poll) works end-to-end on real Daytona.

The old `run_async=False` path held one HTTP read open for the whole command, so any
command running past the ~135s `proxy.app.daytona.io` read-timeout crashed with
ContainerExecError. This smoke drives the **rewritten** `exec` against a live sandbox
and asserts:

  - stdout / stderr are captured separately, exit codes propagate;
  - a **>135s** command (`sleep 200`) completes cleanly — the exact case that used to
    crash — proving the long wait is now short idempotent polls, not one long read.

    PYTHONPATH=cube-resources/cube-infra-daytona/src \
      python cube-resources/cube-infra-daytona/scripts/smoke/daytona_exec_async.py

Prints `SMOKE OK/FAIL/SKIP: daytona_exec_async` and exits 0/1/2.
"""

import os
import sys
import time

from cube_infra_daytona import DaytonaInfraConfig

from cube.resource import DockerServiceConfig


def _banner(status: str, msg: str) -> int:
    print(f"SMOKE {status}: daytona_exec_async — {msg}")
    return {"OK": 0, "FAIL": 1, "SKIP": 2}[status]


def main() -> int:
    if not os.environ.get("DAYTONA_API_KEY"):
        return _banner("SKIP", "DAYTONA_API_KEY not set")

    c = DaytonaInfraConfig().launch(
        DockerServiceConfig(
            name="exec-async-smoke",
            scope="task",
            requires={"container:root"},
            docker_images=["alexgshaw/fix-git:20251031"],
            default_ttl_seconds=1800,
        )
    )
    print(f"sandbox live: {c.id}", flush=True)
    try:
        r = c.exec("echo hello-stdout", timeout=30)
        if r.stdout != "hello-stdout" or r.exit_code != 0:
            return _banner("FAIL", f"echo: stdout={r.stdout!r} exit={r.exit_code}")
        print(f"  echo: stdout={r.stdout!r} exit={r.exit_code} ✓", flush=True)

        r = c.exec("echo oops 1>&2", timeout=30)
        if r.stderr != "oops" or r.exit_code != 0:
            return _banner("FAIL", f"stderr: stderr={r.stderr!r} exit={r.exit_code}")
        print(f"  stderr: stderr={r.stderr!r} ✓", flush=True)

        r = c.exec("exit 3", timeout=30)
        if r.exit_code != 3:
            return _banner("FAIL", f"exit3: exit={r.exit_code}")
        print(f"  exit3: exit={r.exit_code} ✓", flush=True)

        r = c.exec("env | grep -q SMOKE_VAR && echo present", timeout=30, env={"SMOKE_VAR": "1"})
        if r.stdout != "present":
            return _banner("FAIL", f"env: stdout={r.stdout!r}")
        print(f"  env passthrough: stdout={r.stdout!r} ✓", flush=True)

        # The decisive case: a command past the old ~135s read-timeout ceiling.
        t0 = time.monotonic()
        r = c.exec("sleep 200 && echo slept", timeout=260)
        dt = time.monotonic() - t0
        if r.stdout != "slept" or r.exit_code != 0:
            return _banner("FAIL", f">135s: stdout={r.stdout!r} exit={r.exit_code} after {dt:.0f}s")
        print(f"  >135s sleep: stdout={r.stdout!r} exit={r.exit_code} in {dt:.0f}s ✓", flush=True)
    finally:
        c.stop()

    return _banner("OK", "async-poll exec captured streams + survived a >135s command")


if __name__ == "__main__":
    sys.exit(main())
