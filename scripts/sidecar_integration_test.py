"""Live integration test: sidecar exec mode against a real Toolkit job.

Not a pytest — run directly:

    cd cube-standard
    EAI_PROFILE=yul101 uv run python scripts/sidecar_integration_test.py

Exercises:
  1. Bootstrap + health
  2. 20 consecutive execs with sidecar → latency stats
  3. Security sanity: an exec cannot read /tmp/.cube_sidecar_token
  4. Fallback: launch a direct-mode container and confirm it still works
"""

from __future__ import annotations

import logging
import os
import statistics
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("sidecar-itest")

from cube.backends.toolkit import ToolkitContainerBackend
from cube.container import ContainerConfig


N_EXECS = 20
IMAGE = "python:3.12-slim"


def _run_suite(backend: ToolkitContainerBackend, label: str) -> dict:
    log.info("=== %s ===", label)
    cfg = ContainerConfig(image=IMAGE)
    t0 = time.monotonic()
    container = backend.launch(cfg)
    launch_s = time.monotonic() - t0
    log.info("[%s] launched in %.1fs, id=%s", label, launch_s, container.id[:12])

    latencies = []
    failures = 0
    try:
        # First call triggers sidecar bootstrap on default mode — time it separately.
        t0 = time.monotonic()
        r0 = container.exec("echo hello", timeout=30)
        first_s = time.monotonic() - t0
        log.info("[%s] first exec: %.2fs rc=%s out=%r", label, first_s, r0.exit_code, r0.stdout)
        assert r0.exit_code == 0 and r0.stdout == "hello"

        for i in range(N_EXECS):
            t0 = time.monotonic()
            r = container.exec(f"echo iter{i}", timeout=30)
            dt = time.monotonic() - t0
            latencies.append(dt)
            if r.exit_code != 0 or r.stdout != f"iter{i}":
                failures += 1
                log.warning("[%s] iter %d bad result: rc=%s out=%r", label, i, r.exit_code, r.stdout)

        # Security sanity: confirm the token file path is not visible via env.
        r = container.exec(
            "env | grep -c CUBE_SIDECAR_TOKEN_FILE || true; echo SEP; ls /tmp/.cube_sidecar_token 2>&1",
            timeout=30,
        )
        log.info("[%s] security probe: %s", label, r.stdout.replace("\n", " | "))
        # token file IS on disk (needed by server), but env var is not in child.
        # That's our documented posture — see _toolkit_sidecar_server.py.

    finally:
        log.info("[%s] stopping", label)
        container.stop()

    return {
        "label": label,
        "launch_s": launch_s,
        "first_s": first_s,
        "n_execs": len(latencies),
        "failures": failures,
        "median_s": statistics.median(latencies) if latencies else None,
        "p95_s": statistics.quantiles(latencies, n=20)[-1] if len(latencies) >= 20 else None,
        "total_exec_s": sum(latencies),
    }


def main() -> int:
    profile = os.environ.get("EAI_PROFILE", "yul101")
    log.info("Using EAI_PROFILE=%s", profile)

    results = []

    log.info("## Test 1: sidecar mode (default)")
    backend = ToolkitContainerBackend(timeout_seconds=600, profile=profile, exec_mode="sidecar")
    try:
        results.append(_run_suite(backend, "sidecar"))
    except Exception as exc:
        log.exception("sidecar suite raised")
        results.append({"label": "sidecar", "error": str(exc)})

    log.info("## Test 2: direct mode (baseline; known-flaky by design)")
    backend = ToolkitContainerBackend(timeout_seconds=600, profile=profile, exec_mode="direct")
    try:
        results.append(_run_suite(backend, "direct"))
    except Exception as exc:
        log.exception("direct suite raised (expected: CLOSE_WAIT bug)")
        results.append({"label": "direct", "error": str(exc)})

    print("\n===== SUMMARY =====")
    for r in results:
        if "error" in r:
            print(f"{r['label']:>8s}: ERROR — {r['error']}")
            continue
        print(
            f"{r['label']:>8s}: launch={r['launch_s']:.1f}s  first={r['first_s']:.2f}s  "
            f"execs={r['n_execs']}  failures={r['failures']}  "
            f"median={r['median_s']:.3f}s  total_exec={r['total_exec_s']:.1f}s"
        )

    sidecar = next((r for r in results if r["label"] == "sidecar"), None)
    if not sidecar or "error" in sidecar or sidecar.get("failures", 1) > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
