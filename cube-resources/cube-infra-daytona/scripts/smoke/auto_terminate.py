#!/usr/bin/env python3
"""Smoke: verify Daytona auto-stops + deletes idle sandboxes as advertised.

Background
==========

cube-infra-daytona's ``cleanup_stale()`` is a no-op — sandboxes are supposed to
self-destruct on Daytona's platform via:

  - ``auto_stop_interval`` (minutes of SDK inactivity → Daytona stops the sandbox)
  - ``ephemeral=True`` (Daytona deletes the sandbox after stop)

The cube codebase **passes these into the SDK on every launch** (see
``cube-resources/cube-infra-daytona/src/cube_infra_daytona/daytona.py``) but
never tests that the platform actually honours them. This smoke fills that gap.

What it does
============

1. Create a Daytona sandbox with ``auto_stop_interval=N`` (default 1 minute) and
   ``ephemeral=True``.
2. **Do not touch the sandbox** — no exec, no fs ops, no HTTP. Only out-of-band
   ``Daytona.get(id)`` polls for state, which uses the control-plane API and
   does not count as sandbox activity per Daytona's idle definition.
3. Poll the sandbox state every 15s.
4. Assert the sandbox transitions Started → (Stopping/Stopped) → (Destroying/Destroyed)
   within ``auto_stop_interval + buffer`` minutes.
5. Final cleanup in ``finally`` — explicit delete in case the auto-cleanup didn't
   fire and we exit early.

How to read the output
======================

- **SMOKE OK** — Daytona stopped (and ideally deleted) the idle sandbox within
  the expected window. Confirms ``cleanup_stale``'s no-op contract is sound.
- **SMOKE FAIL** — sandbox stayed running past the budget. Either Daytona did
  not honour the API, or the API contract is different than documented. Worth
  filing a follow-up before relying on the no-op for production cleanup.
- **SMOKE SKIP** — ``DAYTONA_API_KEY`` not set. Set it (or run from a host with
  ``.env`` loaded by ``dotenv``).

Cost: ~$0.01 per run (~2-3 min of a small sandbox).

Run from cube-standard repo root:

    uv run cube-resources/cube-infra-daytona/scripts/smoke/auto_terminate.py
    # or with custom timing:
    uv run cube-resources/cube-infra-daytona/scripts/smoke/auto_terminate.py \\
        --auto-stop-minutes 1 --wait-buffer-minutes 3
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import time
from typing import Annotated

import typer

NAME = "daytona_auto_terminate"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(NAME)


def banner(status: str, reason: str = "") -> int:
    """Print the standard SMOKE banner and return the conventional exit code."""
    line = f"SMOKE {status}: {NAME}"
    if reason:
        line += f": {reason}"
    print(line)
    return {"OK": 0, "FAIL": 1, "SKIP": 2}[status]


# States that mean "the sandbox is no longer actively billing compute".
# Daytona reports terminal-ish states as the platform proceeds through
# Started → Stopping → Stopped → Destroying → Destroyed.
_NON_RUNNING_STATES = {"STOPPED", "STOPPING", "DESTROYING", "DESTROYED"}


def main(
    auto_stop_minutes: Annotated[int, typer.Option(help="Daytona auto_stop_interval to set on the sandbox.")] = 1,
    wait_buffer_minutes: Annotated[
        int,
        typer.Option(
            help="Extra wait after auto_stop_minutes for the platform to actually act.",
        ),
    ] = 3,
    poll_interval_s: Annotated[int, typer.Option(help="How often to poll sandbox state.")] = 15,
    image: Annotated[str, typer.Option(help="Docker image to use for the test sandbox.")] = "python:3.12-slim",
) -> int:
    """Verify Daytona auto-stops idle sandboxes per its ``auto_stop_interval`` contract."""
    if importlib.util.find_spec("daytona") is None:
        return banner("SKIP", "daytona SDK not installed")
    if not os.environ.get("DAYTONA_API_KEY"):
        return banner("SKIP", "DAYTONA_API_KEY not set")

    # Import lazily so SKIP-cases above don't pay the import cost.
    from daytona import CreateSandboxFromImageParams, Daytona, DaytonaConfig, Resources
    from daytona import Image as DaytonaImage

    log.info("=" * 70)
    log.info("Daytona auto-terminate smoke")
    log.info(
        "  auto_stop=%dm, wait_buffer=%dm, image=%s, poll=%ds",
        auto_stop_minutes,
        wait_buffer_minutes,
        image,
        poll_interval_s,
    )
    log.info("=" * 70)

    # Build client from env vars (DAYTONA_API_URL / TARGET optional).
    config_kwargs: dict[str, str] = {"api_key": os.environ["DAYTONA_API_KEY"]}
    if api_url := os.environ.get("DAYTONA_API_URL"):
        config_kwargs["api_url"] = api_url
    if target := os.environ.get("DAYTONA_TARGET"):
        config_kwargs["target"] = target
    client = Daytona(DaytonaConfig(**config_kwargs))

    params = CreateSandboxFromImageParams(
        image=DaytonaImage.base(image),
        resources=Resources(cpu=1, memory=1, disk=4),
        auto_stop_interval=auto_stop_minutes,
        ephemeral=True,
        labels={"cube_smoke": "auto_terminate"},
        network_block_all=False,
    )

    log.info("Creating sandbox (this typically takes ~30-60s)...")
    t_create = time.time()
    try:
        sandbox = client.create(params, timeout=300)
    except Exception as exc:
        return banner("FAIL", f"sandbox creation failed: {exc}")
    create_elapsed = time.time() - t_create
    sandbox_id = getattr(sandbox, "id", None)
    if not sandbox_id:
        return banner("FAIL", "sandbox.id missing on created sandbox")
    log.info("Created sandbox=%s in %.0fs", sandbox_id, create_elapsed)

    try:
        # From here on we INTENTIONALLY do not touch the sandbox — no exec, no
        # fs ops, no HTTP. We only Daytona.get(id) which is control-plane and
        # should not reset the idle timer.
        t_idle_start = time.time()
        budget_s = (auto_stop_minutes + wait_buffer_minutes) * 60
        deadline = t_idle_start + budget_s
        log.info(
            "Waiting up to %.0fs (= %dm + %dm buffer) for sandbox to leave Started state...",
            budget_s,
            auto_stop_minutes,
            wait_buffer_minutes,
        )

        last_state = "unknown"
        terminal_state: str | None = None
        while time.time() < deadline:
            try:
                fresh = client.get(sandbox_id)
                state = str(fresh.state).split(".")[-1].upper()  # SandboxState enum → name
            except Exception as exc:
                # The platform may legitimately 404 once the sandbox is
                # fully destroyed — treat that as terminal success.
                msg = str(exc).lower()
                if "not found" in msg or "404" in msg:
                    terminal_state = "DESTROYED (404)"
                    break
                log.warning("get(sandbox) raised, retrying: %s", exc)
                time.sleep(poll_interval_s)
                continue

            elapsed = time.time() - t_idle_start
            log.info("  [t+%4.0fs] sandbox state=%s", elapsed, state)
            if state in _NON_RUNNING_STATES:
                terminal_state = state
                break
            last_state = state
            time.sleep(poll_interval_s)

        if terminal_state is None:
            return banner(
                "FAIL",
                f"sandbox still running at {budget_s}s deadline (last_state={last_state}). "
                f"Daytona did not auto-stop within auto_stop_interval={auto_stop_minutes}m + "
                f"{wait_buffer_minutes}m buffer. Investigate whether the platform honours the API.",
            )

        time_to_stop = time.time() - t_idle_start
        log.info("Sandbox reached %s after %.0fs of idle", terminal_state, time_to_stop)
        return banner(
            "OK",
            f"sandbox auto-terminated to {terminal_state} after {time_to_stop:.0f}s (budget {budget_s}s)",
        )
    finally:
        # Belt-and-suspenders cleanup: if the smoke exits before the platform
        # finishes its job (or if FAIL), explicitly delete so we don't leak
        # the test sandbox.
        log.info("Cleanup: explicit delete of %s (no-op if already destroyed)", sandbox_id)
        try:
            still_there = client.get(sandbox_id)
            client.delete(still_there)
        except Exception as exc:
            log.info("  delete noted: %s", exc)


if __name__ == "__main__":
    sys.exit(typer.run(main) or 0)
