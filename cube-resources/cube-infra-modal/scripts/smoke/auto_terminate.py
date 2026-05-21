#!/usr/bin/env python3
"""Smoke: verify Modal auto-terminates sandboxes at ``timeout`` wall-clock.

Background
==========

cube-infra-modal's ``cleanup_stale()`` is a no-op on the explicit assumption
that Modal Sandboxes auto-terminate at ``timeout_seconds`` (default 1 hour
in ``ModalInfraConfig``). Unlike Daytona, Modal's termination is wall-clock
based — *not* idle-based. The sandbox dies after ``timeout`` seconds
regardless of whether anything is talking to it.

The cube codebase passes ``timeout=self.timeout_seconds`` into
``modal.Sandbox.create()`` on every launch (see
``cube-resources/cube-infra-modal/src/cube_infra_modal/modal_infra.py``)
but never tests that Modal actually honours it. This smoke fills that gap.

What it does
============

1. Create a Modal Sandbox with ``timeout=60`` (1 minute wall-clock cap).
2. Sit idle (the sandbox runs Modal's default entry point with no SDK calls
   from us — same conditions as if the cube process exited).
3. Poll ``sandbox.poll()`` every 10s. ``poll()`` returns ``None`` while the
   sandbox is running, and the exit code once it has terminated.
4. Assert that the sandbox transitions to terminated within
   ``timeout + buffer`` seconds.
5. Explicit ``sandbox.terminate()`` in ``finally`` belt-and-suspenders.

How to read the output
======================

- **SMOKE OK** — Modal terminated the sandbox within
  ``timeout + buffer_seconds``. Confirms ``cleanup_stale``'s no-op contract.
- **SMOKE FAIL** — sandbox still running past the budget. Either Modal did
  not honour the API, or the contract is different than documented. Worth
  filing a follow-up before relying on the no-op for production cleanup.
- **SMOKE SKIP** — Modal SDK not installed OR ``MODAL_TOKEN_ID``/
  ``MODAL_TOKEN_SECRET`` not set.

Cost: ~$0.01-0.02 per run (~90s of a small sandbox).

Run from cube-standard repo root:

    uv run cube-resources/cube-infra-modal/scripts/smoke/auto_terminate.py
    # Or with custom timing:
    uv run cube-resources/cube-infra-modal/scripts/smoke/auto_terminate.py \\
        --timeout-seconds 60 --buffer-seconds 30
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import time
from typing import Annotated

import typer

NAME = "modal_auto_terminate"

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


def main(
    timeout_seconds: Annotated[
        int,
        typer.Option(help="Modal sandbox ``timeout`` to set (wall-clock kill)."),
    ] = 60,
    buffer_seconds: Annotated[
        int,
        typer.Option(help="Extra wait after ``timeout`` for Modal to actually act."),
    ] = 30,
    poll_interval_s: Annotated[int, typer.Option(help="How often to poll sandbox state.")] = 10,
    app_name: Annotated[str, typer.Option(help="Modal app to attach the test sandbox to.")] = "cube-smoke",
) -> int:
    """Verify Modal terminates sandboxes at the configured ``timeout``."""
    if importlib.util.find_spec("modal") is None:
        return banner("SKIP", "modal SDK not installed")
    if not (os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET")):
        # Modal also honours ~/.modal.toml, but env vars are the clearest
        # signal that auth is configured for this process.
        if not os.path.exists(os.path.expanduser("~/.modal.toml")):
            return banner("SKIP", "MODAL_TOKEN_ID/SECRET unset and ~/.modal.toml missing")

    # Import lazily so SKIP paths don't pay the import cost.
    import modal

    log.info("=" * 70)
    log.info("Modal auto-terminate smoke")
    log.info(
        "  timeout=%ds, buffer=%ds, app=%s, poll=%ds",
        timeout_seconds,
        buffer_seconds,
        app_name,
        poll_interval_s,
    )
    log.info("=" * 70)

    try:
        app = modal.App.lookup(app_name, create_if_missing=True)
    except Exception as exc:
        return banner("FAIL", f"could not look up Modal app {app_name!r}: {exc}")
    log.info("Modal app: %s", app.name)

    log.info("Creating sandbox with timeout=%ds...", timeout_seconds)
    t_create = time.time()
    try:
        sandbox = modal.Sandbox.create(
            app=app,
            image=modal.Image.debian_slim(),
            timeout=timeout_seconds,
            cpu=1,
            memory=512,
        )
    except Exception as exc:
        return banner("FAIL", f"sandbox creation failed: {exc}")
    create_elapsed = time.time() - t_create
    sandbox_id = sandbox.object_id
    log.info("Created sandbox=%s in %.0fs", sandbox_id, create_elapsed)

    try:
        # From here on we INTENTIONALLY do not touch the sandbox — no exec,
        # no fs ops. Only sandbox.poll() which is a status read on the
        # control plane and does not affect the wall-clock timer.
        t_idle_start = time.time()
        budget_s = timeout_seconds + buffer_seconds
        deadline = t_idle_start + budget_s
        log.info(
            "Waiting up to %ds (= %ds timeout + %ds buffer) for sandbox to terminate...",
            budget_s,
            timeout_seconds,
            buffer_seconds,
        )

        exit_code: int | None = None
        while time.time() < deadline:
            try:
                exit_code = sandbox.poll()  # None if still running, int if done
            except Exception as exc:
                log.warning("poll() raised, retrying: %s", exc)
                time.sleep(poll_interval_s)
                continue

            elapsed = time.time() - t_idle_start
            state = "RUNNING" if exit_code is None else f"TERMINATED(exit={exit_code})"
            log.info("  [t+%3.0fs] sandbox state=%s", elapsed, state)
            if exit_code is not None:
                break
            time.sleep(poll_interval_s)

        if exit_code is None:
            return banner(
                "FAIL",
                f"sandbox still running at {budget_s}s deadline. "
                f"Modal did not auto-terminate within timeout={timeout_seconds}s + "
                f"{buffer_seconds}s buffer. Investigate whether the platform honours the API.",
            )

        time_to_terminate = time.time() - t_idle_start
        return banner(
            "OK",
            f"sandbox terminated (exit={exit_code}) after {time_to_terminate:.0f}s (budget {budget_s}s)",
        )
    finally:
        # Belt-and-suspenders: if the smoke exits before Modal finishes (e.g.,
        # FAIL or KeyboardInterrupt), force-terminate so we don't leak.
        log.info("Cleanup: explicit terminate of %s (no-op if already gone)", sandbox_id)
        try:
            sandbox.terminate()
        except Exception as exc:
            log.info("  terminate noted: %s", exc)


if __name__ == "__main__":
    sys.exit(typer.run(main) or 0)
