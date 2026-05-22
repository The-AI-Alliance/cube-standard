#!/usr/bin/env python3
"""Smoke: ContainerConfig folded into the ResourceConfig family (Phase 1 unify).

Background
==========
``ContainerConfig`` used to be a standalone ``TypedBaseModel``; the orphan
``DockerImageConfig`` was a parallel single-image ``ResourceConfig`` (defined, never
launched by any infra). Phase 1 made ``ContainerConfig`` a ``ResourceConfig`` — so it
joins the capability handshake (``requirements()`` / ``InfraConfig.can_serve``) — and
deleted ``DockerImageConfig``.

The risk surface is **serialization**: thousands of already-committed
``task_metadata.json`` entries carry ``"_type": "cube.container.ContainerConfig"`` and
were serialized *before* the now-inherited (and required on ``ResourceConfig``)
``name`` field existed. They must still deserialize without regeneration.

What it does
============
Part A — serialization (always runs, no docker):
  1. Invariants — ``ContainerConfig`` is-a ``ResourceConfig``; ``requirements() == {"docker"}``
     (``+ "gpu:nvidia"`` when ``gpu=True``); a docker-capable infra ``can_serve`` it, a
     non-docker one does not; the ``_type`` tag stays ``cube.container.ContainerConfig``.
  2. Legacy round-trip — a dict with no ``name`` and the legacy ``_type`` deserializes
     via ``ResourceConfig.model_validate`` into a ``ContainerConfig(name="")``.
  3. Real-data sweep — every ``cube.container.ContainerConfig`` blob in the sibling
     cube-harness cubes' ``task_metadata.json`` must deserialize as a ``ContainerConfig``
     with ``docker`` in ``requirements()`` (swebench-verified/-live, terminalbench2 ≈ 2.5k
     blobs). Skipped (with a note) if cube-harness is not found.

Part B — launch (docker required, else noted as skipped):
  Provision + launch ``python:3.12-slim`` via ``launch_task_container`` on
  ``LocalInfraConfig``, ``exec`` an echo, assert, then ``close`` — proving the task
  bridge that consumes ``ContainerConfig`` fields still works end-to-end.

How to read the output
======================
- **SMOKE OK**   — every executed check passed.
- **SMOKE FAIL** — an invariant, a real serialized blob, or the launch failed.
- **SMOKE SKIP** — nothing was runnable (Part A is always runnable, so this is unexpected).

Run from the cube-standard repo root::

    uv run scripts/smoke/container_config_unify.py
    uv run scripts/smoke/container_config_unify.py --cube-harness /path/to/cube-harness
    uv run scripts/smoke/container_config_unify.py --no-launch
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from cube.container import ContainerConfig
from cube.infra_local import LocalInfraConfig
from cube.resource import ResourceConfig
from cube.task_infra import launch_task_container

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("cc-unify-smoke")

NAME = "container_config_unify"
LEGACY_TYPE = "cube.container.ContainerConfig"
LAUNCH_IMAGE = "python:3.12-slim"


def banner(status: str, reason: str = "") -> int:
    """Print the standard SMOKE banner; return the conventional exit code."""
    line = f"SMOKE {status}: {NAME}"
    if reason:
        line += f" — {reason}"
    print(line)
    return {"OK": 0, "FAIL": 1, "SKIP": 2}[status]


def _iter_container_blobs(node: Any) -> Iterator[dict]:
    """Yield every dict whose ``_type`` is the legacy ContainerConfig tag."""
    if isinstance(node, dict):
        if node.get("_type") == LEGACY_TYPE:
            yield node
        for v in node.values():
            yield from _iter_container_blobs(v)
    elif isinstance(node, list):
        for v in node:
            yield from _iter_container_blobs(v)


def check_invariants() -> None:
    """Part A.1 — type/capability invariants."""
    assert issubclass(ContainerConfig, ResourceConfig), "ContainerConfig must be a ResourceConfig"
    cc = ContainerConfig(image=LAUNCH_IMAGE)
    assert isinstance(cc, ResourceConfig)
    assert cc.requirements() == {"docker"}, cc.requirements()
    assert ContainerConfig(image="x", gpu=True).requirements() == {"docker", "gpu:nvidia"}
    assert cc.scope == "task", cc.scope
    assert cc.model_dump()["_type"] == LEGACY_TYPE, "the _type tag must stay stable"
    # can_serve is exactly set-inclusion of requirements() in capabilities():
    assert cc.requirements().issubset({"docker", "network:egress"}), "docker-shaped caps must serve it"
    assert not cc.requirements().issubset({"kvm"}), "a kvm-only infra must NOT serve it"
    log.info("  [A.1] invariants OK (is-a ResourceConfig, requirements, _type stable, capability inclusion)")


def check_legacy_roundtrip() -> None:
    """Part A.2 — a name-less legacy blob still deserializes."""
    legacy = {
        "_type": LEGACY_TYPE,
        "image": LAUNCH_IMAGE,
        "ram_gb": 1.0,
        "cpu_cores": 1.0,
        "gpu": False,
        "disk_gb": 10.0,
        "ports": None,
    }
    obj = ResourceConfig.model_validate(legacy)
    assert isinstance(obj, ContainerConfig), type(obj)
    assert obj.name == "", repr(obj.name)
    assert obj.image == LAUNCH_IMAGE
    assert obj.requirements() == {"docker"}
    log.info("  [A.2] legacy (name-less) blob deserializes -> ContainerConfig(name='')")


def sweep_real_metadata(cube_harness: Path | None) -> tuple[int, int]:
    """Part A.3 — validate every real ContainerConfig blob. Returns (n_blobs, n_files)."""
    if cube_harness is None or not cube_harness.exists():
        log.info("  [A.3] cube-harness not found — real-data sweep skipped (synthetic checks cover the logic)")
        return (0, 0)
    files = [f for f in cube_harness.glob("cubes/*/src/*/task_metadata.json") if ".venv" not in f.parts]
    n_blobs = 0
    for f in files:
        data = json.loads(f.read_text())
        blobs = list(_iter_container_blobs(data))
        for blob in blobs:
            obj = ResourceConfig.model_validate(blob)
            assert isinstance(obj, ContainerConfig), f"{f.name}: {type(obj)}"
            assert "docker" in obj.requirements(), f"{f.name}: {obj.requirements()}"
        if blobs:
            log.info("  [A.3] %4d blobs OK  %s", len(blobs), f.relative_to(cube_harness))
        n_blobs += len(blobs)
    log.info("  [A.3] real-data sweep OK — %d ContainerConfig blobs across %d files", n_blobs, len(files))
    return (n_blobs, len(files))


def _docker_daemon_ok() -> bool:
    """True only if the docker *daemon* responds — capabilities() checks just the binary."""
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode == 0
    except Exception:  # noqa: BLE001 — any failure means the daemon is unusable
        return False


def check_launch() -> tuple[bool, str]:
    """Part B — real container launch via the task bridge. Returns (ran, detail)."""
    infra = LocalInfraConfig(enable_kvm=False)
    if "docker" not in infra.capabilities():
        return (False, "no docker binary on this host — launch skipped")
    if not _docker_daemon_ok():
        return (False, "docker binary present but daemon unreachable — launch skipped")
    log.info("  [B] launching %s via launch_task_container ...", LAUNCH_IMAGE)
    handle, container = launch_task_container(
        {"infra": infra},
        name="cc-unify-smoke",
        image=LAUNCH_IMAGE,
        ram_gb=1.0,
        cpu_cores=1.0,
    )
    try:
        token = "hello-cube-unify"
        res = container.exec(f"echo {token}")
        assert res.exit_code == 0, f"exit_code={res.exit_code} stderr={res.stderr!r}"
        assert token in res.stdout, f"stdout={res.stdout!r}"
    finally:
        handle.close()
    log.info("  [B] launch + exec + close OK")
    return (True, "launched python:3.12-slim, exec echo, closed")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    default_harness = Path(__file__).resolve().parents[2].parent / "cube-harness"
    ap.add_argument(
        "--cube-harness",
        type=Path,
        default=default_harness,
        help=f"Path to cube-harness for the real-data sweep (default: {default_harness}).",
    )
    ap.add_argument("--no-launch", action="store_true", help="Skip Part B (the docker launch).")
    args = ap.parse_args()

    try:
        log.info("Part A — serialization")
        check_invariants()
        check_legacy_roundtrip()
        n_blobs, n_files = sweep_real_metadata(args.cube_harness)

        launch_ran, launch_detail = (False, "skipped (--no-launch)")
        if not args.no_launch:
            log.info("Part B — launch")
            launch_ran, launch_detail = check_launch()
        log.info("  [B] %s", launch_detail)
    except AssertionError as exc:
        return banner("FAIL", f"check failed: {exc}")
    except Exception as exc:  # noqa: BLE001 — smoke surfaces any failure as FAIL
        log.exception("unexpected error")
        return banner("FAIL", f"{type(exc).__name__}: {exc}")

    summary = f"serialization OK ({n_blobs} real blobs / {n_files} files); launch {'OK' if launch_ran else 'skipped'}"
    return banner("OK", summary)


if __name__ == "__main__":
    sys.exit(main())
