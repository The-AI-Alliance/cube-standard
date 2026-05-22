#!/usr/bin/env python3
"""Smoke: infra capability gate (`container:root` + `on_incompatible`) — beyond CI.

The unit tests exercise the gate against a *stub* infra. This smoke validates the two
things only a real infra + a real container can show:

Part A — handshake against the REAL `LocalInfraConfig.capabilities()` (no daemon needed;
  capabilities() keys off the docker *binary*): when docker is present, local advertises
  `container:root` and `can_serve` a root-requiring `ContainerConfig`, and refuses one
  that needs a capability it lacks.

Part B — `BenchmarkConfig.make()` gate end-to-end with real local infra capabilities
  (install() no-op'd — orthogonal system-dep step we are not testing):
    * a non-root infra (real local minus `container:root`, mirroring EAI Toolkit) →
      `on_incompatible="raise"` aborts with `IncompatibleInfraError`; `"skip"` drops only
      the root task; `"force"` keeps everything.
    * a root-capable infra serves both tasks.

Part C — capability TRUTH (docker daemon required, else SKIP): launch a container on local
  docker and assert `id -u` == 0. CI asserts the *advertised* token; only a real launch
  proves the container actually runs as uid 0.

SMOKE OK / FAIL / SKIP. Run from the cube-standard repo root::

    uv run scripts/smoke/capability_gate_smoke.py
"""

from __future__ import annotations

import logging
import subprocess
import sys

from cube.benchmark import Benchmark, BenchmarkConfig, BenchmarkMetadata
from cube.infra_local import LocalInfraConfig
from cube.resource import ContainerConfig, IncompatibleInfraError
from cube.task import TaskConfig, TaskMetadata

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("cap-gate-smoke")

NAME = "capability_gate_smoke"
IMAGE = "python:3.12-slim"


def banner(status: str, reason: str = "") -> int:
    line = f"SMOKE {status}: {NAME}"
    if reason:
        line += f" — {reason}"
    print(line)
    return {"OK": 0, "FAIL": 1, "SKIP": 2}[status]


# ── A real LocalInfraConfig, with install() no-op'd (not what this smoke tests) ──


class _RootLocal(LocalInfraConfig):
    def install(self) -> None:  # avoid the orthogonal `brew/apt install qemu` side-effect
        pass


class _NonRootLocal(_RootLocal):
    """Mirrors a non-root infra (e.g. EAI Toolkit): real local minus `container:root`."""

    def capabilities(self) -> set[str]:
        return super().capabilities() - {"container:root"}


# ── Minimal benchmark: one root-requiring task, one plain ──────────────────────


class _GateTaskConfig(TaskConfig):
    def make(self, runtime_context=None):
        raise NotImplementedError("the gate smoke never spawns tasks")


class _GateBench(Benchmark):
    def _setup(self) -> None:
        pass

    def close(self) -> None:
        pass


class _GateBenchConfig(BenchmarkConfig):
    benchmark_metadata = BenchmarkMetadata(name="cap-gate-smoke", version="1", description="x")
    task_metadata = {
        "root": TaskMetadata(id="root", container_config=ContainerConfig(image=IMAGE, requires={"container:root"})),
        "plain": TaskMetadata(id="plain", container_config=ContainerConfig(image=IMAGE)),
    }
    task_config_class = _GateTaskConfig
    benchmark_class = _GateBench


def _raises_incompatible(infra) -> bool:
    try:
        _GateBenchConfig().make(infra)
        return False
    except IncompatibleInfraError:
        return True


def check_handshake() -> None:
    """Part A — real LocalInfraConfig capability handshake."""
    caps = LocalInfraConfig().capabilities()
    has_docker = "docker" in caps
    root_cc = ContainerConfig(image=IMAGE, requires={"container:root"})
    if has_docker:
        assert "container:root" in caps, f"local advertises docker but not container:root: {sorted(caps)}"
        assert LocalInfraConfig().can_serve(root_cc) is True
    assert _NonRootLocal().can_serve(root_cc) is False, "a non-root infra must NOT serve a root task"
    assert _NonRootLocal().can_serve(ContainerConfig(image=IMAGE)) is has_docker
    log.info("  [A] capability handshake OK (docker=%s, container:root=%s)", has_docker, "container:root" in caps)


def check_gate() -> None:
    """Part B — make() gate with real local capabilities."""
    has_docker = "docker" in LocalInfraConfig().capabilities()

    # raise: a non-root infra refuses the root task (holds regardless of docker —
    # container:root is absent either way).
    assert _raises_incompatible(_NonRootLocal(on_incompatible="raise")), "raise mode must abort on the root task"
    log.info("  [B] raise -> IncompatibleInfraError on non-root infra OK")

    if not has_docker:
        log.info("  [B] no docker binary — skipping skip/force/positive cases (plain task needs docker)")
        return

    # skip: drop only the incompatible (root) task.
    bench = _GateBenchConfig().make(_NonRootLocal(on_incompatible="skip"))
    try:
        assert set(bench.config.tasks()) == {"plain"}, set(bench.config.tasks())
    finally:
        bench.close()
    log.info("  [B] skip -> {plain} kept, {root} dropped OK")

    # force: keep everything.
    bench = _GateBenchConfig().make(_NonRootLocal(on_incompatible="force"))
    try:
        assert set(bench.config.tasks()) == {"root", "plain"}
    finally:
        bench.close()
    log.info("  [B] force -> all tasks kept OK")

    # root-capable infra serves both under the default raise policy.
    bench = _GateBenchConfig().make(_RootLocal(on_incompatible="raise"))
    try:
        assert set(bench.config.tasks()) == {"root", "plain"}
    finally:
        bench.close()
    log.info("  [B] root-capable infra serves the root task OK")


def _docker_daemon_ok() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode == 0
    except Exception:  # noqa: BLE001
        return False


def check_runtime_uid() -> tuple[bool, str]:
    """Part C — the container actually runs as uid 0. Returns (ran, detail)."""
    if "docker" not in LocalInfraConfig().capabilities():
        return (False, "no docker binary — runtime uid check skipped")
    if not _docker_daemon_ok():
        return (False, "docker daemon unreachable — runtime uid check skipped")
    from cube.task_infra import launch_task_container

    log.info("  [C] launching %s to verify uid 0 ...", IMAGE)
    handle, container = launch_task_container(
        {"infra": LocalInfraConfig(enable_kvm=False)},
        name="cap-gate-smoke",
        image=IMAGE,
        ram_gb=1.0,
        cpu_cores=1.0,
    )
    try:
        res = container.exec("id -u")
        assert res.exit_code == 0, f"exit_code={res.exit_code} stderr={res.stderr!r}"
        assert res.stdout.strip() == "0", f"container:root claimed but `id -u` returned {res.stdout!r}"
    finally:
        handle.close()
    log.info("  [C] container runs as uid 0 — container:root claim is truthful")
    return (True, "launched, id -u == 0, closed")


def main() -> int:
    try:
        log.info("Part A — capability handshake")
        check_handshake()
        log.info("Part B — make() gate")
        check_gate()
        log.info("Part C — runtime uid")
        uid_ran, uid_detail = check_runtime_uid()
        log.info("  [C] %s", uid_detail)
    except AssertionError as exc:
        return banner("FAIL", f"check failed: {exc}")
    except Exception as exc:  # noqa: BLE001
        log.exception("unexpected error")
        return banner("FAIL", f"{type(exc).__name__}: {exc}")

    return banner("OK", f"handshake + make() gate OK; runtime uid {'verified' if uid_ran else 'skipped'}")


if __name__ == "__main__":
    sys.exit(main())
