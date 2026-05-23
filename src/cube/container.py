"""Container — a ``ResourceHandle`` that adds exec / port-forwarding.

A ``Container`` IS a ``ResourceHandle``.  ``InfraConfig.launch()`` returns one
directly — no wrapper indirection.  Subclasses carry the handle bookkeeping
(run_id / resource / infra / created_at / expires_at) alongside their driver-
specific state.

``ContainerConfig`` (the serializable description of *what* container a task needs,
``TaskMetadata.container_config``) is a ``ResourceConfig`` and now lives in
``cube.resource`` alongside the other resource configs; it is re-exported from this
module for backward compatibility.  The old ``ContainerBackend`` factory has been
removed; provisioning is now done exclusively through ``InfraConfig``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict

from cube.resource import ResourceHandle

logger = logging.getLogger(__name__)


@dataclass
class ExecResult:
    """Result of running a command inside a container."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    duration_seconds: float = 0.0


@dataclass
class ContainerStatus:
    """Snapshot of container health / resource usage."""

    running: bool = False
    healthy: bool = False
    resource_usage: dict[str, float] = field(default_factory=dict)
    backend_info: dict[str, Any] = field(default_factory=dict)


class ContainerError(Exception):
    """Base exception for all container-related errors."""


class ContainerLaunchError(ContainerError):
    """Raised when a container fails to start."""


class HealthCheckError(ContainerError):
    """Raised when a container's health check fails."""


class ContainerExecError(ContainerError):
    """Raised when command execution inside a container fails."""


class Container(ResourceHandle, ABC):
    """Live handle to a running container.

    Inherits ``ResourceHandle`` bookkeeping (run_id, resource, infra, created_at,
    expires_at, endpoint, endpoints) and adds the container-specific capability
    surface: exec / port-forward / status / id.

    ``ResourceHandle.close()`` is satisfied by delegating to ``stop()`` so callers
    can treat the handle uniformly via the ``with`` statement or ``close()``.
    """

    @abstractmethod
    def exec(
        self,
        command: str,
        timeout: int | None = None,
        workdir: str | None = None,
        env: Dict[str, str] | None = None,
    ) -> ExecResult:
        """Execute *command* inside the container."""

    @abstractmethod
    def forward_port(self, container_port: int) -> int:
        """Return the reachable port for *container_port* on this backend."""

    @abstractmethod
    def get_url(self, container_port: int) -> str:
        """Return a URL that reaches *container_port*."""

    @abstractmethod
    def stop(self, timeout: int = 10) -> None:
        """Stop (and optionally remove) the container. Idempotent."""

    @abstractmethod
    def get_status(self) -> ContainerStatus:
        """Return current container status."""

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique, backend-specific container identifier."""

    def close(self) -> None:
        """Delegates to ``stop()`` — satisfies ``ResourceHandle.close()``."""
        self.stop()

    @property
    def container(self) -> "Container":
        """Return self — Container IS ResourceHandle, no wrapper.

        Kept for uniformity with legacy handle types (e.g.
        ``LocalDockerServiceHandle.container``) that wrap multiple containers
        and need an indirection property.
        """
        return self


def relocate_if_readonly(
    container: Container,
    working_dir: str,
    new_wd: str,
    *,
    extra_setup: str | None = None,
    force: bool = False,
) -> str:
    """Copy *working_dir* to *new_wd* if it isn't writable by the runtime user.

    Returns the effective working directory (either the original if writable, or
    *new_wd* after the copy).  Cubes that need additional setup after the copy
    (e.g. ``git config safe.directory``) can pass the commands as *extra_setup*
    — they are appended to the ``cp -a`` invocation with ``&&``.

    The default probe (``test -w working_dir``) only checks the **top** dir. A
    writable top dir can still hold root-owned, non-writable *subdirectories*
    (some upstream Docker images bake them in) that a non-root runtime user
    can neither write nor reparent in place — the caller knows this from its own
    deeper writability check and passes ``force=True`` to relocate unconditionally
    to a fully runtime-user-owned copy (``cp -a`` creates everything owned by the
    runtime user). Tests run from *new_wd* import the relocated copy because
    ``python -m pytest`` puts the cwd first on ``sys.path``.

    Typical usage in a cube's ``_build_tool()``::

        new_wd = relocate_if_readonly(
            self._container, self.tool_config.working_dir, "/tmp/testbed",
            extra_setup="git config --global --add safe.directory /tmp/testbed",
        )
        self._tool = self.tool_config.model_copy(update={"working_dir": new_wd}).make(
            container=self._container
        )
    """
    # auto-fix(205)↓ force= skips the top-dir probe so a caller can relocate even
    # when `test -w {working_dir}` passes — it misses root-owned non-writable subdirs.
    if not force:
        probe = container.exec(f"test -w {working_dir} && echo W || echo R", timeout=30)
        if "W" in probe.stdout:
            return working_dir
    # /auto-fix(205)
    logger.info("%s not writable by runtime user — copying to %s", working_dir, new_wd)
    cmd = f"cp -a {working_dir} {new_wd}"
    if extra_setup:
        cmd += f" && {extra_setup}"
    # auto-fix(176)↓
    result = container.exec(cmd, timeout=300)
    if result.exit_code != 0:
        raise ContainerExecError(
            f"relocate_if_readonly: '{cmd}' failed (exit {result.exit_code}); "
            f"working dir {new_wd!r} was not created. stderr: {result.stderr.strip()}"
        )
    # /auto-fix(176)
    return new_wd


def port_from_url(url: str) -> int:
    """Extract the effective port from a URL (443 for https, 80 for http)."""
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.port is not None:
        return parsed.port
    if parsed.scheme == "https":
        return 443
    if parsed.scheme == "http":
        return 80
    raise ContainerError(f"Could not determine port from URL: {url}")


# ── Task container requirements ──────────────────────────────────────────────


# ``ContainerConfig`` now lives in ``cube.resource`` (it is a ``ResourceConfig``).
# Re-exported here so that ``_type: cube.container.ContainerConfig`` strings — serialized
# before the move — still resolve on deserialize (importlib + getattr on this module).
from cube.resource import ContainerConfig  # noqa: E402,F401


# === auto-fix notes ===  (spec: openspec/specs/auto-fix/spec.md)
# auto-fix-note(176) {class=L1 issue=176 hash=PENDING ctx=docker/tbench2:prove-plus-comm/cube-standard@0e91ae1}
#   symptoms:  tbench2 task prove-plus-comm -- image has a read-only/absent
#              /app; `cp -a` failed but the exit code was discarded and the
#              never-created new_wd returned, so callers chdir'd into a
#              phantom dir. Trigger = image shape; infra-agnostic.
#   invariant: relocate_if_readonly returns a dir that exists, or raises
#              ContainerExecError -- never a path that was never created.
#   why:       check result.exit_code; raise the module's domain exception
#              (ContainerExecError), consistent with its error hierarchy.
#   tested:    tests/test_container.py relocate cases.
#   hash=PENDING: stamped by scripts/auto_fix_lint.py (Tier-1) on first run.
# auto-fix-note(205) {class=L1 issue=205 hash=PENDING ctx=toolkit/uid-13011/swebench-verified/cube-standard@a9d98a7}
#   symptoms:  non-root runtime (EAI toolkit uid 13011) — `test -w /testbed` passes
#              but a root-owned subdir (psf/requests' /testbed/requests/) is
#              non-writable, so relocate never fired and patches hit Permission denied.
#   invariant: relocate_if_readonly yields a FULLY writable working dir; force= lets
#              a caller that detected nested non-writable paths relocate regardless.
#   why:       force= is additive (default False = prior probe behavior); only the
#              caller knows its tree must be fully writable (it patches every file).
#   tested:    cube-harness #443 — psf/requests gold patch 0/6 -> 6/6 on toolkit.
#   hash=PENDING: stamped by scripts/auto_fix_lint.py (Tier-1) on first run.
