"""Convenience helpers for launching per-task (L3) Docker containers via InfraConfig.

Cubes that provision one Docker container per task (SWE-bench, MLE-bench, terminal-bench,
CTF, ...) all share the same launch shape: build a ``DockerServiceConfig(scope="task")``
with one image, delegate to ``runtime_context["infra"].launch()``, and consume the
handle's ``.container`` in the tool layer.

``launch_task_container()`` wraps that pattern so each cube's ``Task.model_post_init``
is two lines: call the helper, bind the returned ``(handle, container)`` to the Task.
"""

from __future__ import annotations

import uuid

from cube.container import Container
from cube.resource import DockerServiceConfig, ResourceHandle
from cube.task import RuntimeContext

__all__ = ["launch_task_container", "build_docker_run_script"]


# Gate+apply security capability tokens → the ``docker run`` flag each one applies.
# Deterministic order keeps the generated script stable. See resource/spec.md.
_SECURITY_FLAGS: dict[str, str] = {
    "container:privileged": "--privileged",
    "container:cgroupns-host": "--cgroupns=host",
}


def build_docker_run_script(
    container_name: str,
    image: str,
    ram_gb: float,
    cpu_cores: float,
    requires: set[str] | None = None,
) -> str:
    """Return a bash snippet that `docker run`s ``image`` as a detached ``sleep infinity`` container.

    Used as the ``launch_script`` on ``DockerServiceConfig(scope="task")``.  Infras that
    don't shell-out to bash (Daytona, Toolkit) ignore the script and create the container
    directly from ``docker_images[0]``.

    Tags the container with ``--label cube.launch=$CUBE_LAUNCH_ID`` — ``LocalInfraConfig``
    injects ``CUBE_LAUNCH_ID`` into the shell env before running the script, so the
    label filter in ``_launch_docker_service`` can identify just this container under
    concurrent launches (pytest-xdist, Ray workers, etc.).  ``:-unlabeled`` lets the
    script still run when invoked outside the infra (manual debugging).

    ``requires`` carries the resource's capability tokens (``ResourceConfig.requirements()``);
    each gate+apply security token present (``container:privileged``,
    ``container:cgroupns-host``) is translated to its ``docker run`` flag. Such tokens are
    capability-gated by ``can_serve`` before launch, so an infra only ever applies a flag it
    advertised.
    """
    security = "".join(f"{flag} " for token, flag in _SECURITY_FLAGS.items() if token in (requires or set()))
    return (
        f"docker run -d "
        f"--name {container_name} "
        f'--label "cube.launch=${{CUBE_LAUNCH_ID:-unlabeled}}" '
        f"--memory={int(ram_gb * 1024)}m "
        f"--cpus={cpu_cores} "
        f"{security}"
        f"{image} "
        f"sleep infinity"
    )


def launch_task_container(
    runtime_context: RuntimeContext,
    *,
    name: str,
    image: str,
    ram_gb: float = 4.0,
    cpu_cores: float = 2.0,
    requires: set[str] | None = None,
) -> tuple[ResourceHandle, Container]:
    """Provision + launch a single-container L3 resource via ``runtime_context["infra"]``.

    Args:
        runtime_context: Benchmark's runtime_context dict.  MUST contain an ``"infra"``
            key bound to an ``InfraConfig`` (convention set by ``Benchmark._setup()``).
        name: Stable resource identifier (e.g. ``f"swebench-verified-{task_id}"``).
            Used for ProvisionStore keys and as the docker container-name prefix.
        image: Container image reference (e.g. ``"swebench/sweb.eval.x86_64.django-11099:latest"``).
        ram_gb / cpu_cores: Resource requirements hinted to the infra.
        requires: Capability tokens (``container_config.requirements()``). Gate+apply
            security tokens (``container:privileged``, ``container:cgroupns-host``) are
            applied as ``docker run`` flags; all are already capability-gated at
            ``BenchmarkConfig.make()`` time.

    Returns:
        ``(handle, container)`` — the ResourceHandle (for lifecycle teardown via
        ``handle.close()``) and the ``Container`` the tool layer drives.

    Raises:
        KeyError: ``runtime_context`` has no ``"infra"`` key.
        ResourceNotReadyError: Image hasn't been provisioned and the infra's
            ``provision_status()`` still reports ``"needs_provisioning"`` after
            an idempotent ``provision()`` call.
    """
    infra = runtime_context["infra"]
    container_name = f"{name}-{uuid.uuid4().hex[:8]}"
    resource = DockerServiceConfig(
        name=name,
        scope="task",
        docker_images=[image],
        launch_script=build_docker_run_script(container_name, image, ram_gb, cpu_cores, requires),
        requires=set(requires or set()),
    )
    if infra.provision_status(resource) == "needs_provisioning":
        infra.provision(resource)
    handle = infra.launch(resource)
    return handle, handle.container
