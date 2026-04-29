"""DEPRECATED — use ``cube_infra_toolkit.ToolkitInfraConfig`` instead.

This module is kept only to serve the legacy ``ContainerBackend`` API and
existing callers (examples, tests, older scripts).  The canonical
``ToolkitContainer`` driver (and its exec relay server) now lives in
``cube_infra_toolkit.container``.
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from typing import Literal

from cube_infra_toolkit.container import ToolkitContainer, _run_eai
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from cube.container import (
    ContainerBackend,
    ContainerConfig,
    ContainerLaunchError,
    HealthCheckError,
)

warnings.warn(
    "cube.backends.toolkit is deprecated — use "
    "cube_infra_toolkit.ToolkitInfraConfig for new code. "
    "The ToolkitContainer driver has moved to cube_infra_toolkit.container.",
    DeprecationWarning,
    stacklevel=2,
)

logger = logging.getLogger(__name__)

__all__ = ["ToolkitContainer", "ToolkitContainerBackend"]

_retry_launch = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    reraise=True,
    retry=retry_if_not_exception_type(HealthCheckError),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)


class ToolkitContainerBackend(ContainerBackend):
    """DEPRECATED.  Launches containers as EAI Toolkit jobs via the legacy API."""

    profile: str | None = None
    account: str | None = None
    interactive: bool = False
    preemptable: bool = False
    exec_mode: Literal["exec_relay", "direct"] = "exec_relay"

    def launch(self, config: ContainerConfig) -> ToolkitContainer:
        return self._launch_with_retry(config)

    @_retry_launch
    def _launch_with_retry(self, config: ContainerConfig) -> ToolkitContainer:
        cmd: list[str] = ["job", "new"]

        if self.interactive:
            cmd.append("--interactive")
        elif self.preemptable:
            cmd.append("--preemptable")
        else:
            cmd.append("--non-preemptable")

        cmd += ["--tunnel"]
        cmd += ["--format", "json", "--no-header"]
        cmd += ["-i", config.image]
        cmd += ["--cpu", str(int(config.cpu_cores))]
        cmd += ["--mem", f"{int(config.ram_gb)}"]

        if config.gpu:
            cmd += ["--gpu", "1"]

        cmd += ["--", "sleep", "infinity"]

        logger.info("Creating EAI Toolkit job with image %s …", config.image)

        result = _run_eai(
            cmd,
            profile=self.profile,
            account=self.account,
            timeout=self.timeout_seconds,
            retries=0,
        )

        if result.returncode != 0:
            raise ContainerLaunchError(f"Failed to create EAI job: {result.stderr.strip()}")

        try:
            output = json.loads(result.stdout)
            job_id = output["id"]
        except (json.JSONDecodeError, KeyError) as exc:
            first_line = result.stdout.strip().split("\n")[0].strip()
            if first_line:
                job_id = first_line
            else:
                raise ContainerLaunchError(f"Could not parse job ID from eai output: {result.stdout}") from exc

        logger.info("EAI job created: %s — waiting for RUNNING state …", job_id)

        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            status_result = _run_eai(
                ["job", "get", job_id, "--format", "json", "--no-header"],
                profile=self.profile,
                account=self.account,
                timeout=30,
            )
            try:
                info = json.loads(status_result.stdout)
                state = info.get("state", "").lower()
            except (json.JSONDecodeError, AttributeError):
                state = ""

            if state == "running":
                break
            if state in ("failed", "cancelled", "killed"):
                raise ContainerLaunchError(f"EAI job {job_id} entered terminal state: {state}")
            logger.info("EAI job [%s] state=%s, waiting …", job_id[:8], state)
            time.sleep(5)
        else:
            try:
                _run_eai(
                    ["job", "kill", job_id],
                    profile=self.profile,
                    account=self.account,
                    timeout=30,
                )
            except Exception:
                pass
            raise ContainerLaunchError(
                f"EAI job {job_id} did not reach RUNNING state within {self.timeout_seconds}s (last state: {state})"
            )

        container = ToolkitContainer(
            job_id,
            profile=self.profile,
            account=self.account,
            exec_mode=self.exec_mode,
        )
        logger.info("EAI job running: %s", job_id)

        self._run_health_check(container)
        return container
