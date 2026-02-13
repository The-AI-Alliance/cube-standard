"""EAI Toolkit / HPC backend — stub (not yet implemented)."""

from __future__ import annotations

from cube.container import Container, ContainerBackend, ContainerSpec


class ToolkitContainerBackend(ContainerBackend):
    """Run containers via EAI Toolkit on HPC / SLURM — stub, not yet implemented."""

    def launch(self, spec: ContainerSpec) -> Container:
        raise NotImplementedError(
            "ToolkitContainerBackend is not yet implemented. "
            "See design/docker_wrapper.md for the intended design."
        )
