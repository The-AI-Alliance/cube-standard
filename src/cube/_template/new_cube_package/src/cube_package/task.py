"""Task and TaskConfig for cube_package.

Task owns a Tool instance and implements the episode loop:
  reset()    → (Observation, info dict)   called once before each episode
  evaluate() → (reward: float, info dict) called after every step
  finished() → bool                       optional early-termination check

Store per-task parameters in metadata.extra_info — don't add new Pydantic
fields to Task unless they are shared across all tasks in the benchmark.

TaskConfig is the serialisable boundary that crosses process/network lines.
It carries its own metadata (stamped by BenchmarkConfig.get_task_configs()
on the driver) so workers never need to import the owning BenchmarkConfig.
Implement make() using self.metadata directly.
"""

from typing import Any

from cube.benchmark import RuntimeContext
from cube.container import ContainerBackend
from cube.core import Observation
from cube.task import Task, TaskConfig
from cube_package.tool import CubeToolConfig


class CubeTask(Task):
    """One episode: interact with CubeTool to satisfy the goal.

    The goal is defined by metadata.extra_info (set in benchmark.py).
    Read it via self.metadata.extra_info["key"].
    """

    def reset(self) -> tuple[Observation, dict[str, Any]]:
        """Initialise the tool and return the opening observation."""
        self.tool.reset()
        # TODO: build a meaningful opening observation.
        obs = Observation.from_text("Episode started. Use available actions to complete the task.")
        return obs, {}

    def evaluate(self, obs: Observation | None = None) -> tuple[float, dict[str, Any]]:
        """Score the current state.

        Returns
        -------
        reward : float
            1.0 = solved, 0.0 = not solved.  Partial credit is allowed.
        info : dict
            Evaluation details (e.g. {"solved": True, "value": 42}).
        """
        # TODO: inspect obs and/or self.tool to determine the reward.
        solved = False  # replace with real check
        return (1.0 if solved else 0.0), {"solved": solved}

    def finished(self, obs: Observation | None = None) -> bool:
        """Return True to end the episode early (before max steps)."""
        # TODO: return True when the goal is achieved.
        return False


class CubeTaskConfig(TaskConfig):
    """Serialisable factory that produces a CubeTask.

    Self-contained: ``self.metadata`` carries the TaskMetadata, stamped onto
    the config by ``CubeBenchmarkConfig.get_task_configs()`` on the driver.
    No import of the owning BenchmarkConfig needed on workers.

    tool_config precedence (highest → lowest):
      1. Explicit tool_config set on this TaskConfig instance.
      2. Per-task tool_config in self.metadata.extra_info["tool_config"].
      3. CubeToolConfig defaults.
    """

    def make(
        self,
        runtime_context: RuntimeContext | None = None,
        container_backend: ContainerBackend | None = None,
    ) -> CubeTask:
        tool_cfg = self.tool_config or CubeToolConfig(**self.metadata.extra_info.get("tool_config", {}))
        return CubeTask(
            metadata=self.metadata,
            tool_config=tool_cfg,
            runtime_context=runtime_context,
            container_backend=container_backend,
        )
