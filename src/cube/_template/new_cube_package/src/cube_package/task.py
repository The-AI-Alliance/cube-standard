"""Task and TaskConfig for cube_package.

Task owns a Tool instance and implements the episode loop:
  reset()    → (Observation, info dict)   called once before each episode
  evaluate() → (reward: float, info dict) called after every step
  finished() → bool                       optional early-termination check

Store per-task parameters in metadata.extra_info — don't add new Pydantic
fields to Task unless they are shared across all tasks in the benchmark.

TaskConfig is the serialisable handle that produces a Task via make().
"""

from typing import Any

from cube.benchmark import RuntimeContext
from cube.containers import ContainerBackend
from cube.core import Observation
from cube.task import Task, TaskConfig, TaskMetadata
from cube_package.tool import CubeToolConfig


class CubeTask(Task):
    """One episode: interact with CubeTool to satisfy the goal.

    The goal is defined by metadata.extra_info (set in benchmark.py).
    Read it via self.metadata.extra_info["key"].
    """

    def reset(self) -> tuple[Observation, dict[str, Any]]:
        """Initialise the tool and return the opening observation.

        Returns
        -------
        obs : Observation
            What the agent sees at the start of the episode.
        info : dict
            Arbitrary metadata (not shown to the agent).
        """
        self.tool.reset()
        # TODO: build a meaningful opening observation.
        obs = Observation.from_text("Episode started. Use available actions to complete the task.")
        return obs, {}

    def evaluate(self, obs: Observation) -> tuple[float, dict[str, Any]]:
        """Score the current state.

        Returns
        -------
        reward : float
            1.0 = solved, 0.0 = not solved.  Partial credit is allowed.
        info : dict
            Evaluation details (e.g. {"solved": True, "value": 42}).
        """
        # TODO: inspect self.tool._env to determine whether the goal is met.
        solved = False  # replace with real check
        return (1.0 if solved else 0.0), {"solved": solved}

    def finished(self, obs: Observation) -> bool:
        """Return True to end the episode early (before max steps).

        Called automatically by Task.step() after each action.
        """
        # TODO: return True when the goal is definitely achieved.
        return False


class CubeTaskConfig(TaskConfig):
    """Serialisable factory that produces a CubeTask.

    tool_config precedence (highest → lowest):
      1. Explicit tool_config set on this TaskConfig instance.
      2. Per-task tool_config in metadata.extra_info["tool_config"].
      3. CubeToolConfig defaults.
    """

    def make(
        self,
        runtime_context: RuntimeContext | None = None,
        container_backend: ContainerBackend | None = None,
    ) -> CubeTask:
        # Deferred import to avoid circular dependency (benchmark imports task).
        from cube_package.benchmark import CubeBenchmark  # noqa: PLC0415

        task_metadata: TaskMetadata = CubeBenchmark.task_metadata[self.task_id]
        tool_cfg = self.tool_config or CubeToolConfig(**task_metadata.extra_info.get("tool_config", {}))
        return CubeTask(
            metadata=task_metadata,
            tool_config=tool_cfg,
            runtime_context=runtime_context,
            container_backend=container_backend,
        )
