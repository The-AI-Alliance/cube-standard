"""Task and TaskConfig for counter-cube.

Task owns a Tool and implements the episode loop (reset / step / evaluate / close).
Must implement: reset() → (Observation, info), evaluate(obs) → (reward, info).
Optional: finished() for early termination, filter_actions() to restrict actions.

For per-task state, prefer typed Pydantic fields over stringly-typed dicts:
``CounterTaskMetadata`` carries the per-task ``target`` and an optional
``tool_overrides`` slot for tasks that need a non-default tool config.
TaskConfig is serialisable; implement make() to produce a Task.
"""

from typing import Any, Literal

from cube.benchmark import RuntimeContext
from cube.container import ContainerBackend
from cube.core import Observation
from cube.task import Task, TaskConfig, TaskMetadata
from counter_cube.tool import CounterToolConfig


class CounterTaskMetadata(TaskMetadata):
    """Per-task metadata for counter tasks — typed replacement for ``extra_info``."""

    target: int
    """Counter value the agent must reach to solve the task."""

    difficulty: Literal["easy", "medium", "hard"] = "easy"
    """Task difficulty label, surfaced via ``subset_from_glob`` / registry filters."""

    tool_overrides: CounterToolConfig | None = None
    """Optional default ``CounterToolConfig`` for this task — used by
    ``CounterTaskConfig.make`` when the TaskConfig does not carry an
    explicit ``tool_config``."""


class ReachTargetTask(Task):
    """Task: increment the counter until it equals ``target``.

    Read the typed ``target`` field directly from the (subclassed) metadata.
    """

    metadata: CounterTaskMetadata  # type: ignore[assignment]

    @property
    def target(self) -> int:
        return self.metadata.target

    def reset(self) -> tuple[Observation, dict[str, Any]]:
        """Reset tool state and return the opening observation."""
        self.tool.reset()
        obs = Observation.from_text(f"Counter starts at 0. Use 'increment' action to reach {self.target}.")
        return obs, {"task_type": "reach_target", "target": self.target}

    def evaluate(self, obs: Observation | None = None) -> tuple[float, dict[str, Any]]:
        value = self.tool._env.counter

        if value == self.target:
            return 1.0, {"solved": True, "value": value}

        progress = min(1.0, value / self.target) if self.target > 0 else 0.0
        return progress * 0.5, {"solved": False, "value": value, "target": self.target}

    def finished(self, obs: Observation | None = None) -> bool:
        return self.tool._env.counter == self.target


class CounterTaskConfig(TaskConfig):
    """Serializable configuration that produces a ReachTargetTask.

    Self-contained: ``self.metadata`` carries a ``CounterTaskMetadata`` instance,
    stamped onto the config by ``CounterBenchmarkConfig.get_task_configs()`` on
    the driver.
    """

    def make(
        self,
        runtime_context: RuntimeContext | None = None,
        container_backend: ContainerBackend | None = None,
    ) -> ReachTargetTask:
        """Build the task.

        tool_config precedence (highest to lowest):
          1. explicit ``tool_config`` set on this TaskConfig instance
          2. ``self.metadata.tool_overrides`` (per-task default declared on metadata)
          3. ``CounterToolConfig`` defaults
        """
        tool_cfg = self.tool_config or self.metadata.tool_overrides or CounterToolConfig()
        return ReachTargetTask(
            metadata=self.metadata,
            tool_config=tool_cfg,
            runtime_context=runtime_context,
            container_backend=container_backend,
        )
