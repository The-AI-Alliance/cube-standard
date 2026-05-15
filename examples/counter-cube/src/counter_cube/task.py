"""Task and TaskConfig for counter-cube.

Task owns a Tool and implements the episode loop (reset / step / evaluate / close).
Must implement: reset() → (Observation, info), evaluate(obs) → (reward, info).
Optional: finished() for early termination, filter_actions() to restrict actions.

For per-task state, prefer typed Pydantic fields over stringly-typed dicts.
``CounterTaskMetadata`` carries semantic task descriptors (target, difficulty).
Per-task tool variation is handled in ``BenchmarkConfig.get_task_configs()``,
not stored on ``TaskMetadata`` — metadata describes *what* a task is, not *how*
to run it.
"""

from typing import Any, Literal

from cube.benchmark import RuntimeContext
from cube.core import Observation
from cube.task import Task, TaskConfig, TaskMetadata
from counter_cube.tool import CounterToolConfig


class CounterTaskMetadata(TaskMetadata):
    """Per-task metadata for counter tasks — typed replacement for ``extra_info``."""

    target: int
    """Counter value the agent must reach to solve the task."""

    difficulty: Literal["easy", "medium", "hard"] = "easy"
    """Task difficulty label, surfaced via ``subset_from_glob`` / registry filters."""


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
    ) -> ReachTargetTask:
        """Build the task.

        Per-task tool variation is set by ``CounterBenchmarkConfig.get_task_configs()``
        on ``self.tool_config``; fall back to the default config if none was set.
        """
        tool_cfg = self.tool_config or CounterToolConfig()
        return ReachTargetTask(
            metadata=self.metadata,
            tool_config=tool_cfg,
            runtime_context=runtime_context,
        )
