"""Task and TaskConfig for counter-cube.

Task owns a Tool and implements the episode loop (reset / step / evaluate / close).
Must implement: reset() → (Observation, info), evaluate(obs) → (reward, info).
Optional: finished() for early termination, filter_actions() to restrict actions.

Store per-task parameters in metadata.extra_info — don't add new Pydantic fields.
TaskConfig is serializable; implement make() to produce a Task.
"""

from typing import Any

from cube.benchmark import RuntimeContext
from cube.container import ContainerBackend
from cube.core import Observation
from cube.task import Task, TaskConfig, TaskMetadata
from counter_cube.tool import CounterToolConfig


class ReachTargetTask(Task):
    """Task: increment the counter until it equals `target`.

    Target value is read from metadata.extra_info["target"], which is set
    in CounterBenchmark.task_metadata (see benchmark.py).
    """

    @property
    def target(self) -> int:
        return self.metadata.extra_info["target"]

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
    """Serializable configuration that produces a ReachTargetTask."""

    def make(
        self,
        runtime_context: RuntimeContext | None = None,
        container_backend: ContainerBackend | None = None,
    ) -> ReachTargetTask:
        """Build the task.

        tool_config precedence (highest to lowest):
          1. explicit tool_config set on this TaskConfig instance
          2. per-task tool_config in metadata.extra_info["tool_config"]
          3. CounterToolConfig defaults
        """
        # Deferred import to avoid circular dependency (benchmark imports task).
        from counter_cube.benchmark import CounterBenchmark  # noqa: PLC0415

        task_metadata: TaskMetadata = CounterBenchmark.task_metadata[self.task_id]
        tool_cfg = self.tool_config or CounterToolConfig(**task_metadata.extra_info.get("tool_config", {}))
        return ReachTargetTask(
            metadata=task_metadata,
            tool_config=tool_cfg,
            runtime_context=runtime_context,
            container_backend=container_backend,
        )
