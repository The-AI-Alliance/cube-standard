"""Step 3 & 4a of 4 — Task and TaskConfig.

Task owns one Tool instance and implements the gym-like episode loop:

  obs, info = task.reset()          # initialise; return first observation
  while not done:
      env_out = task.step(action)   # execute action, get next obs + reward
  reward, info = task.evaluate(obs) # final scoring
  task.close()                      # cleanup

Two abstract methods you must implement:
  reset()    — initialise the tool, return (Observation, info_dict)
  evaluate() — return (reward: float, info: dict) given current obs

Optional hooks worth knowing:
  finished()       — return True to trigger early termination (done=True)
                     without waiting for the agent to call the stop action.
  filter_actions() — restrict the action set at the Task level without
                     touching the Tool. Preferred over overriding Tool.action_set
                     because it keeps the tool stateless and reusable.

Task is a Pydantic model. Store task-specific parameters in
metadata.extra_info (a plain dict). Do NOT add new Pydantic fields for
per-task data — extra_info is the intended extension point.

TaskConfig is the serializable description of a single task instance. Like
ToolConfig it must be a Pydantic model (for cross-process transport) and
must implement make() to produce a Task.

The typical pattern:
  1. Look up TaskMetadata from the Benchmark class by task_id.
  2. Build the ToolConfig: prefer an explicit override on this TaskConfig,
     fall back to per-task defaults stored in metadata.extra_info.
  3. Construct and return the Task.

task_id and seed are inherited from the base TaskConfig class; you rarely
need to add new fields here.
"""

from typing import Any

from cube.benchmark import RuntimeContext
from cube.containers import ContainerBackend
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

    def evaluate(self, obs: Observation) -> tuple[float, dict[str, Any]]:

        value = self.tool._env.counter

        if value == self.target:
            return 1.0, {"solved": True, "value": value}

        progress = min(1.0, value / self.target) if self.target > 0 else 0.0
        return progress * 0.5, {"solved": False, "value": value, "target": self.target}

    def finished(self, obs: Observation) -> bool:
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
        # Import here to avoid circular import (benchmark imports task)
        from counter_cube.benchmark import CounterBenchmark
        #TODO: find a proper solution for this circular import issue.
        
        task_metadata: TaskMetadata = CounterBenchmark.task_metadata[self.task_id]
        tool_cfg = self.tool_config or CounterToolConfig(**task_metadata.extra_info.get("tool_config", {}))
        return ReachTargetTask(
            metadata=task_metadata,
            tool_config=tool_cfg,
            runtime_context=runtime_context,
            container_backend=container_backend,
        )
