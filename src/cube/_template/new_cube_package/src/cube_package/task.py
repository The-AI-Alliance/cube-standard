"""Task and TaskConfig for cube_package.

Task owns a Tool instance and implements the episode loop:
  reset()    → (Observation, info dict)   called once before each episode
  evaluate() → (reward: float, info dict) called after every step
  finished() → bool                       optional early-termination check

For per-task data, prefer typed Pydantic fields over stringly-typed dicts:
  - Lightweight, eager-loaded values (always available, ship in
    task_metadata.json) → declare them on a `TaskMetadata` subclass.
  - Heavy, lazy values populated by `BenchmarkConfig.install()` (problem
    statements, patches, archives, …) → declare them on a
    `TaskExecutionInfo` subclass and surface them via `Task.execution_info`.

TaskConfig is the serialisable boundary that crosses process/network lines.
It carries its own metadata (stamped by BenchmarkConfig.get_task_configs()
on the driver) so workers never need to import the owning BenchmarkConfig.
Implement make() using self.metadata directly; populate execution_info from
self.load_task_execution_info() when your cube ships heavy data.
"""

from typing import Any

from cube.benchmark import RuntimeContext
from cube.container import ContainerBackend
from cube.core import Observation, TaskResult
from cube.task import Task, TaskConfig, TaskExecutionInfo  # noqa: F401  (TaskExecutionInfo used in commented CubeExecutionInfo example below)
from cube_package.tool import CubeToolConfig


# Optional: declare a TaskExecutionInfo subclass when your cube ships heavy
# per-task data via BenchmarkConfig.install(). Cubes with no heavy data
# can leave this commented out and keep `Task.execution_info = None`.
#
# class CubeExecutionInfo(TaskExecutionInfo):
#     """Heavy per-task data populated on the worker by CubeTaskConfig.make()."""
#
#     instruction: str
#     # patch: str = ""
#     # ...


class CubeTask(Task):
    """One episode: interact with CubeTool to satisfy the goal.

    Read static per-task config from `self.metadata.<field>` (typed) and
    heavy lazy data — when used — from `self.execution_info.<field>` (typed).
    """

    def reset(self) -> tuple[Observation, dict[str, Any]]:
        """Initialise the tool and return the opening observation."""
        self.tool.reset()
        # TODO: build a meaningful opening observation. If your cube uses
        # a CubeExecutionInfo subclass, read it via self.execution_info.<field>.
        obs = Observation.from_text("Episode started. Use available actions to complete the task.")
        return obs, {}

    def evaluate(self, obs: Observation | None = None) -> TaskResult:
        """Score the current state.

        Returns
        -------
        TaskResult with reward (1.0 = solved, 0.0 = not), checks, and info.
        """
        # TODO: inspect obs and/or self.tool to determine the reward.
        solved = False  # replace with real check
        return TaskResult(reward=1.0 if solved else 0.0, checks=[], info={"solved": solved})

    def finished(self, obs: Observation | None = None) -> bool:
        """Return True to end the episode early (before max steps)."""
        # TODO: return True when the goal is achieved.
        return False


class CubeTaskConfig(TaskConfig):
    """Serialisable factory that produces a CubeTask.

    Self-contained: ``self.metadata`` carries the (possibly subclassed)
    TaskMetadata, stamped onto the config by
    ``CubeBenchmarkConfig.get_task_configs()`` on the driver.
    """

    def verify_installed(self) -> None:
        """Optional fail-fast check before workers attempt to load heavy data.

        Default base implementation is a no-op. Override when your cube ships
        heavy data via ``BenchmarkConfig.install()`` so misconfigured workers
        error early with an actionable message instead of timing out::

            cache_dir = type(self).task_execution_cache_dir()
            if not cache_dir.exists() or not any(cache_dir.iterdir()):
                raise RuntimeError(
                    f"Run `cube install new-cube-package` first."
                )
        """

    def make(
        self,
        runtime_context: RuntimeContext | None = None,
        container_backend: ContainerBackend | None = None,
    ) -> CubeTask:
        # By convention, fail fast if this worker is misconfigured.
        self.verify_installed()
        # If your cube uses heavy per-task data, hydrate it here:
        # exec_info = CubeExecutionInfo.model_validate(
        #     self.load_task_execution_info()
        # )
        return CubeTask(
            metadata=self.metadata,
            execution_info=None,  # set to ``exec_info`` if you populate it above
            tool_config=self.tool_config or CubeToolConfig(),
            runtime_context=runtime_context,
            container_backend=container_backend,
        )
