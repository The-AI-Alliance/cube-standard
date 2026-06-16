"""Counter benchmark - demonstrates auto-loading metadata from JSON files.

This example is structurally identical to toy_benchmark/counter.py, except that
CounterBenchmark does not define benchmark_metadata or task_metadata inline.
They are automatically loaded from benchmark_metadata.json and task_metadata.json
sitting next to this file.

Note on TaskMetadata subclasses + JSON auto-loading
---------------------------------------------------
``BenchmarkConfig.task_metadata_from_json`` validates each entry as the base
``TaskMetadata`` class — which dispatches to the right subclass when the JSON
entry includes a ``"_type"`` field (TypedBaseModel's polymorphic discriminator).
``task_metadata.json`` here uses ``"_type": "counter.CounterTaskMetadata"``
so each entry is loaded as a typed ``CounterTaskMetadata`` with ``target`` populated.
Using the real module name (``counter``) rather than ``__main__`` ensures the JSON
loads correctly whether the file is run directly or imported as a module.
"""

from collections.abc import Generator
from typing import Any, ClassVar, Dict, Literal, Tuple

from cube.benchmark import Benchmark, BenchmarkConfig, RuntimeContext
from cube.container import Container
from cube.core import Action, ActionSchema, Observation
from cube.task import Task, TaskConfig, TaskMetadata
from cube.tool import Tool, ToolConfig, tool_action


class ConfigurableCounterTool(Tool):
    def __init__(self, increment_by: int = 1, enable_decrement: bool = False):
        self.counter = 0
        self.history: list[str] = []
        self.increment_by = increment_by
        self.enable_decrement = enable_decrement

    def reset(self) -> None:
        self.counter = 0
        self.history = []

    @tool_action
    def increment(self) -> str:
        """Increment the counter"""
        self.counter += self.increment_by
        self.history.append("increment")
        return f"Counter is now {self.counter}"

    @tool_action
    def get_value(self) -> str:
        """Get the current counter value"""
        return f"Counter value is: {self.counter}"

    @tool_action
    def decrement(self) -> str:
        """Decrement the counter by 1"""
        self.counter -= 1
        self.history.append("decrement")
        return f"Counter is now {self.counter}"

    @property
    def action_set(self) -> list[ActionSchema]:
        return [a for a in super().action_set if a.name != "decrement" or self.enable_decrement]


class CounterToolConfig(ToolConfig):
    increment_by: int = 1
    enable_decrement: bool = False

    def make(self, container: Container | None = None) -> ConfigurableCounterTool:
        return ConfigurableCounterTool(increment_by=self.increment_by, enable_decrement=self.enable_decrement)


class CounterTaskMetadata(TaskMetadata):
    """Per-task metadata with a typed ``target``."""

    target: int
    difficulty: Literal["easy", "medium", "hard"] = "easy"


class ReachTargetTask(Task):
    metadata: CounterTaskMetadata  # type: ignore[assignment]

    @property
    def target(self) -> int:
        return self.metadata.target

    def reset(self) -> Tuple[Observation, Dict[str, Any]]:
        self.tool.reset()
        obs = Observation.from_text(f"Counter starts at 0. Use 'increment' to reach {self.target}.")
        return obs, {"target": self.target}

    def evaluate(self, obs: Observation | None = None) -> Tuple[float, Dict[str, Any]]:
        assert isinstance(self.tool, ConfigurableCounterTool)
        solved = self.tool.counter == self.target
        return (1.0 if solved else 0.0), {"solved": solved, "value": self.tool.counter}

    def finished(self, obs: Observation | None = None) -> bool:
        assert isinstance(self.tool, ConfigurableCounterTool)
        return self.tool.counter == self.target


class CounterTaskConfig(TaskConfig):
    def make(
        self,
        runtime_context: RuntimeContext | None = None,
    ) -> ReachTargetTask:
        tool_cfg = self.tool_config or CounterToolConfig()
        return ReachTargetTask(
            metadata=self.metadata,
            tool_config=tool_cfg,
            runtime_context=runtime_context,
        )


class CounterBenchmark(Benchmark):
    def _setup(self) -> None:
        pass

    def close(self) -> None:
        pass


# benchmark_metadata and task_metadata are intentionally omitted:
# they are auto-loaded from benchmark_metadata.json and task_metadata.json
# in the same directory as this file.
class CounterBenchmarkConfig(BenchmarkConfig):
    task_config_class: ClassVar[type[TaskConfig]] = CounterTaskConfig
    benchmark_class: ClassVar[type[Benchmark]] = CounterBenchmark

    _TASK_TOOL_CONFIGS: ClassVar[dict[str, CounterToolConfig]] = {
        "count-to-3-with-decrement": CounterToolConfig(enable_decrement=True),
        "count-by-2": CounterToolConfig(increment_by=2),
    }

    def get_task_configs(self) -> Generator[CounterTaskConfig, None, None]:
        for task_id, tm in self.tasks().items():
            yield CounterTaskConfig(
                metadata=tm,
                tool_config=self._TASK_TOOL_CONFIGS.get(task_id),
            )


if __name__ == "__main__":
    config = CounterBenchmarkConfig()

    print(f"benchmark_metadata.name   = {config.benchmark_metadata.name}")
    print(f"benchmark_metadata.tags   = {config.benchmark_metadata.tags}")
    print(f"tasks loaded              = {list(config.task_metadata.keys())}")

    task_configs = list(config.get_task_configs())
    assert len(task_configs) == 3

    with config.make() as bench:
        task = bench.spawn(task_configs[0])
        obs, _ = task.reset()
        print(f"\nFirst task reset obs: {obs.contents[0].data}")

        for _ in range(3):
            env_out = task.step(Action(name="increment", arguments={}))
        assert env_out.done  # type: ignore
        print(f"After 3 increments: done={env_out.done}, reward={env_out.reward}")  # type: ignore

    print("\nAll checks passed.")
