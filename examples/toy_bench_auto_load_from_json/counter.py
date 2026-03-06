"""Counter benchmark - demonstrates auto-loading metadata from JSON files.

This example is structurally identical to toy_benchmark/counter.py, except that
CounterBenchmark does not define benchmark_metadata or task_metadata inline.
They are automatically loaded from benchmark_metadata.json and task_metadata.json
sitting next to this file.
"""

from typing import Any, ClassVar, Dict, Tuple

from cube.benchmark import Benchmark, RuntimeContext
from cube.container import Container, ContainerBackend
from cube.core import Action, ActionSchema, Observation
from cube.environment import Environment, EnvironmentConfig, environment_action
from cube.task import Task, TaskConfig


class ConfigurableCounterEnvironment(Environment):
    def __init__(self, increment_by: int = 1, enable_decrement: bool = False):
        self.counter = 0
        self.history: list[str] = []
        self.increment_by = increment_by
        self.enable_decrement = enable_decrement

    def reset(self) -> None:
        self.counter = 0
        self.history = []

    @environment_action
    def increment(self) -> str:
        """Increment the counter"""
        self.counter += self.increment_by
        self.history.append("increment")
        return f"Counter is now {self.counter}"

    @environment_action
    def get_value(self) -> str:
        """Get the current counter value"""
        return f"Counter value is: {self.counter}"

    @environment_action
    def decrement(self) -> str:
        """Decrement the counter by 1"""
        self.counter -= 1
        self.history.append("decrement")
        return f"Counter is now {self.counter}"

    @property
    def action_set(self) -> list[ActionSchema]:
        return [a for a in super().action_set if a.name != "decrement" or self.enable_decrement]


class CounterEnvironmentConfig(EnvironmentConfig):
    increment_by: int = 1
    enable_decrement: bool = False

    def make(self, container: Container | None = None) -> ConfigurableCounterEnvironment:
        return ConfigurableCounterEnvironment(increment_by=self.increment_by, enable_decrement=self.enable_decrement)


class ReachTargetTask(Task):
    @property
    def target(self) -> int:
        return self.metadata.extra_info["target"]

    def reset(self) -> Tuple[Observation, Dict[str, Any]]:
        self.env.reset()
        obs = Observation.from_text(f"Counter starts at 0. Use 'increment' to reach {self.target}.")
        return obs, {"target": self.target}

    def evaluate(self, obs: Observation) -> Tuple[float, Dict[str, Any]]:
        assert isinstance(self.env, ConfigurableCounterEnvironment)
        solved = self.env.counter == self.target
        return (1.0 if solved else 0.0), {"solved": solved, "value": self.env.counter}

    def finished(self, obs: Observation) -> bool:
        assert isinstance(self.env, ConfigurableCounterEnvironment)
        return self.env.counter == self.target


class CounterTaskConfig(TaskConfig):
    def make(
        self,
        runtime_context: RuntimeContext | None = None,
        container_backend: ContainerBackend | None = None,
    ) -> ReachTargetTask:
        task_metadata = CounterBenchmark.task_metadata[self.task_id]
        env_cfg = self.env_config or CounterEnvironmentConfig(**task_metadata.extra_info.get("env_config", {}))
        return ReachTargetTask(
            metadata=task_metadata,
            env_config=env_cfg,
            runtime_context=runtime_context,
            container_backend=container_backend,
        )


# benchmark_metadata and task_metadata are intentionally omitted:
# they are auto-loaded from benchmark_metadata.json and task_metadata.json
# in the same directory as this file.
class CounterBenchmark(Benchmark):
    task_config_class: ClassVar[type[TaskConfig]] = CounterTaskConfig

    def _setup(self) -> None:
        pass

    def close(self) -> None:
        pass


if __name__ == "__main__":
    bench = CounterBenchmark()

    print(f"benchmark_metadata.name   = {bench.benchmark_metadata.name}")
    print(f"benchmark_metadata.tags   = {bench.benchmark_metadata.tags}")
    print(f"tasks loaded              = {list(bench.task_metadata.keys())}")

    task_configs = list(bench.get_task_configs())
    assert len(task_configs) == 3

    task = task_configs[0].make()
    obs, _ = task.reset()
    print(f"\nFirst task reset obs: {obs.contents[0].data}")

    for _ in range(3):
        env_out = task.step(Action(name="increment", arguments={}))
    assert env_out.done  # type: ignore
    print(f"After 3 increments: done={env_out.done}, reward={env_out.reward}")  # type: ignore

    bench.close()
    print("\nAll checks passed.")
