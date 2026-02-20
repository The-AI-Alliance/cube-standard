"""Minimal counter benchmark - toy bench

This example demonstrates:
1. Basic task/tool implementation
2. ToolConfig flexibility with alternative implementations
3. High-level and low-level task/tool interaction patterns
"""

from typing import Any, ClassVar, Dict, Tuple

from cube.benchmark import Benchmark, BenchmarkMetadata, RuntimeContext
from cube.containers import Container, ContainerBackend
from cube.core import Action, ActionSchema, Observation
from cube.task import Task, TaskConfig, TaskMetadata
from cube.tool import Tool, ToolConfig, tool_action


# Tool Implementation
class ConfigurableCounterTool(Tool):
    """Configurable counter tool.

    Supports:
    - Configurable increment amount (increment_by)
    - Optional decrement action
    - Optional reset action
    """

    def __init__(
        self,
        increment_by: int = 1,
        enable_decrement: bool = False,
        enable_reset: bool = False,
    ):
        """Initialize counter tool with configurable features."""
        self.counter = 0
        self.history: list[str] = []
        self.increment_by = increment_by
        self.enable_decrement = enable_decrement
        self.enable_reset = enable_reset

    def reset(self) -> None:
        """Reset tool to initial state."""
        self.counter = 0
        self.history = []

    @tool_action
    def increment(self) -> str:
        """Increment the counter"""
        self.counter += self.increment_by
        self.history.append("increment")
        if self.increment_by == 1:
            return f"Counter is now {self.counter}"
        else:
            return f"Counter is now {self.counter} (incremented by {self.increment_by})"

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

    @tool_action
    def reset_counter(self) -> str:
        """Reset counter to 0"""
        self.counter = 0
        self.history.append("reset")
        return "Counter reset to 0"

    @property
    def action_set(self) -> list[ActionSchema]:
        """Return only enabled actions based on configuration."""
        all_actions = super().action_set

        # Filter based on configuration
        enabled_actions = []
        for action in all_actions:
            if action.name == "decrement" and not self.enable_decrement:
                continue
            if action.name == "reset_counter" and not self.enable_reset:
                continue
            enabled_actions.append(action)

        return enabled_actions


# ToolConfig Implementation
class CounterToolConfig(ToolConfig):
    """Tool configuration for counter benchmark."""

    increment_by: int = 1
    enable_decrement: bool = False
    enable_reset: bool = False

    def make(self, container: Container | None = None) -> ConfigurableCounterTool:
        """Create tool instance with configured features."""
        return ConfigurableCounterTool(
            increment_by=self.increment_by,
            enable_decrement=self.enable_decrement,
            enable_reset=self.enable_reset,
        )


# Task Implementation
class ReachTargetTask(Task):
    """Task: Increment counter to reach target value."""

    @property
    def target(self) -> int:
        return self.metadata.extra_info["target"]

    def reset(self) -> Tuple[Observation, Dict[str, Any]]:
        """Reset the task to its initial state."""
        # Reset tool to initial state
        self.tool.reset()
        obs = Observation.from_text(f"Counter starts at 0. Use 'increment' action to reach {self.target}.")
        return obs, {"task_type": "reach_target", "target": self.target}

    def evaluate(self, obs: Observation) -> Tuple[float, Dict[str, Any]]:
        """Validate if counter reached target."""
        assert isinstance(self.tool, ConfigurableCounterTool)
        counter_value = self.tool.counter
        steps_taken = len(self.tool.history)

        if counter_value == self.target:
            return 1.0, {
                "solved": True,
                "value": counter_value,
                "steps": steps_taken,
            }

        # Partial reward based on progress
        progress = min(1.0, counter_value / self.target) if self.target > 0 else 0.0
        return progress * 0.5, {
            "solved": False,
            "value": counter_value,
            "target": self.target,
            "steps": steps_taken,
        }

    def finished(self, obs: Observation) -> bool:
        """Check if task is complete."""
        assert isinstance(self.tool, ConfigurableCounterTool)
        return self.tool.counter == self.target


# TaskConfig Implementation
class CounterTaskConfig(TaskConfig):
    """Configuration for counter tasks."""

    def make(
        self,
        runtime_context: RuntimeContext | None = None,
        container_backend: ContainerBackend | None = None,
    ) -> ReachTargetTask:
        """Create task instance from config.

        Looks up task metadata from CounterBenchmark.task_metadata_dict by task_id.
        Builds CounterToolConfig from extra_info["tool_config"] if present, otherwise uses defaults.
        An explicit tool_config on this TaskConfig always takes precedence.
        """
        task_metadata = CounterBenchmark.task_metadata_dict[self.task_id]
        tool_cfg = self.tool_config or CounterToolConfig(**task_metadata.extra_info.get("tool_config", {}))
        return ReachTargetTask(
            metadata=task_metadata,
            tool_config=tool_cfg,
            runtime_context=runtime_context,
            container_backend=container_backend,
        )


# Benchmark Implementation
class CounterBenchmark(Benchmark):
    """Minimal benchmark with counter tasks."""

    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="toy-counter",
        version="1.0.0",
        description="Simplest possible benchmark - count to target value",
        num_tasks=3,
        tags=["toy", "counter", "minimal"],
    )
    task_metadata_dict: ClassVar[dict[str, TaskMetadata]] = {
        "count-to-3": TaskMetadata(
            id="count-to-3",
            abstract_description="Increment counter to reach value 3",
            recommended_max_steps=5,
            extra_info={"target": 3, "difficulty": "easy"},
        ),
        "count-to-3-with-decrement": TaskMetadata(
            id="count-to-3-with-decrement",
            abstract_description="Increment counter to reach value 3, with decrement available",
            recommended_max_steps=7,
            extra_info={"target": 3, "difficulty": "easy", "tool_config": {"enable_decrement": True}},
        ),
        "count-by-2": TaskMetadata(
            id="count-by-2",
            abstract_description="Counter with custom increment amount",
            recommended_max_steps=4,
            extra_info={"target": 4, "difficulty": "easy", "tool_config": {"increment_by": 2}},
        ),
    }
    task_config_class: ClassVar[type[TaskConfig]] = CounterTaskConfig

    def _setup(self) -> None:
        """No shared infrastructure needed for this simple benchmark."""
        pass

    def close(self) -> None:
        """No resources to clean up for this simple benchmark."""
        pass


# Test Function
def test_counter_benchmark():
    """Comprehensive test demonstrating task-tool interaction and ToolConfig flexibility."""
    print("Starting counter benchmark tests...")
    print("=" * 60)

    benchmark = CounterBenchmark()
    benchmark.setup()

    task_configs = {c.task_id: c for c in benchmark.get_task_configs()}

    # === Test 1: Single action execution via .step (with default config) ===
    print("\n" + "=" * 60)
    print("Test 1: Single action execution via .step")
    print("=" * 60)

    task1 = task_configs["count-to-3"].make()
    obs, _ = task1.reset()

    print(f"Initial observation: {obs.contents[0].data}")
    print(f"Available actions: {[a.name for a in task1.action_set]}")

    # Verify default config doesn't have decrement/reset
    action_names = [a.name for a in task1.action_set]
    assert "decrement" not in action_names, "Default config should not have 'decrement'"
    assert "reset_counter" not in action_names, "Default config should not have 'reset_counter'"

    # Agent loop: obs -> action -> step -> obs
    env_output = None
    for step_num in range(1, 4):
        action = Action(name="increment", arguments={})
        env_output = task1.step(action)
        print(f"Step {step_num}: {env_output.obs.contents[0].data} (reward={env_output.reward})")

    assert isinstance(task1.tool, ConfigurableCounterTool)
    assert task1.tool.counter == 3, f"Expected counter to be 3, got {task1.tool.counter}"
    assert env_output is not None and env_output.done, "Task should be done"
    print("✓ Single action execution works!")

    # === Test 2: Multiple actions in one step ===
    print("\n" + "=" * 60)
    print("Test 2: Multiple action execution via .step")
    print("=" * 60)

    task2 = task_configs["count-to-3"].make()
    task2.reset()

    # Execute multiple actions at once
    actions = [
        Action(name="increment", arguments={}),
        Action(name="get_value", arguments={}),
        Action(name="increment", arguments={}),
    ]
    print(f"Executing {len(actions)} actions: {[a.name for a in actions]}")

    env_output = task2.step(actions)
    print("Observations:")
    for i, content in enumerate(env_output.obs.contents):
        print(f"  {i + 1}. {content.data}")

    assert isinstance(task2.tool, ConfigurableCounterTool)
    assert task2.tool.counter == 2, f"Expected counter to be 2, got {task2.tool.counter}"
    print("✓ Multiple action execution works!")

    # === Test 3: Lower level API (.tool.execute_action) ===
    print("\n" + "=" * 60)
    print("Test 3: Lower level API (.tool.execute_action)")
    print("=" * 60)

    task3 = task_configs["count-to-3"].make()
    task3.reset()

    # Lower level: directly call tool.execute_action()
    action = Action(name="increment", arguments={})
    result = task3.tool.execute_action(action)
    assert isinstance(result, Observation), "Expected Observation from tool action"
    assert isinstance(task3.tool, ConfigurableCounterTool)
    assert task3.tool.counter == 1, f"Expected counter to be 1, got {task3.tool.counter}"
    print(f"Direct tool execution: {result.contents[0].data}")
    print("Note: Bypasses task.step() and doesn't trigger evaluation")
    print("✓ Lower level API works!")

    # === Test 4: ToolConfig flexibility - enable decrement ===
    print("\n" + "=" * 60)
    print("Test 4: ToolConfig flexibility - enable decrement")
    print("=" * 60)

    task4 = task_configs["count-to-3-with-decrement"].make()
    task4.reset()

    # Verify decrement is now available
    action_names = [a.name for a in task4.action_set]
    print(f"Available actions: {action_names}")
    assert "decrement" in action_names, "Expected 'decrement' action with enable_decrement=True"

    # Test increment and decrement
    env_output = task4.step(Action(name="increment", arguments={}))
    print(f"Increment: {env_output.obs.contents[0].data}")
    assert isinstance(task4.tool, ConfigurableCounterTool)
    assert task4.tool.counter == 1

    env_output = task4.step(Action(name="decrement", arguments={}))
    print(f"Decrement: {env_output.obs.contents[0].data}")
    assert task4.tool.counter == 0, "Decrement should reduce counter"
    print("✓ ToolConfig can add/remove actions!")

    # === Test 5: ToolConfig flexibility - custom increment amount ===
    print("\n" + "=" * 60)
    print("Test 5: ToolConfig flexibility - custom increment amount")
    print("=" * 60)

    task5 = task_configs["count-by-2"].make()
    task5.reset()

    # Test increment by 2
    env_output = task5.step(Action(name="increment", arguments={}))
    print(f"Increment: {env_output.obs.contents[0].data}")
    assert isinstance(task5.tool, ConfigurableCounterTool)
    assert task5.tool.counter == 2, f"Expected counter to be 2, got {task5.tool.counter}"

    env_output = task5.step(Action(name="increment", arguments={}))
    print(f"Increment: {env_output.obs.contents[0].data}")
    assert task5.tool.counter == 4, f"Expected counter to be 4, got {task5.tool.counter}"
    assert env_output.done, "Task should be done"
    print("✓ ToolConfig can change action behavior!")

    # === Test 6: Task isolation ===
    print("\n" + "=" * 60)
    print("Test 6: Task isolation")
    print("=" * 60)

    assert task1.tool.counter == 3, "Task1 should still have counter=3"
    assert task2.tool.counter == 2, "Task2 should have counter=2"
    assert task3.tool.counter == 1, "Task3 should have counter=1"
    assert task4.tool.counter == 0, "Task4 should have counter=0"
    assert task5.tool.counter == 4, "Task5 should have counter=4"
    print("✓ Task isolation verified - each task maintains independent state")

    # Cleanup
    task1.close()
    task2.close()
    task3.close()
    task4.close()
    task5.close()
    benchmark.close()

    print("\n" + "=" * 60)
    print("✓✓ All tests passed!")
    print("=" * 60)
    print("\nKey features demonstrated:")
    print("- Single action execution via .step()")
    print("- Multiple action execution via .step()")
    print("- Lower level API via .tool.execute_action()")
    print("- ToolConfig adds/removes actions (decrement, reset)")
    print("- ToolConfig changes action behavior (increment by 2)")
    print("- Task isolation (independent tool instances)")


# Main
if __name__ == "__main__":
    test_counter_benchmark()
