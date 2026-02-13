"""Minimal counter benchmark - toy bench"""

from typing import Any, Dict, Tuple

from cube.benchmark import Benchmark, BenchmarkMetadata, RuntimeContext
from cube.containers import ContainerBackend
from cube.core import Action, Observation
from cube.task import Task, TaskConfig, TaskMetadata
from cube.tool import Tool, ToolConfig, tool_action


# Tool Implementation
class CounterTool(Tool):
    """Counter tool with basic counter operations."""

    def __init__(self, target: int):
        """Initialize counter tool."""
        self.counter = 0
        self.target = target
        self.history: list[str] = []

    def reset(self) -> None:
        """Reset tool to initial state."""
        self.counter = 0
        self.history = []

    @tool_action
    def increment(self) -> str:
        """Increment the counter by 1"""
        self.counter += 1
        self.history.append("increment")
        return f"Counter is now {self.counter}"

    @tool_action
    def get_value(self) -> str:
        """Get the current counter value"""
        return f"Counter value is: {self.counter}"


# ToolConfig Implementation
class CounterToolConfig(ToolConfig):
    """Tool configuration for counter benchmark."""

    target: int = 0  # Target value for the counter

    def make(self) -> Tool:
        """Create tool instance."""
        return CounterTool(target=self.target)


# Task Implementation
class ReachTargetTask(Task):
    """Task: Increment counter to reach target value."""

    tool: CounterTool  # type: ignore[assignment]  # Narrowing the type from AbstractTool

    def __init__(self, task_id: str, target: int):
        """Initialize reach target task."""
        self.metadata = TaskMetadata(
            id=task_id,
            description=f"Increment counter to reach value {target}",
            tags=["counter", "simple"],
            difficulty="easy",
            domain="counter",
            max_steps=target + 2,
        )
        self.target = target

    def setup(self) -> Tuple[Observation, Dict[str, Any]]:
        """Set up the task."""
        # Reset tool to initial state
        self.tool.reset()
        obs = Observation.from_text(f"Counter starts at 0. Use 'increment' action to reach {self.target}.")
        return obs, {"task_type": "reach_target", "target": self.target}

    def evaluate(self, obs: Observation) -> Tuple[float, Dict[str, Any]]:
        """Validate if counter reached target."""
        # Access tool state
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
        return self.tool.counter == self.target


# TaskConfig Implementation
class CounterTaskConfig(TaskConfig):
    """Configuration for counter tasks."""

    task_id: str
    target: int
    tool_config: ToolConfig

    def make(
        self, runtime_context: RuntimeContext | None = None, container_backend: ContainerBackend | None = None
    ) -> ReachTargetTask:
        """Create task instance from config."""
        # Create tool
        tool = self.tool_config.make()

        # Create task
        task = ReachTargetTask(task_id=self.task_id, target=self.target)
        task.tool = tool  # type: ignore[assignment]  # We know this is a CounterTool
        task.runtime_context = runtime_context

        return task


# Benchmark Implementation
class CounterBenchmark(Benchmark):
    """Minimal benchmark with counter tasks."""

    def __init__(self):
        """Initialize counter benchmark."""
        metadata = BenchmarkMetadata(
            name="toy-counter",
            version="1.0.0",
            description="Simplest possible benchmark - count to target value",
            num_tasks=2,
            tags=["toy", "counter", "minimal"],
        )
        super().__init__(metadata=metadata)

    def setup(self) -> RuntimeContext:
        """Set up the benchmark."""
        # No shared resources needed for this simple benchmark
        runtime_context = RuntimeContext()
        self._runtime_info = runtime_context
        return runtime_context

    def load_tasks(self, cache: bool = True) -> list[TaskConfig]:
        """Load counter tasks."""
        if len(self._task_list) > 0 and cache:
            return self._task_list

        # Create task configs
        self._task_list = [
            CounterTaskConfig(
                task_id="count-to-3",
                target=3,
                tool_config=CounterToolConfig(target=3),
            ),
            CounterTaskConfig(
                task_id="count-to-5",
                target=5,
                tool_config=CounterToolConfig(target=5),
            ),
        ]
        return self._task_list

    def close(self) -> None:
        """Clean up benchmark resources."""
        # No resources to clean up for this simple benchmark
        pass


# Test Function
def test_simple_counting():
    """Test the counter benchmark with new API - demonstrating agent-task interaction."""
    print("Starting counter benchmark test with new API...")
    print("=" * 60)

    # Create and setup benchmark
    benchmark = CounterBenchmark()
    benchmark.setup()

    # Load tasks
    task_configs: list[CounterTaskConfig] = benchmark.load_tasks()  # type: ignore (CouterTaskConfig is a subclass of TaskConfig)
    assert len(task_configs) == 2, "Expected 2 tasks"
    print(f"✓ Loaded {len(task_configs)} tasks")

    # === Test 1: Single action steps (typical agent loop) ===
    print("\n" + "=" * 60)
    print("Test 1: Single action steps (count-to-3)")
    print("=" * 60)

    task_config = task_configs[0]
    task = task_config.make()

    # Get initial observation
    obs, info = task.setup()
    print(f"Initial observation: {obs.contents[0].data}")
    print(f"Task info: {info}")
    print(f"Available actions: {[a.name for a in task.action_set]}")

    # Agent loop: obs -> action -> step -> obs
    step_num = 0
    while not obs or not task.finished(obs):
        step_num += 1

        # Agent decides action based on observation
        action = Action(name="increment", arguments={})
        print(f"\nStep {step_num}: Agent chose action '{action.name}'")

        # Execute step
        env_output = task.step(action)
        obs = env_output.obs

        print(f"  Observation: {obs.contents[0].data}")
        print(f"  Reward: {env_output.reward}")
        print(f"  Done: {env_output.done}")
        print(f"  Info: {env_output.info}")

        if env_output.done:
            break

        if step_num >= 5:  # Safety limit
            break

    assert task.tool.counter == 3, f"Expected counter to be 3, got {task.tool.counter}"
    print(f"\n✓ Task completed successfully in {step_num} steps!")

    # === Test 2: Multiple actions in one step ===
    print("\n" + "=" * 60)
    print("Test 2: Multiple actions in one step (count-to-5)")
    print("=" * 60)

    task_config2 = task_configs[1]
    task2 = task_config2.make()

    # Get initial observation
    obs, info = task2.setup()
    print(f"Initial observation: {obs.contents[0].data}")

    # Agent predicts multiple actions at once
    actions = [
        Action(name="increment", arguments={}),
        Action(name="get_value", arguments={}),
        Action(name="increment", arguments={}),
    ]
    print(f"\nAgent chose {len(actions)} actions: {[a.name for a in actions]}")

    # Execute multiple actions in one step
    env_output = task2.step(actions)
    obs = env_output.obs

    print("Observations from all actions:")
    for i, content in enumerate(obs.contents):
        print(f"  {i + 1}. {content.data}")
    print(f"Reward: {env_output.reward}, Done: {env_output.done}")

    # Continue with single actions
    while not env_output.done:
        action = Action(name="increment", arguments={})
        env_output = task2.step(action)
        print(f"Action '{action.name}' -> {env_output.obs.contents[0].data}")

        if task2.tool.counter >= 5:
            break

    assert task2.tool.counter == 5, f"Expected counter to be 5, got {task2.tool.counter}"
    assert env_output.reward == 1.0, f"Expected reward 1.0, got {env_output.reward}"
    print(f"\n✓ Task completed with reward {env_output.reward}!")

    # === Test 3: Tool action execution (lower level) ===
    print("\n" + "=" * 60)
    print("Test 3: Direct tool action execution (lower level API)")
    print("=" * 60)

    task3 = task_configs[0].make()
    task3.setup()

    # Lower level: directly call tool.execute_action()
    print("Using task.tool.execute_action() directly:")
    action = Action(name="increment", arguments={})
    result = task3.tool.execute_action(action)
    assert isinstance(result, Observation), "Expected an Observation result from tool action"
    print(f"  Result: {result.contents[0].data}")
    print("Note: This bypasses task.step() and doesn't trigger evaluation")

    # === Test 4: Task isolation ===
    print("\n" + "=" * 60)
    print("Test 4: Task isolation")
    print("=" * 60)

    task4 = task_configs[0].make()
    task4.setup()

    assert task4.tool.counter == 0, "New task should have fresh tool with counter=0"
    assert task.tool.counter == 3, "Original task should still have counter=3"
    print("✓ Task isolation verified:")
    print(f"  - New task counter: {task4.tool.counter}")
    print(f"  - Original task counter: {task.tool.counter}")

    # Cleanup
    task.close()
    task2.close()
    task3.close()
    task4.close()
    benchmark.close()

    print("\n" + "=" * 60)
    print("✓ All tests passed! New API working correctly.")
    print("=" * 60)


# Main
if __name__ == "__main__":
    test_simple_counting()
