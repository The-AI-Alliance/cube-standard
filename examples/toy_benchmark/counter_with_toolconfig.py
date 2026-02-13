"""
Example demonstrating ToolConfig for research flexibility.

This example shows how researchers can create custom ToolConfig implementations
to swap tool behavior without modifying benchmark code.
"""

from counter import CounterBenchmark, CounterTaskConfig

from cube.core import Action, ActionSchema
from cube.tool import Tool, ToolConfig, tool_action


# Example 1: Basic ToolConfig with configurable features
class ConfigurableCounterTool(Tool):
    """Counter tool with optional decrement and reset."""

    def __init__(self, target: int, enable_decrement: bool = False, enable_reset: bool = False):
        """Initialize counter tool with optional features."""
        self.counter = 0
        self.target = target
        self.history: list[str] = []
        self.enable_decrement = enable_decrement
        self.enable_reset = enable_reset

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


class ConfigurableCounterToolConfig(ToolConfig):
    """Custom tool configuration with configurable features."""

    target: int = 0
    enable_decrement: bool = False
    enable_reset: bool = False

    def make(self) -> Tool:
        """Create tool instance with configured features."""
        return ConfigurableCounterTool(
            target=self.target,
            enable_decrement=self.enable_decrement,
            enable_reset=self.enable_reset,
        )


# Example 2: Advanced ToolConfig with different increment behavior
class DoubleIncrementTool(Tool):
    """Counter tool that increments by 2."""

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
        """Increment the counter by 2 (research variant)"""
        self.counter += 2
        self.history.append("increment")
        return f"Counter is now {self.counter} (incremented by 2)"

    @tool_action
    def get_value(self) -> str:
        """Get the current counter value"""
        return f"Counter value is: {self.counter}"


class DoubleIncrementToolConfig(ToolConfig):
    """Alternative tool implementation that increments by 2."""

    target: int = 0

    def make(self) -> Tool:
        """Create double increment tool."""
        return DoubleIncrementTool(target=self.target)


def test_configurable_toolconfig():
    """Test configurable ToolConfig with decrement enabled."""
    print("\n" + "=" * 60)
    print("Test 1: Configurable ToolConfig with decrement")
    print("=" * 60)

    # Create benchmark with custom ToolConfig
    benchmark = CounterBenchmark()
    benchmark.setup()

    # Create task config with decrement enabled, reset disabled
    task_config = CounterTaskConfig(
        task_id="count-to-3",
        target=3,
        tool_config=ConfigurableCounterToolConfig(target=3, enable_decrement=True, enable_reset=False),
    )

    # Create task
    task = task_config.make()  # type: ignore[assignment]
    task.setup()

    # Check available actions
    actions = task.action_set
    action_names = [action.name for action in actions]
    print(f"Available actions: {action_names}")
    assert "decrement" in action_names, "Expected 'decrement' action"
    assert "reset_counter" not in action_names, "Should not have 'reset_counter' action (disabled)"

    # Test increment
    env_output = task.step(Action(name="increment", arguments={}))
    print(f"Increment: {env_output.obs.contents[0].data}")
    assert task.tool.counter == 1  # type: ignore[attr-defined]

    # Test decrement (only available with ToolConfig)
    env_output = task.step(Action(name="decrement", arguments={}))
    print(f"Decrement: {env_output.obs.contents[0].data}")
    assert task.tool.counter == 0, f"Expected counter to be 0, got {task.tool.counter}"  # type: ignore[attr-defined]

    print("✓ Configurable ToolConfig test passed!")


def test_double_increment_toolconfig():
    """Test alternative tool implementation."""
    print("\n" + "=" * 60)
    print("Test 2: Double Increment ToolConfig")
    print("=" * 60)

    # Create benchmark with different ToolConfig
    benchmark = CounterBenchmark()
    benchmark.setup()

    # Create task config with double increment
    task_config = CounterTaskConfig(
        task_id="count-to-4",
        target=4,
        tool_config=DoubleIncrementToolConfig(target=4),
    )

    # Create task
    task = task_config.make()  # type: ignore[assignment]
    task.setup()

    # Test double increment
    env_output = task.step(Action(name="increment", arguments={}))
    print(f"Increment: {env_output.obs.contents[0].data}")
    assert task.tool.counter == 2, f"Expected counter to be 2, got {task.tool.counter}"  # type: ignore[attr-defined]

    # Increment again
    env_output = task.step(Action(name="increment", arguments={}))
    print(f"Increment: {env_output.obs.contents[0].data}")
    assert task.tool.counter == 4, f"Expected counter to be 4, got {task.tool.counter}"  # type: ignore[attr-defined]
    assert env_output.done, "Task should be done"
    assert env_output.reward == 1.0, f"Expected reward 1.0, got {env_output.reward}"

    print("✓ Double increment ToolConfig test passed!")


def test_default_toolconfig():
    """Test default ToolConfig from counter.py."""
    print("\n" + "=" * 60)
    print("Test 3: Default ToolConfig (from counter.py)")
    print("=" * 60)

    # Create benchmark - uses task configs with default CounterToolConfig
    benchmark = CounterBenchmark()
    benchmark.setup()

    # Load tasks (they use the default CounterToolConfig)
    task_configs = benchmark.load_tasks()
    task = task_configs[0].make()
    task.setup()

    # List tools
    actions = task.action_set
    action_names = [action.name for action in actions]
    print(f"Available actions: {action_names}")
    assert "increment" in action_names, "Expected 'increment' action"
    assert "get_value" in action_names, "Expected 'get_value' action"
    # Decrement and reset should not be available with default config
    assert "decrement" not in action_names, "Should not have 'decrement' action (default config)"
    assert "reset_counter" not in action_names, "Should not have 'reset_counter' action (default config)"

    # Test standard increment
    env_output = task.step(Action(name="increment", arguments={}))
    print(f"Increment: {env_output.obs.contents[0].data}")
    assert task.tool.counter == 1, f"Expected counter to be 1, got {task.tool.counter}"  # type: ignore[attr-defined]

    print("✓ Default ToolConfig test passed!")


def main():
    """Run all ToolConfig examples."""
    print("=" * 60)
    print("ToolConfig Examples - Research Flexibility Demo")
    print("=" * 60)

    test_configurable_toolconfig()
    test_double_increment_toolconfig()
    test_default_toolconfig()

    print("\n" + "=" * 60)
    print("All ToolConfig examples passed!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("- ToolConfig enables swapping tool implementations")
    print("- Researchers can add/remove tools via configuration")
    print("- Researchers can change tool behavior (e.g., increment by 2)")
    print("- Every benchmark must provide a ToolConfig implementation")
    print("- Tools can filter their action_set based on configuration")


if __name__ == "__main__":
    main()
