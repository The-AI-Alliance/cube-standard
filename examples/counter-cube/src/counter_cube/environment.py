from cube.container import Container
from cube.core import ActionSchema
from cube.environment import Environment, EnvironmentConfig, environment_action


class CounterEnvironmentConfig(EnvironmentConfig):
    """Configuration for CounterEnvironment."""

    enable_increment_by: bool = False
    enable_decrement: bool = False

    def make(self, container: Container | None = None) -> "CounterEnvironment":
        """Instantiate the environment. `container` is None for environments that don't need a VM."""
        return CounterEnvironment(self)


class CounterEnvironment(Environment):
    def __init__(self, config: CounterEnvironmentConfig):
        self.counter = 0
        self._config = config

    def reset(self):
        self.counter = 0

    @property
    def action_set(self) -> list[ActionSchema]:
        return [
            a
            for a in super().action_set
            if (a.name != "decrement" or self._config.enable_decrement)
            and (a.name != "increment_by" or self._config.enable_increment_by)
        ]

    # default actions
    @environment_action
    def increment(self) -> str:
        """Increment the counter by 1."""
        self.counter += 1
        return f"Counter value is: {self.counter}"

    @environment_action
    def get_value(self) -> str:
        """Get the current counter value."""
        return f"Counter value is: {self.counter}"

    @environment_action
    def decrement(self) -> str:
        """Decrement the counter by 1."""
        self.counter -= 1
        return f"Counter value is: {self.counter}"

    @environment_action
    def increment_by(self, value: int) -> str:
        """Increment the counter by a specified amount."""
        self.counter += value
        return f"Counter value is: {self.counter}"
