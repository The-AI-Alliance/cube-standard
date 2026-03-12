from cube.container import Container
from cube.core import ActionSchema
from cube.tool import Tool, ToolConfig, tool_action


class CounterEnv:
    """Simple counter environment"""

    def __init__(self, initial_value: int = 0):
        self.counter = initial_value

    def reset(self):
        self.counter = 0


class CounterToolConfig(ToolConfig):
    """Configuration for CounterTool."""

    enable_increment_by: bool = False
    enable_decrement: bool = False

    def make(self, container: Container | None = None) -> "CounterTool":
        """Instantiate the tool. `container` is None for tools that don't need a VM."""
        return CounterTool(self)


class CounterTool(Tool):
    def __init__(self, config: CounterToolConfig):
        # TODO: _env for envionrment reference may become a requireed convention
        self._env = CounterEnv()  # put env pointer always in _env.
        self._config = config

    def reset(self):
        self._env.reset()

    @property
    def action_set(self) -> list[ActionSchema]:
        return [
            a
            for a in super().action_set
            if (a.name != "decrement" or self._config.enable_decrement)
            and (a.name != "increment_by" or self._config.enable_increment_by)
        ]

    # default actions
    @tool_action
    def increment(self) -> str:
        """Increment the counter by 1."""
        self._env.counter += 1
        return f"Counter value is: {self._env.counter}"

    @tool_action
    def get_value(self) -> str:
        """Get the current counter value."""
        return f"Counter value is: {self._env.counter}"

    @tool_action
    def decrement(self) -> str:
        """Decrement the counter by 1."""
        self._env.counter -= 1
        return f"Counter value is: {self._env.counter}"

    @tool_action
    def increment_by(self, value: int) -> str:
        """Increment the counter by a specified amount."""
        self._env.counter += value
        return f"Counter value is: {self._env.counter}"
