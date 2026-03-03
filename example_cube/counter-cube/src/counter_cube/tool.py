"""Step 1 & 2 of 4 — Tool and ToolConfig.

A Tool wraps whatever environment the agent acts in: a web browser, a VM
desktop, a database connection, or — as here — a plain Python object.

Rules:
  • Keep __init__ plain Python (not Pydantic). ToolConfig (Step 2) is the
    serializable factory; Tool itself is ephemeral runtime state.
  • Implement reset() if the tool needs to be restarted between episodes.
    Task.reset() calls this for you (see task.py).
  • All @tool_action methods must return a value that can be wrapped in an
    Observation: str, PIL image, dict, etc. Returning a plain str is fine
    for text-only environments.

ToolConfig is a Pydantic model that carries the parameters needed to build
a Tool. Its two jobs:
  1. Be JSON-serializable so it can cross process boundaries (e.g. when
     tasks are dispatched to remote workers with Ray or an RPC server).
  2. Act as the factory: implement make() to instantiate the Tool.

The split between ToolConfig (data) and Tool (runtime object) is
intentional. You should never pass a live Tool over the wire — only its
config. Workers deserialize the config and call make() locally.

ToolConfig is also where you document the knobs available to benchmark
users who want to customise tool behaviour without forking the benchmark.
"""

from functools import partial
from typing import Callable

from cube.containers import Container
from cube.tool import Tool, ToolConfig, tool_action


class CounterEnv:
    """Simple counter environment"""

    def __init__(self, initial_value: int = 0):
        self.counter = initial_value

    def reset(self):
        self.counter = 0


type env = CounterEnv

## Simple action functions


def get_value(env) -> str:
    """Get the current counter value."""
    return f"Counter value is: {env.counter}"


def decrement(env) -> str:
    """Decrement the counter by 1."""
    env.counter -= 1
    return f"Counter value is: {env.counter}"


def increment_by(env, value: int) -> str:
    """Increment the counter by a specified amount."""
    env.counter += value
    return f"Counter value is: {env.counter}"


class CounterToolConfig(ToolConfig):
    """Configuration for CounterTool."""

    enable_increment_by: bool = False
    enable_decrement: bool = False

    def make(self, container: Container | None = None) -> "CounterTool":
        """Instantiate the tool. `container` is None for tools that don't need a VM."""
        return CounterTool(self)


class CounterTool(Tool):
    def __init__(self, config: CounterToolConfig | None = None):
        self._env = CounterEnv()  # put env pointer always in _env. # TODO: Make this a requireed convention

        if config is None:
            config = CounterToolConfig()  # default config if none provided.

        if config.enable_decrement:
            self.add_tool_action(decrement)

        if config.enable_increment_by:
            self.add_tool_action(increment_by)

    def reset(self):
        self._env.reset()

    # default actions
    @tool_action
    def increment(self) -> str:
        """Increment the counter by 1."""
        self._env.counter += 1
        return f"Counter value is: {self._env.counter}"

    def add_tool_action(self, func) -> None:
        """Dynamically add a plain function as an action on this instance.

        func must accept env as its first argument. All remaining parameters
        become the agent-visible arguments in the ActionSchema.
        """
        bound: Callable = partial(func, self._env)
        bound._is_action = True
        bound.__name__ = func.__name__
        bound.__doc__ = func.__doc__
        setattr(self, bound.__name__, bound)
