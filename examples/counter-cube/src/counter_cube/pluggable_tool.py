"""Alternative tool pattern: plug in plain functions as actions at runtime via add_tool_action()."""

from functools import partial
from typing import Callable

from cube.containers import Container
from cube.tool import Tool, ToolConfig, tool_action
from .tool import CounterEnv


type env = CounterEnv

## Simple action functions originally scoped outside the tool class.


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

    def make(self, container: Container | None = None) -> "CounterToolPluggable":
        """Instantiate the tool. `container` is None for tools that don't need a VM."""
        return CounterToolPluggable(self)


class CounterToolPluggable(Tool):
    def __init__(self, config: CounterToolConfig):
        self._env = CounterEnv()  # put env pointer always in _env. # TODO: Make this a requireed convention

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
