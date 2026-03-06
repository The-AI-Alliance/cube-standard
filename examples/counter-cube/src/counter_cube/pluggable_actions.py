"""Alternative environment pattern: plug in plain functions as actions at runtime via add_environment_action()."""

from functools import partial
from typing import Callable

from cube.container import Container
from cube.environment import Environment, EnvironmentConfig, environment_action


## Simple action functions originally scoped outside the environment class.


def get_value(env: "CounterEnvironmentPluggable") -> str:
    """Get the current counter value."""
    return f"Counter value is: {env.counter}"


def decrement(env: "CounterEnvironmentPluggable") -> str:
    """Decrement the counter by 1."""
    env.counter -= 1
    return f"Counter value is: {env.counter}"


def increment_by(env: "CounterEnvironmentPluggable", value: int) -> str:
    """Increment the counter by a specified amount."""
    env.counter += value
    return f"Counter value is: {env.counter}"


class CounterEnvironmentPluggableConfig(EnvironmentConfig):
    """Configuration for CounterEnvironmentPluggable."""

    enable_increment_by: bool = False
    enable_decrement: bool = False

    def make(self, container: Container | None = None) -> "CounterEnvironmentPluggable":
        """Instantiate the environment. `container` is None for environments that don't need a VM."""
        return CounterEnvironmentPluggable(self)


class CounterEnvironmentPluggable(Environment):
    def __init__(self, config: CounterEnvironmentPluggableConfig):
        self.counter = 0

        if config.enable_decrement:
            self.add_environment_action(decrement)

        if config.enable_increment_by:
            self.add_environment_action(increment_by)

    def reset(self):
        self.counter = 0

    # default actions
    @environment_action
    def increment(self) -> str:
        """Increment the counter by 1."""
        self.counter += 1
        return f"Counter value is: {self.counter}"

    def add_environment_action(self, func) -> None:
        """Dynamically add a plain function as an action on this instance.

        func must accept env as its first argument. All remaining parameters
        become the agent-visible arguments in the ActionSchema.
        """
        bound: Callable = partial(func, self)
        bound._is_action = True
        bound.__name__ = func.__name__
        bound.__doc__ = func.__doc__
        setattr(self, bound.__name__, bound)
