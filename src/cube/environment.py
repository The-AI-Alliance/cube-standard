"""
Environment configuration for CUBE benchmarks.

This module defines AbstractEnvironment, Environment, EnvironmentConfig, and the
@environment_action decorator for implementing and configuring agent action interfaces.

Abstract classes:
    AbstractEnvironment — subclasses must implement:
        - execute_action(action: Action) -> Observation | StepError
        - action_set (property) -> list[ActionSchema]
    Environment is a concrete subclass of AbstractEnvironment that implements both
    automatically via the @environment_action decorator — subclass Environment instead
    of AbstractEnvironment directly.

    EnvironmentConfig — subclasses must implement:
        - make(container) -> AbstractEnvironment    instantiate the environment from
                                                    serialized config data, connecting
                                                    to the container if one was launched

Example — defining a custom environment and its config:

    from cube.environment import Environment, EnvironmentConfig, environment_action
    from cube.container import Container

    class BrowserEnvironment(Environment):
        base_url: str

        @environment_action
        def navigate(self, url: str) -> str:
            '''Navigate to a URL and return the page title.'''
            ...

        @environment_action
        def click(self, selector: str) -> str:
            '''Click on an element identified by a CSS selector.'''
            ...

    class BrowserEnvironmentConfig(EnvironmentConfig):
        base_url: str = "http://localhost:9222"

        def make(self, container: Container | None = None) -> BrowserEnvironment:
            url = container.get_url(port=9222) if container else self.base_url
            return BrowserEnvironment(base_url=url)

The BrowserEnvironmentConfig can then be passed to a Task or Benchmark, letting
harness users swap environment backends without touching benchmark logic.
"""

import logging
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Callable, List

from cube.container import Container
from cube.core import Action, ActionSchema, Content, Observation, StepError, TypedBaseModel

logger = logging.getLogger(__name__)


class AbstractEnvironment(ABC):
    """
    Abstract interface for objects that can react on a list of actions.
    List defined by the functions that environment implements.
    """

    def reset(self) -> None:
        """Optional: reset the environment to its initial state."""
        pass

    def close(self) -> None:
        """Optional: clean up environment resources (connections, processes, files, etc.)."""
        pass

    @abstractmethod
    def execute_action(self, action: Action) -> Any:
        """Execute a single action and return the result."""
        pass

    @property
    @abstractmethod
    def action_set(self) -> List[ActionSchema]:
        """
        Returns list of actions supported by that environment.
        Environment definitions in litellm-compatible format.

        Returns a JSON-serializable list of action descriptors, each with:
        - type: "function"
        - function: {name, description, parameters (JSON Schema)}

        This format is compatible with litellm/OpenAI function calling.
        Agents use this to discover available actions without knowing
        environment implementations in advance.

        Example return value:
        [
            {
                "type": "function",
                "function": {
                    "name": "click",
                    "description": "Click on a web element",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "selector": {"type": "string", "description": "CSS selector"}
                        },
                        "required": ["selector"]
                    }
                }
            }
        ]
        """
        pass


class EnvironmentConfig(TypedBaseModel, ABC):
    """
    Configuration for creating task-specific environments.
    """

    @abstractmethod
    def make(self, container: Container | None = None) -> AbstractEnvironment:
        """
        Instantiate Environment from configuration data.

        Args:
            container: The launched container for this task, if any. Use it to
                       extract connection info (host, ports) to configure the
                       environment's endpoint. None if the task needs no container.

        Returns:
            AbstractEnvironment instance
        """
        pass


def environment_action(func: Callable) -> Callable:
    """
    Decorator to mark a method as a primitive action in an Environment.

    This decorator automatically registers methods as actions that will be
    discovered by the Environment's action_set property.

    Usage:
        class MyEnvironment(AbstractEnvironment):
            state = 0

            @environment_action
            def my_action(self, param: str) -> str:
                '''Action description.'''
                state += 1
                return f"State is now {state}"
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    # Mark the function as an action
    setattr(wrapper, "_is_action", True)
    return wrapper


class Environment(AbstractEnvironment):
    """
    Base class for environments with automatic action discovery via decorators.

    Environment subclasses should mark their primitive action methods with the @environment_action decorator.
    The action_set property will automatically discover and expose these methods.

    Example:
        ```python
        from cube.environment import Environment, environment_action, Action

        class CalculatorEnvironment(Environment):
            '''Calculator environment with basic arithmetic operations.'''
            result: float = 0.0

            @environment_action
            def add(self, a: float, b: float) -> str:
                '''Add two numbers together.'''
                self.result = a + b
                return f"Result: {self.result}"

        # Usage
        calc = CalculatorEnvironment()

        # Automatic discovery of actions
        print("Available actions:")
        for action_schema in calc.action_set:
            print(f"  - {action_schema.name}: {action_schema.description}")
        # Output: - add: Add two numbers together.

        # Execute an action
        action = Action(name="add", arguments={"a": 5.0, "b": 3.0})
        result = calc.execute_action(action)
        print(result.contents[0].data)  # "Result: 8.0"
        ```

    Benefits:
        - Zero boilerplate: Just add @environment_action decorator
        - Single source of truth: Method signature and docstring define the action
        - No duplication: Each function defined exactly once
        - Clear intent: Obvious which methods are actions
    """

    def get_action_method(self, action: Action) -> Callable:
        """Return the bound method for an action, or raise ValueError if it is not registered.

        Raises distinct errors for:
        - Method that does not exist on the class at all.
        - Method that exists but is not decorated with @environment_action.
        """
        # Check instance dict first — catches dynamically attached actions (not in any class dict)
        method = self.__dict__.get(action.name)
        if method and callable(method) and getattr(method, "_is_action", False):
            return method
        method = getattr(self, action.name, None)
        if not method:
            raise ValueError(f"Action {action.name} does not exist in {self.__class__.__name__}.")
        is_registered = any(
            getattr(cls.__dict__.get(action.name), "_is_action", False)
            for cls in type(self).__mro__
            if action.name in cls.__dict__
        )
        if not is_registered:
            raise ValueError(
                f"Action {action.name} exists in {self.__class__.__name__} but is not decorated with @environment_action. Add @environment_action to expose it as an action."
            )
        return method

    def execute_action(self, action: Action) -> Observation | StepError:
        """Execute an action by name."""
        method = self.get_action_method(action)
        try:
            action_result = method(**action.arguments) or "Success"
        except Exception as e:
            action_result = f"Error executing action {action.name}: {e}"
            logger.exception(action_result)
            return StepError.from_exception(e)
        return Observation(contents=[Content.from_data(action_result, tool_call_id=action.id)])

    async def async_execute_action(self, action: Action) -> Observation | StepError:
        """Async version of execute_action for tools whose action methods are coroutines."""
        method = self.get_action_method(action)
        try:
            action_result = (await method(**action.arguments)) or "Success"
        except Exception as e:
            action_result = f"Error executing action {action.name}: {e}"
            logger.exception(action_result)
            return StepError.from_exception(e)
        return Observation(contents=[Content.from_data(action_result, tool_call_id=action.id)])

    @property
    def action_set(self) -> List[ActionSchema]:
        """Automatically discover all methods marked with @environment_action decorator."""
        actions = []

        # Introspect the class to find all methods marked as actions
        for attr_name in dir(self):
            # Skip private/protected methods and the action_set property itself
            if attr_name.startswith("_") or attr_name == "action_set":
                continue

            # Skip properties — calling getattr on a property invokes its getter,
            # which may have side effects (e.g. raising if a resource is not yet initialized).
            if any(
                isinstance(cls.__dict__.get(attr_name), property)
                for cls in type(self).__mro__
                if attr_name in cls.__dict__
            ):
                continue

            attr = getattr(self, attr_name)

            # Check if this attr_name is a method marked as an action.
            # We walk up the class hierarchy (method resolution order, MRO)
            # because a subclass may override a method without repeating
            # @environment_action - as long as the decorator appears on the method
            # in any parent class, the override is still treated as an action.
            is_action = any(
                getattr(cls.__dict__.get(attr_name), "_is_action", False)
                for cls in type(self).__mro__
                if attr_name in cls.__dict__
            )
            if callable(attr) and is_action:
                actions.append(ActionSchema.from_function(attr))

        # Also discover instance-level actions attached via setattr (not in any class dict)
        for name, attr in self.__dict__.items():
            if not name.startswith("_") and callable(attr) and getattr(attr, "_is_action", False):
                actions.append(ActionSchema.from_function(attr))

        return actions
