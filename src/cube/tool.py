import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Type

from typing_extensions import get_protocol_members

from cube.core import Action, ActionSchema, ActionSpace, Content, Observation, TypedBaseModel

logger = logging.getLogger(__name__)


class AbstractTool(ABC):
    """
    Abstract interface for objects that can react on a list of actions.
    List defined by the ActionSpace that tool inherits.
    """

    def reset(self) -> None:
        """Optional reset the tool to its initial state."""
        pass

    @abstractmethod
    def execute_action(self, action: Action) -> Any:
        """Execute a single action and return the result."""
        pass

    @abstractmethod
    def get_actions(self) -> List[ActionSchema]:
        """Returns list of actions supported by that tool."""
        pass

    def close(self) -> None:
        """Optional clean up tool resources."""
        pass


class ToolConfig(TypedBaseModel, ABC):
    """Base class for tool configurations."""

    @abstractmethod
    def make(self) -> AbstractTool:
        pass


class Tool(AbstractTool):
    """
    Base class for tool that implements an action space protocol.

    :var Returns: Description
    """

    action_space: Type[ActionSpace]

    def get_action_method(self, action) -> Callable:
        if not getattr(self.action_space, action.name, None):
            raise ValueError(f"Action {action.name} is not a part of {self.action_space}.")
        if not (fn := getattr(self, action.name, None)):
            raise ValueError(f"Action {action.name} is not implemented in {self.__class__.__name__}.")
        return fn

    def execute_action(self, action: Action) -> Observation:
        fn = self.get_action_method(action)
        try:
            action_result = fn(**action.arguments) or "Success"
        except Exception as e:
            action_result = f"Error executing action {action.name}: {e}"
            logger.exception(action_result)
        return Observation(contents=[Content(data=action_result, tool_call_id=action.id)])

    def get_actions(self) -> List[ActionSchema]:
        """Returns list of actions supported by that tool."""
        action_names = get_protocol_members(self.action_space)
        return [ActionSchema.from_function(getattr(self, name)) for name in action_names]
