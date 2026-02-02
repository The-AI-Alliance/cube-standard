"""
@Nicolas, Not sure why we need this for CUBE specifications after shifting to MCP python API.    
"""


import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Protocol, Type, TypeAlias

import litellm.utils
from mcp.types import Tool as MCPTool
from typing_extensions import get_protocol_members

from cube.types import Action, Content, Observation, TypedBaseModel

logger = logging.getLogger(__name__)

class ActionSpace(Protocol):
    """Base class for action spaces."""

    pass


ActionSubset: TypeAlias = tuple[Callable, ...]


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
    def get_actions(self) -> list[MCPTool]:
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

    def get_actions(self) -> list[MCPTool]:
        """Returns list of actions supported by that tool."""
        action_names = get_protocol_members(self.action_space)
        tools = []
        for name in action_names:
            func = getattr(self, name)
            schema = litellm.utils.function_to_dict(func)
            # litellm returns 'parameters', rename to 'inputSchema' for MCP compliance
            if "parameters" in schema:
                schema["inputSchema"] = schema.pop("parameters")
            tools.append(MCPTool(**schema))
        return tools
