"""Abstract chat session contract for cube-standard tools.

A ChatSession is a side-channel communication interface between the agent and the task.
Tasks use it to post messages (goals, instructions, info) and to wait for agent responses.
Agents use it to send messages, which unblocks the task.

ChatConfig is the serializable factory for creating a ChatSession.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, TypedDict

from cube.core import TypedBaseModel

ChatRole = Literal["user", "assistant", "info", "infeasible"]


class ChatMessage(TypedDict):
    role: ChatRole
    timestamp: float
    message: str


class ChatConfig(TypedBaseModel, ABC):
    """Abstract serializable config for a chat session.

    Call make() to create a live ChatSession. The config holds all parameters
    needed to reproduce the session and must be fully serializable.

    Subclasses:
    - BasicChatConfig: In-memory queue-based session (in cube-resources)
    """

    @abstractmethod
    def make(self) -> "ChatSession":
        """Create and return a live ChatSession."""
        ...


class ChatSession(ABC):
    """Abstract live chat session handle.

    Manages bidirectional communication between the task and the agent.
    Tasks call add_message() to post messages and wait_for_user_message() to
    block until the agent responds. Agents call send_message() to respond.

    Subclasses:
    - BasicChatSession: In-memory queue implementation (in cube-resources)
    """

    @property
    @abstractmethod
    def messages(self) -> list[ChatMessage]:
        """Full message history."""
        ...

    @abstractmethod
    def add_message(self, role: ChatRole, msg: str) -> None:
        """Post a message to the chat from the task side.

        Parameters
        ----------
        role : ChatRole
            Role of the message sender. "info" messages are not stored in history.
        msg : str
            Message content.
        """
        ...

    @abstractmethod
    def wait_for_user_message(self) -> str:
        """Block until the agent sends a message and return it.

        Returns
        -------
        str
            The agent's message text.
        """
        ...

    @abstractmethod
    def send_message(self, text: str) -> None:
        """Send a message from the agent side, unblocking wait_for_user_message().

        Implementations must append the message to history with role "assistant"
        in addition to unblocking any caller waiting in wait_for_user_message().

        Parameters
        ----------
        text : str
            Message content to send.
        """
        ...

    @abstractmethod
    def stop(self) -> None:
        """Clean up resources held by this session."""
        ...
