"""In-memory chat session implementation using a queue."""

import queue
import time

from cube.resources.chat_session import ChatConfig, ChatRole, ChatSession

_STORED_ROLES = {"user", "assistant", "infeasible"}
_VALID_ROLES = {"user", "assistant", "info", "infeasible"}


class SimpleChatConfig(ChatConfig):
    """Configuration for an in-memory SimpleChatSession.

    No configuration fields are needed for the in-memory implementation.
    """

    def make(self) -> "SimpleChatSession":
        """Create and return a new SimpleChatSession."""
        return SimpleChatSession()


class SimpleChatSession(ChatSession):
    """In-memory chat session backed by a queue.

    Messages posted with add_message() for roles "user", "assistant", and
    "infeasible" are stored in history. The "info" role is silently dropped
    (not stored). send_message() appends an "assistant" message and unblocks
    any caller waiting in wait_for_user_message().
    """

    def __init__(self) -> None:
        self._messages: list[dict] = []
        self._queue: queue.SimpleQueue[str] = queue.SimpleQueue()

    @property
    def messages(self) -> list[dict]:
        """Return a copy of the full message history."""
        return list(self._messages)

    def add_message(self, role: ChatRole, msg: str) -> None:
        """Post a message to the chat from the task side.

        Parameters
        ----------
        role : ChatRole
            Role of the message sender. "info" messages are not stored.
        msg : str
            Message content.

        Raises
        ------
        ValueError
            If role is not a valid ChatRole.
        """
        if role not in _VALID_ROLES:
            raise ValueError(f"Invalid role '{role}'. Must be one of: {sorted(_VALID_ROLES)}")
        if role in _STORED_ROLES:
            self._messages.append({"role": role, "timestamp": time.time(), "message": msg})

    def wait_for_user_message(self) -> str:
        """Block until the agent sends a message and return it.

        Returns
        -------
        str
            The agent's message text.
        """
        return self._queue.get()

    def send_message(self, text: str) -> None:
        """Send a message from the agent side, unblocking wait_for_user_message().

        Appends an "assistant" message to history and puts the text into the queue.

        Parameters
        ----------
        text : str
            Message content to send.
        """
        self._messages.append({"role": "assistant", "timestamp": time.time(), "message": text})
        self._queue.put(text)

    def stop(self) -> None:
        """No-op: in-memory sessions hold no external resources."""
        pass
