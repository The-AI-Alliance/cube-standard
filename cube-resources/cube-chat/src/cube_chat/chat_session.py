"""In-memory chat session implementation using a queue."""

import queue
import time

from cube.resources.chat_session import ChatConfig, ChatMessage, ChatRole, ChatSession

_STORED_ROLES = {"user", "assistant", "infeasible"}
_VALID_ROLES = {"user", "assistant", "info", "infeasible"}
_STOP_SENTINEL = None


class SessionStoppedError(Exception):
    """Raised by wait_for_user_message() when stop() has been called."""


class BasicChatConfig(ChatConfig):
    """Configuration for an in-memory BasicChatSession.

    No configuration fields are needed for the in-memory implementation.
    """

    def make(self) -> "BasicChatSession":
        """Create and return a new BasicChatSession."""
        return BasicChatSession()


class BasicChatSession(ChatSession):
    """In-memory chat session backed by a queue.

    Messages posted with add_message() for roles "user", "assistant", and
    "infeasible" are stored in history. The "info" role is silently dropped
    (not stored). send_message() appends an "assistant" message and unblocks
    any caller waiting in wait_for_user_message().
    """

    def __init__(self) -> None:
        self._messages: list[ChatMessage] = []
        self._queue: queue.SimpleQueue[str | None] = queue.SimpleQueue()

    @property
    def messages(self) -> list[ChatMessage]:
        """Return a copy of the full message history."""
        return self._messages[:]

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
            self._messages.append(ChatMessage(role=role, timestamp=time.time(), message=msg))

    def wait_for_user_message(self) -> str:
        """Block until the agent sends a message and return it.

        Returns
        -------
        str
            The agent's message text.

        Raises
        ------
        SessionStoppedError
            If stop() was called while blocked or before this call.
        """
        msg = self._queue.get()
        if msg is _STOP_SENTINEL:
            raise SessionStoppedError("Session has been stopped")
        return msg

    def send_message(self, text: str) -> None:
        """Send a message from the agent side, unblocking wait_for_user_message().

        Appends an "assistant" message to history and puts the text into the queue.

        Parameters
        ----------
        text : str
            Message content to send.
        """
        self._messages.append(ChatMessage(role="assistant", timestamp=time.time(), message=text))
        self._queue.put(text)

    def stop(self) -> None:
        """Unblock any thread waiting in wait_for_user_message() and mark the session stopped."""
        self._queue.put(_STOP_SENTINEL)
