"""Chat tool for cube-standard benchmarks.

Provides ``ChatToolConfig`` and ``ChatTool``, a concrete implementation of the
``Tool`` base class that wraps a ``ChatSession`` and exposes ``send_message``
and ``report_infeasible`` as the agent-facing actions.
"""

import datetime

from cube.resources.chat_session import ChatConfig, ChatRole, ChatSession
from cube.tool import Tool, ToolConfig, tool_action
from cube_chat import BasicChatConfig
from pydantic import Field


class ChatToolConfig(ToolConfig):
    """Configuration for ``ChatTool``.

    Parameters
    ----------
    chat : ChatConfig
        Chat session configuration. Defaults to an in-memory BasicChatConfig.
    """

    chat: ChatConfig = Field(default_factory=BasicChatConfig)

    def make(self, container=None) -> "ChatTool":
        """Create a ChatTool from this configuration."""
        session = self.chat.make()
        return ChatTool(config=self, session=session)


class ChatTool(Tool):
    """Chat tool that exposes a side-channel message interface to the agent.

    The agent uses ``send_message`` to communicate with the task. The task uses
    ``add_message`` and ``wait_for_user_message`` to drive the conversation.
    After every agent action, ``chat_obs()`` is appended to the observation so
    the agent always sees the up-to-date message history.

    Call ``reset()`` between episodes to get a fresh session. Call ``close()``
    to release session resources.
    """

    def __init__(self, config: ChatToolConfig, session: ChatSession) -> None:
        self.config = config
        self._session = session

    @property
    def session(self) -> ChatSession:
        """Expose the raw ChatSession for advanced use (escape hatch, SR-003)."""
        return self._session

    @property
    def messages(self) -> list[dict]:
        """Delegate to session.messages."""
        return self._session.messages

    # ------------------------------------------------------------------
    # Tool lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Stop the current session and create a fresh one."""
        self._session.stop()
        self._session = self.config.chat.make()

    def close(self) -> None:
        """Release session resources."""
        self._session.stop()

    # ------------------------------------------------------------------
    # Task-internal methods (not agent-facing actions)
    # ------------------------------------------------------------------

    def add_message(self, role: ChatRole, msg: str) -> None:
        """Post a message to the chat from the task side.

        Parameters
        ----------
        role : ChatRole
            Role of the message sender.
        msg : str
            Message content.
        """
        self._session.add_message(role, msg)

    def wait_for_user_message(self) -> str:
        """Block until the agent sends a message and return it.

        Returns
        -------
        str
            The agent's message text.
        """
        return self._session.wait_for_user_message()

    def chat_obs(self) -> str:
        """Format the full message history as a string.

        Returns
        -------
        str
            Formatted chat history with one line per message.
        """
        lines = []
        for entry in self._session.messages:
            ts = datetime.datetime.fromtimestamp(entry["timestamp"]).strftime("%H:%M:%S")
            lines.append(f"[{ts}] {entry['role']}: {entry['message']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Agent-facing actions
    # ------------------------------------------------------------------

    @tool_action
    def send_message(self, text: str) -> str:
        """Send a message to the task.

        Parameters
        ----------
        text : str
            Message content to send.
        """
        self._session.send_message(text)
        return self.chat_obs()

    @tool_action
    def report_infeasible(self, reason: str) -> str:
        """Report that the current task instructions are infeasible.

        Use this action when the task cannot be completed as described,
        for example due to missing information, contradictory instructions,
        or unavailable resources.

        Parameters
        ----------
        reason : str
            Explanation of why the task is infeasible.
        """
        self._session.add_message("infeasible", reason)
        return self.chat_obs()
