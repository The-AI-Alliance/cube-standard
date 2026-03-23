"""Unit tests for BasicChatConfig and BasicChatSession."""

import threading
import time

import pytest
from cube.resources.chat_session import ChatConfig, ChatSession

from cube_chat import BasicChatConfig, BasicChatSession


def test_chat_config_is_abstract() -> None:
    with pytest.raises(TypeError):
        ChatConfig()  # type: ignore[abstract]


def test_chat_session_is_abstract() -> None:
    with pytest.raises(TypeError):
        ChatSession()  # type: ignore[abstract]


def test_messages_starts_empty() -> None:
    session = BasicChatSession()
    assert session.messages == []


def test_messages_returns_copy() -> None:
    session = BasicChatSession()
    session.add_message("user", "hello")
    msgs = session.messages
    msgs.append({"role": "hack"})
    assert len(session.messages) == 1


def test_add_message_user_stored() -> None:
    session = BasicChatSession()
    session.add_message("user", "hello")
    msgs = session.messages
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert msgs[0]["message"] == "hello"
    assert "timestamp" in msgs[0]


def test_add_message_assistant_stored() -> None:
    session = BasicChatSession()
    session.add_message("assistant", "hi there")
    msgs = session.messages
    assert len(msgs) == 1
    assert msgs[0]["role"] == "assistant"
    assert msgs[0]["message"] == "hi there"


def test_add_message_infeasible_stored() -> None:
    session = BasicChatSession()
    session.add_message("infeasible", "cannot do this")
    msgs = session.messages
    assert len(msgs) == 1
    assert msgs[0]["role"] == "infeasible"


def test_add_message_info_not_stored() -> None:
    session = BasicChatSession()
    session.add_message("info", "this is background info")
    assert session.messages == []


def test_add_message_invalid_role_raises() -> None:
    session = BasicChatSession()
    with pytest.raises(ValueError, match="Invalid role"):
        session.add_message("system", "bad role")  # type: ignore[arg-type]


def test_send_message_appends_assistant_message() -> None:
    session = BasicChatSession()

    # Drain queue in background so send_message doesn't need a consumer
    def drain() -> None:
        session._queue.get()

    t = threading.Thread(target=drain, daemon=True)
    t.start()
    session.send_message("Paris")
    t.join(timeout=1)
    msgs = session.messages
    assert len(msgs) == 1
    assert msgs[0]["role"] == "assistant"
    assert msgs[0]["message"] == "Paris"


def test_send_message_unblocks_wait_for_user_message() -> None:
    session = BasicChatSession()
    result: list[str] = []

    def waiter() -> None:
        result.append(session.wait_for_user_message())

    t = threading.Thread(target=waiter, daemon=True)
    t.start()
    time.sleep(0.05)  # ensure waiter is blocked
    session.send_message("unblocked!")
    t.join(timeout=2)
    assert result == ["unblocked!"]


def test_wait_for_user_message_blocks_until_send() -> None:
    session = BasicChatSession()
    received: list[str] = []

    def waiter() -> None:
        received.append(session.wait_for_user_message())

    t = threading.Thread(target=waiter, daemon=True)
    t.start()
    assert not received  # nothing yet
    session.send_message("response")
    t.join(timeout=2)
    assert received == ["response"]


def test_stop_is_noop() -> None:
    session = BasicChatSession()
    session.add_message("user", "hello")
    session.stop()  # should not raise
    assert len(session.messages) == 1


def test_basic_chat_config_make() -> None:
    config = BasicChatConfig()
    session = config.make()
    assert isinstance(session, BasicChatSession)
    assert session.messages == []
