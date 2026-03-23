"""Unit tests for ChatToolConfig and ChatTool."""

from unittest.mock import MagicMock

from cube.core import Action, Observation, StepError
from cube_chat import BasicChatConfig
from cube_chat_tool import ChatTool, ChatToolConfig


def _make_tool() -> tuple[ChatTool, MagicMock]:
    """Create a ChatTool with a mock session."""
    mock_session = MagicMock()
    mock_session.messages = []
    config = ChatToolConfig()
    tool = ChatTool(config=config, session=mock_session)
    return tool, mock_session


def test_config_serialization_round_trip() -> None:
    config = ChatToolConfig()
    data = config.model_dump()
    restored = ChatToolConfig.model_validate(data)
    # Both round-trips should produce a valid ChatToolConfig with the same chat type
    assert type(restored) is ChatToolConfig
    assert type(restored.chat) is BasicChatConfig


def test_action_set_contains_only_send_message() -> None:
    tool, _ = _make_tool()
    names = {a.name for a in tool.action_set}
    assert names == {"send_message"}


def test_execute_action_dispatches_send_message() -> None:
    tool, mock_session = _make_tool()
    mock_session.messages = [{"role": "user", "timestamp": 0.0, "message": "hello"}]
    action = Action(name="send_message", arguments={"text": "Paris"})
    result = tool.execute_action(action)
    mock_session.send_message.assert_called_once_with("Paris")
    assert isinstance(result, Observation)


def test_execute_action_appends_chat_obs() -> None:
    tool, mock_session = _make_tool()
    mock_session.messages = [{"role": "user", "timestamp": 0.0, "message": "Q"}]
    action = Action(name="send_message", arguments={"text": "A"})
    result = tool.execute_action(action)
    assert isinstance(result, Observation)
    # The observation should contain a chat_history content
    names = {c.name for c in result.contents if hasattr(c, "name")}
    assert "chat_history" in names


def test_execute_action_returns_step_error_on_session_error() -> None:
    tool, mock_session = _make_tool()
    mock_session.send_message.side_effect = RuntimeError("connection lost")
    action = Action(name="send_message", arguments={"text": "hi"})
    result = tool.execute_action(action)
    assert isinstance(result, StepError)


def test_add_message_delegates_to_session() -> None:
    tool, mock_session = _make_tool()
    tool.add_message("user", "hello")
    mock_session.add_message.assert_called_once_with("user", "hello")


def test_wait_for_user_message_delegates_to_session() -> None:
    tool, mock_session = _make_tool()
    mock_session.wait_for_user_message.return_value = "agent reply"
    result = tool.wait_for_user_message()
    assert result == "agent reply"
    mock_session.wait_for_user_message.assert_called_once()


def test_chat_obs_formats_messages() -> None:
    tool, mock_session = _make_tool()
    import time

    ts = time.time()
    mock_session.messages = [
        {"role": "user", "timestamp": ts, "message": "What is 2+2?"},
        {"role": "assistant", "timestamp": ts, "message": "4"},
    ]
    obs = tool.chat_obs()
    assert isinstance(obs, Observation)
    text = obs.contents[0].data
    assert "user: What is 2+2?" in text
    assert "assistant: 4" in text


def test_chat_obs_empty_history() -> None:
    tool, mock_session = _make_tool()
    mock_session.messages = []
    obs = tool.chat_obs()
    assert isinstance(obs, Observation)
    assert obs.contents[0].data == ""


def test_close_calls_session_stop() -> None:
    tool, mock_session = _make_tool()
    tool.close()
    mock_session.stop.assert_called_once()


def test_reset_stops_old_session_and_creates_new() -> None:
    tool, mock_session = _make_tool()
    new_session = MagicMock()
    new_session.messages = []
    mock_config = MagicMock()
    mock_config.chat.make.return_value = new_session
    tool.config = mock_config
    tool.reset()
    mock_session.stop.assert_called_once()
    assert tool.session is new_session


def test_session_property_exposes_raw_session() -> None:
    tool, mock_session = _make_tool()
    assert tool.session is mock_session


def test_messages_property_delegates_to_session() -> None:
    tool, mock_session = _make_tool()
    mock_session.messages = [{"role": "user", "timestamp": 0.0, "message": "hi"}]
    assert tool.messages is mock_session.messages
