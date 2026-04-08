"""Tests for cube.core - core data models."""

import json

import pytest
from PIL import Image as PILImage
from pydantic import BaseModel

from cube.core import (
    Action,
    ActionSchema,
    Content,
    ImageContent,
    Observation,
    StepError,
    StructuredContent,
    TextContent,
)

# --- TypedBaseModel (via Action) ---


def test_typed_base_model_round_trip():
    """Serialization adds _type; deserialization restores the correct subclass."""
    original = Action(name="click", arguments={"selector": "#btn"})
    data = original.model_dump()
    assert data == {"_type": "cube.core.Action", "id": None, "name": "click", "arguments": {"selector": "#btn"}}
    restored = Action.model_validate(data)
    assert restored == original


# --- ActionSchema ---


def test_action_schema_from_function():
    """ActionSchema.from_function extracts name, description, and parameters."""

    def greet(name: str) -> str:
        """Say hello to someone."""
        ...

    schema = ActionSchema.from_function(greet)
    assert schema.name == "greet"
    assert schema.description == "Say hello to someone."
    assert schema.parameters == {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }


def test_action_schema_as_dict_llm_format():
    """as_dict produces a well-formed LLM tool descriptor."""
    schema = ActionSchema(name="click", description="Click element", parameters={})
    assert schema.as_dict() == {
        "type": "function",
        "function": {"name": "click", "description": "Click element", "parameters": {}},
    }


# --- Action.from_openai_tool_call ---


def test_action_from_openai_tool_call_flat_format():
    """Parse flat/Responses API format."""
    action = Action.from_openai_tool_call({"id": "call_1", "name": "click", "arguments": '{"x": 10, "y": 20}'})
    assert action.id == "call_1"
    assert action.name == "click"
    assert action.arguments == {"x": 10, "y": 20}


def test_action_from_openai_tool_call_chat_completions_format():
    """Parse nested Chat Completions format."""
    action = Action.from_openai_tool_call(
        {
            "id": "call_2",
            "type": "function",
            "function": {"name": "type_text", "arguments": '{"text": "hello"}'},
        }
    )
    assert action.id == "call_2"
    assert action.name == "type_text"
    assert action.arguments == {"text": "hello"}


def test_action_from_openai_tool_call_dict_arguments():
    """Arguments can be a dict (already parsed) instead of a JSON string."""
    action = Action.from_openai_tool_call({"id": "call_3", "name": "greet", "arguments": {"name": "World"}})
    assert action.arguments == {"name": "World"}


def test_action_from_openai_tool_call_call_id():
    """Responses API uses call_id instead of id."""
    action = Action.from_openai_tool_call({"call_id": "resp_call_1", "name": "click", "arguments": "{}"})
    assert action.id == "resp_call_1"


# --- Content.from_data dispatch ---


def test_content_from_data_str_gives_text_content():
    assert isinstance(Content.from_data("hello"), TextContent)


def test_content_from_data_dict_gives_structured_content():
    assert isinstance(Content.from_data({"key": "val"}), StructuredContent)


def test_content_from_data_image_gives_image_content():
    assert isinstance(Content.from_data(PILImage.new("RGB", (10, 10))), ImageContent)


def test_content_from_data_base_model_gives_structured_content():
    class Point(BaseModel):
        x: int
        y: int

    assert isinstance(Content.from_data(Point(x=1, y=2)), StructuredContent)


def test_content_from_data_bytes_raises_type_error():
    with pytest.raises(TypeError, match="bytes"):
        Content.from_data(b"raw bytes")


# --- TextContent rendering ---


def test_text_content_to_markdown():
    assert TextContent(data="hello").to_markdown() == "hello"
    assert TextContent(data="hello", name="greeting").to_markdown() == "## greeting\nhello"


def test_text_content_to_llm_message():
    assert TextContent(data="hello").to_llm_message() == {"role": "user", "content": "hello"}
    assert TextContent(data="hello", tool_call_id="call-1").to_llm_message() == {
        "role": "tool",
        "content": "hello",
        "tool_call_id": "call-1",
    }


# --- StructuredContent rendering ---


def test_structured_content_to_markdown_json_block():
    expected = f"```json\n{json.dumps({'score': 42}, indent=2)}\n```"
    assert StructuredContent(data={"score": 42}).to_markdown() == expected


def test_structured_content_coerces_base_model_to_dict():
    class Point(BaseModel):
        x: int
        y: int

    content = StructuredContent(data=Point(x=3, y=4))
    assert content.data == {"x": 3, "y": 4}


def test_structured_content_to_llm_message():
    assert StructuredContent(data={"score": 42}).to_llm_message() == {
        "role": "user",
        "content": json.dumps({"score": 42}),
    }


# --- ImageContent round-trip ---


def test_image_content_round_trip():
    """PIL Image serializes to base64 data URL and deserializes back."""
    img = PILImage.new("RGB", (20, 20), color=(0, 128, 255))
    serialized = ImageContent(data=img).model_dump()
    assert serialized["data"][: len(ImageContent._image_prefix)] == ImageContent._image_prefix
    restored = ImageContent.model_validate(serialized)
    assert isinstance(restored.data, PILImage.Image)
    assert restored.data.size == (20, 20)


# --- Observation ---


def test_observation_from_text_and_add():
    obs = Observation.from_text("hello")
    assert obs.contents == [TextContent(data="hello")]
    combined = obs + Observation.from_text("world")
    assert combined.contents == [TextContent(data="hello"), TextContent(data="world")]


def test_observation_round_trip_with_image():
    img = PILImage.new("RGB", (10, 10), color=(0, 0, 255))
    obs = Observation(contents=[ImageContent(data=img)])
    data = obs.model_dump_json()
    assert ImageContent._image_prefix in data
    restored = Observation.model_validate_json(data)
    assert isinstance(restored.contents[0], ImageContent)
    assert restored.contents[0].data.size == img.size
    assert restored.contents[0].data.getpixel((0, 0)) == (0, 0, 255)


# --- StepError ---


def test_step_error_from_exception():
    try:
        raise ValueError("something went wrong")
    except ValueError as e:
        err = StepError.from_exception(e)
    assert err.error_type == "ValueError"
    assert err.exception_str == "something went wrong"
