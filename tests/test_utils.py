"""Tests for cube.utils - function_to_dict and helpers."""

import typing

import pytest

from cube.utils import _json_schema_type, function_to_dict

# --- _json_schema_type ---


def test_json_schema_type_known_types():
    assert _json_schema_type("str") == "string"
    assert _json_schema_type("int") == "integer"
    assert _json_schema_type("float") == "number"
    assert _json_schema_type("bool") == "boolean"
    assert _json_schema_type("list") == "array"
    assert _json_schema_type("dict") == "object"
    assert _json_schema_type("NoneType") == "null"


def test_json_schema_type_unknown_falls_back_to_string():
    assert _json_schema_type("MyCustomClass") == "string"
    assert _json_schema_type("") == "string"
    assert _json_schema_type("bytes") == "string"


# --- function_to_dict ---


def simple_func(x: int, y: str) -> None:
    """Add x to y.

    Parameters
    ----------
    x : int
        The integer value.
    y : str
        The string value.
    """


def test_function_name():
    result = function_to_dict(simple_func)
    assert result["name"] == "simple_func"


def test_function_description():
    result = function_to_dict(simple_func)
    assert result["description"] == "Add x to y."


def test_function_parameters_structure():
    result = function_to_dict(simple_func)
    params = result["parameters"]
    assert params["type"] == "object"
    assert "properties" in params


def test_function_parameter_types_from_annotations():
    result = function_to_dict(simple_func)
    props = result["parameters"]["properties"]
    assert props["x"]["type"] == "integer"
    assert props["y"]["type"] == "string"


def test_function_parameter_descriptions():
    result = function_to_dict(simple_func)
    props = result["parameters"]["properties"]
    assert props["x"]["description"] == "The integer value."
    assert props["y"]["description"] == "The string value."


def test_all_required_when_no_defaults():
    result = function_to_dict(simple_func)
    assert result["parameters"]["required"] == ["x", "y"]


def func_with_defaults(a: int, b: str = "hello") -> None:
    """Do something.

    Parameters
    ----------
    a : int
        Required param.
    b : str, optional
        Optional param.
    """


def test_optional_param_not_in_required():
    result = function_to_dict(func_with_defaults)
    required = result["parameters"].get("required", [])
    assert "a" in required
    assert "b" not in required


def test_optional_type_stripped_from_optional_keyword():
    result = function_to_dict(func_with_defaults)
    props = result["parameters"]["properties"]
    # "str, optional" in docstring → type should be "string", not contain "optional"
    assert props["b"]["type"] == "string"


def func_no_params() -> None:
    """A function with no parameters."""


def test_no_parameters():
    result = function_to_dict(func_no_params)
    assert result["parameters"]["properties"] == {}
    assert "required" not in result["parameters"]


def func_with_enum(color: str) -> None:
    """Pick a color.

    Parameters
    ----------
    color : {'red', 'green', 'blue'}
        The color to pick.
    """


def test_enum_param_type_is_string():
    result = function_to_dict(func_with_enum)
    props = result["parameters"]["properties"]
    assert props["color"]["type"] == "string"


def test_enum_param_has_enum_key():
    result = function_to_dict(func_with_enum)
    props = result["parameters"]["properties"]
    assert "enum" in props["color"]


def test_enum_param_is_list_not_string() -> None:
    result = function_to_dict(func_with_enum)
    result["parameters"]["properties"]["color"]["enum"].sort()
    assert result == {
        "name": "func_with_enum",
        "description": "Pick a color.",
        "parameters": {
            "type": "object",
            "properties": {
                "color": {"type": "string", "description": "The color to pick.", "enum": ["blue", "green", "red"]},
            },
            "required": ["color"],
        },
    }


def func_no_annotation(x, y) -> None:
    """A function without type annotations.

    Parameters
    ----------
    x : str
        First param.
    y : int
        Second param.
    """


def test_type_from_docstring_when_no_annotation():
    result = function_to_dict(func_no_annotation)
    props = result["parameters"]["properties"]
    assert props["x"]["type"] == "string"
    assert props["y"]["type"] == "integer"


def test_output_keys():
    result = function_to_dict(simple_func)
    assert set(result.keys()) == {"name", "description", "parameters"}


def test_multiline_description():
    def multi() -> None:
        """First line.
        Second line.

        Parameters
        ----------
        """

    result = function_to_dict(multi)
    assert "First line" in result["description"]


def func_all_types(a: int, b: float, c: bool, d: list, e: dict) -> None:
    """Test all types.

    Parameters
    ----------
    a : int
        Integer.
    b : float
        Float.
    c : bool
        Boolean.
    d : list
        List.
    e : dict
        Dict.
    """


def test_all_type_annotations():
    result = function_to_dict(func_all_types)
    props = result["parameters"]["properties"]
    assert props["a"]["type"] == "integer"
    assert props["b"]["type"] == "number"
    assert props["c"]["type"] == "boolean"
    assert props["d"]["type"] == "array"
    assert "items" in props["d"]
    assert props["e"]["type"] == "object"


def func_with_optional_str_default(x: str | None = None) -> None:
    """A function with an optional string defaulting to None.

    Parameters
    ----------
    x : str, optional
        An optional string.
    """


def test_union_x_or_none_with_none_default() -> None:
    assert function_to_dict(func_with_optional_str_default) == {
        "name": "func_with_optional_str_default",
        "description": "A function with an optional string defaulting to None.",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "string", "description": "An optional string."},
            },
        },
    }


def func_with_generic_list(tags: list[str]) -> None:
    """A function with a generic list.

    Parameters
    ----------
    tags : list
        A list of string tags.
    """


def test_generic_list_annotation_resolves_to_array() -> None:
    assert function_to_dict(func_with_generic_list) == {
        "name": "func_with_generic_list",
        "description": "A function with a generic list.",
        "parameters": {
            "type": "object",
            "properties": {
                "tags": {"type": "array", "description": "A list of string tags.", "items": {}},
            },
            "required": ["tags"],
        },
    }


def func_with_optional_str_no_default(x: str | None) -> None:
    """A function with an optional string but no default.

    Parameters
    ----------
    x : str, optional
        An optional string.
    """


def test_union_x_or_none_without_default_raises() -> None:
    with pytest.raises(ValueError, match="no default value"):
        function_to_dict(func_with_optional_str_no_default)


def func_with_optional_str_non_none_default(x: str | None = "hello") -> None:
    """A function with an optional string defaulting to a non-None value.

    Parameters
    ----------
    x : str, optional
        An optional string.
    """


def test_union_x_or_none_with_non_none_default_raises() -> None:
    with pytest.raises(ValueError, match="not None"):
        function_to_dict(func_with_optional_str_non_none_default)


def func_with_optional_list(tags: list[str] | None = None) -> None:
    """A function with an optional list of tags.

    Parameters
    ----------
    tags : list
        A list of string tags.
    """


def test_union_with_generic_inner_type() -> None:
    assert function_to_dict(func_with_optional_list) == {
        "name": "func_with_optional_list",
        "description": "A function with an optional list of tags.",
        "parameters": {
            "type": "object",
            "properties": {
                "tags": {"type": "array", "description": "A list of string tags.", "items": {}},
            },
        },
    }


def func_with_typing_optional(x: typing.Optional[str] = None) -> None:
    """A function using typing.Optional.

    Parameters
    ----------
    x : str, optional
        An optional string.
    """


def test_typing_optional_resolves_to_string() -> None:
    assert function_to_dict(func_with_typing_optional) == {
        "name": "func_with_typing_optional",
        "description": "A function using typing.Optional.",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "string", "description": "An optional string."},
            },
        },
    }


def func_with_none_first(x: None | str = None) -> None:
    """A function with None listed first in the union.

    Parameters
    ----------
    x : str, optional
        An optional string.
    """


def test_union_none_first_resolves_correctly() -> None:
    assert function_to_dict(func_with_none_first) == {
        "name": "func_with_none_first",
        "description": "A function with None listed first in the union.",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "string", "description": "An optional string."},
            },
        },
    }


def func_with_multi_union(x: str | int | None = None) -> None:
    """A function with multiple non-None types in union.

    Parameters
    ----------
    x : str
        A value.
    """


def test_union_with_multiple_non_none_types_raises() -> None:
    with pytest.raises(ValueError, match="only 'X | None' is supported"):
        function_to_dict(func_with_multi_union)


# --- Google-style docstrings ---


def google_func(x: int, y: str) -> None:
    """Add x to y.

    Args:
        x: The integer value.
        y: The string value.
    """


def test_function_to_dict_google_description():
    result = function_to_dict(google_func)
    assert result["description"] == "Add x to y."


def test_function_to_dict_google_params():
    result = function_to_dict(google_func)
    props = result["parameters"]["properties"]
    assert props["x"]["description"] == "The integer value."
    assert props["y"]["description"] == "The string value."
    assert props["x"]["type"] == "integer"
    assert props["y"]["type"] == "string"
