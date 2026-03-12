"""Tests for cube.utils - function_to_dict and helpers."""

from typing import Optional

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
    assert props["e"]["type"] == "object"


def func_with_optional_str(x: str | None) -> None:
    """A function with an optional string.

    Parameters
    ----------
    x : str, optional
        An optional string.
    """


def test_union_type_annotation_resolves_to_base_type() -> None:
    result = function_to_dict(func_with_optional_str)
    props = result["parameters"]["properties"]
    assert props["x"]["type"] == "string"


def func_with_list_param(items: list) -> None:
    """A function with a list parameter.

    Parameters
    ----------
    items : list
        A list of items.
    """


def test_list_param_has_items_key() -> None:
    result = function_to_dict(func_with_list_param)
    props = result["parameters"]["properties"]
    assert props["items"]["type"] == "array"
    assert "items" in props["items"]
    assert props["items"]["items"] == {}


def func_with_optional_list(x: list | None) -> None:
    """A function with an optional list.

    Parameters
    ----------
    x : list, optional
        An optional list.
    """


def test_union_list_none_resolves_to_array_with_items() -> None:
    result = function_to_dict(func_with_optional_list)
    props = result["parameters"]["properties"]
    assert props["x"]["type"] == "array"
    assert props["x"].get("items") == {}


def func_with_optional_int(n: int | None) -> None:
    """A function with an optional int.

    Parameters
    ----------
    n : int, optional
        An optional integer.
    """


def test_union_int_none_resolves_to_integer() -> None:
    result = function_to_dict(func_with_optional_int)
    props = result["parameters"]["properties"]
    assert props["n"]["type"] == "integer"


def func_with_typing_optional(s: Optional[str]) -> None:
    """A function with typing.Optional.

    Parameters
    ----------
    s : str, optional
        An optional string via typing.Optional.
    """


def test_typing_optional_resolves_to_string() -> None:
    result = function_to_dict(func_with_typing_optional)
    props = result["parameters"]["properties"]
    assert props["s"]["type"] == "string"


def func_with_generic_list(tags: list[str]) -> None:
    """A function with a generic list.

    Parameters
    ----------
    tags : list
        A list of string tags.
    """


def test_generic_list_annotation_resolves_to_array() -> None:
    result = function_to_dict(func_with_generic_list)
    props = result["parameters"]["properties"]
    assert props["tags"]["type"] == "array"
