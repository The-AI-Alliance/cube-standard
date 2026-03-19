# ==================================================================
# These functions were copy/pasted from litellm.utils to avoid a dependency on litellm for cube
# - json_schema_type --> _json_schema_type
# - function_to_dict
# ==================================================================

import inspect
import types
import typing
from ast import literal_eval
from collections.abc import Callable

import docstring_parser


def _json_schema_type(python_type_name: str) -> str:
    """Converts standard python types to json schema types

    Parameters
    ----------
    python_type_name : str
        __name__ of type

    Returns
    -------
    str
        a standard JSON schema type, "string" if not recognized.
    """
    python_to_json_schema_types = {
        str.__name__: "string",
        int.__name__: "integer",
        float.__name__: "number",
        bool.__name__: "boolean",
        list.__name__: "array",
        dict.__name__: "object",
        "NoneType": "null",
    }

    return python_to_json_schema_types.get(python_type_name, "string")


def _get_type_name(annotation: type) -> str:
    """Returns the base type name for an annotation, handling generic aliases like list[str]."""
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    if hasattr(annotation, "__origin__"):
        return annotation.__origin__.__name__
    return str(annotation)


def function_to_dict(input_function: Callable) -> dict:  # noqa: C901
    """Using type hints and numpy- or Google-style docstring,
    produce a dictionary usable for OpenAI function calling

    Parameters
    ----------
    input_function : function
        A function with a numpy- or Google-style docstring

    Returns
    -------
    dictionnary
        A dictionnary to add to the list passed to `functions` parameter of `litellm.completion`
    """
    # Get function name and docstring
    name = input_function.__name__
    docstring = inspect.getdoc(input_function)
    if not docstring:
        raise ValueError(
            f"Function '{name}' has no docstring. A docstring is required to extract parameter information."
        )
    parsed = docstring_parser.parse(docstring)
    description = parsed.short_description or ""

    # Get function parameters and their types from annotations and docstring
    parameters = {}
    required_params = []
    param_info = inspect.signature(input_function).parameters

    for param_name, param in param_info.items():
        if hasattr(param, "annotation") and param.annotation is not inspect.Parameter.empty:
            annotation = param.annotation
            is_union = (
                isinstance(annotation, types.UnionType) or getattr(annotation, "__origin__", None) is typing.Union
            )
            if is_union:
                non_none = [a for a in annotation.__args__ if a is not type(None)]
                if len(non_none) != 1:
                    raise ValueError(f"Unsupported union type {annotation}: only 'X | None' is supported.")
                if param.default is inspect.Parameter.empty:
                    raise ValueError(
                        f"Parameter '{param_name}' has type '{annotation}' but has no default value. 'X | None' parameters must default to None."
                    )
                if param.default is not None:
                    raise ValueError(
                        f"Parameter '{param_name}' has type '{annotation}' but default is {param.default!r}, not None."
                    )
                annotation = non_none[0]
            param_type = _json_schema_type(_get_type_name(annotation))
        else:
            param_type = None
        param_description = None
        param_enum = None

        # Try to extract param description from docstring
        for param_data in parsed.params:
            if param_data.arg_name == param_name:
                if param_data.type_name:
                    # replace type from docstring rather than annotation
                    param_type = param_data.type_name
                    if "optional" in param_type:
                        param_type = param_type.split(",")[0]
                    elif "{" in param_type:
                        # may represent a set of acceptable values
                        # translating as enum for function calling
                        try:
                            param_enum = list(literal_eval(param_type))
                            param_type = "string"
                        except Exception:
                            pass
                    param_type = _json_schema_type(param_type)
                param_description = param_data.description

        param_dict = {
            "type": param_type,
            "description": param_description,
            "enum": param_enum,
        }

        param_schema = {k: v for k, v in param_dict.items() if v is not None}
        if param_type == "array":
            param_schema["items"] = {}
        parameters[param_name] = param_schema

        # Check if the parameter has no default value (i.e., it's required)
        if param.default == param.empty:
            required_params.append(param_name)

    # Create the dictionary
    result = {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": parameters,
        },
    }

    # Add "required" key if there are required parameters
    if required_params:
        result["parameters"]["required"] = required_params

    return result
