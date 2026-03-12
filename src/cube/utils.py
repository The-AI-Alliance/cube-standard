# ==================================================================
# These functions were copy/pasted from litellm.utils to avoid a dependency on litellm for cube
# - json_schema_type --> _json_schema_type
# - function_to_dict
# ==================================================================

import inspect
from ast import literal_eval

from numpydoc.docscrape import NumpyDocString


def _json_schema_type(python_type_name: str):
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


def function_to_dict(input_function) -> dict:  # noqa: C901
    """Using type hints and numpy-styled docstring,
    produce a dictionary usable for OpenAI function calling

    Parameters
    ----------
    input_function : function
        A function with a numpy-style docstring

    Returns
    -------
    dictionnary
        A dictionnary to add to the list passed to `functions` parameter of `litellm.completion`
    """
    # Get function name and docstring
    name = input_function.__name__
    docstring = inspect.getdoc(input_function)
    numpydoc = NumpyDocString(docstring)
    description = "\n".join([s.strip() for s in numpydoc["Summary"]])

    # Get function parameters and their types from annotations and docstring
    parameters = {}
    required_params = []
    param_info = inspect.signature(input_function).parameters

    for param_name, param in param_info.items():
        annotation = param.annotation
        if annotation is inspect.Parameter.empty:
            param_type = None
        elif isinstance(annotation, str):
            param_type = _json_schema_type(annotation)
        elif hasattr(annotation, "__name__"):
            param_type = _json_schema_type(annotation.__name__)
        else:
            # Handle Union types (e.g., list | None, str | None)
            args = getattr(annotation, "__args__", None)
            if args:
                first = next((a for a in args if a is not type(None)), None)
                param_type = _json_schema_type(first.__name__) if first and hasattr(first, "__name__") else None
            else:
                param_type = None
        param_description = None
        param_enum = None

        # Try to extract param description from docstring using numpydoc
        for param_data in numpydoc["Parameters"]:
            if param_data.name == param_name:
                if hasattr(param_data, "type"):
                    # replace type from docstring rather than annotation
                    param_type = param_data.type
                    if "optional" in param_type:
                        param_type = param_type.split(",")[0]
                    elif "{" in param_type:
                        # may represent a set of acceptable values
                        # translating as enum for function calling
                        try:
                            param_enum = str(list(literal_eval(param_type)))
                            param_type = "string"
                        except Exception:
                            pass
                    param_type = _json_schema_type(param_type)
                param_description = "\n".join([s.strip() for s in param_data.desc])

        param_dict = {
            "type": param_type,
            "description": param_description,
            "enum": param_enum,
        }

        param_entry = {k: v for k, v in param_dict.items() if isinstance(v, str)}
        if param_type == "array":
            param_entry["items"] = {}
        parameters[param_name] = param_entry

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
