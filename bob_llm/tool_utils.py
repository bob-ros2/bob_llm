import sys
import inspect
from typing import Any, Dict, List, TypedDict

# Typing information for OpenAI-compatible tools
class FunctionParameters(TypedDict):
    type: str
    properties: Dict[str, Any]
    required: List[str]

class FunctionDefinition(TypedDict):
    name: str
    description: str
    parameters: FunctionParameters

class Tool(TypedDict):
    type: str
    function: FunctionDefinition

# Map Python types to JSON Schema types
TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}

def register(module: Any, node: Any = None) -> List[Tool]:
    """
    Inspects a module and builds a list of OpenAI-compatible tool definitions
    from its functions. It ignores functions starting with '_' and 'register'.
    """
    tools = []
    for name, func in inspect.getmembers(module, inspect.isfunction):
        if name.startswith('_') or name == 'register' or name == 'main':
            continue
        
        # Ignore functions not defined in the module itself
        if func.__module__ != module.__name__:
            continue

        sig = inspect.signature(func)
        docstring = inspect.getdoc(func) or "No description provided."
        
        # Extract the main description from the docstring (first line).
        description = docstring.strip().split('\n')[0]

        parameters = {
            "type": "object",
            "properties": {},
            "required": [],
        }

        for param_name, param in sig.parameters.items():
            if param.annotation is inspect.Parameter.empty:
                param_type = "string" # Default to string if no type hint
            else:
                param_type = TYPE_MAP.get(param.annotation, "string")

            parameters["properties"][param_name] = {"type": param_type}
            if param.default is inspect.Parameter.empty:
                parameters["required"].append(param_name)

        tool: Tool = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            }
        }
        tools.append(tool)

    return tools