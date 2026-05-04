# Copyright 2026 Bob Ros
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Any, Dict, List, TypedDict


# Typing information for OpenAI-compatible tools
class FunctionParameters(TypedDict):
    """Parameters for a function definition."""

    type: str  # noqa: A003
    properties: Dict[str, Any]
    required: List[str]


class FunctionDefinition(TypedDict):
    """Definition of a function for an LLM tool."""

    name: str
    description: str
    parameters: FunctionParameters


class Tool(TypedDict):
    """OpenAI-compatible tool definition."""

    type: str  # noqa: A003
    function: FunctionDefinition


# Map Python types to JSON Schema types
TYPE_MAP = {
    str: 'string',
    int: 'integer',
    float: 'number',
    bool: 'boolean',
}


def register(module: Any, node: Any = None) -> List[Tool]:
    """
    Inspect a module and build a list of tool definitions.

    It ignores functions starting with '_' and 'register'.
    """
    tools = []
    for name, func in inspect.getmembers(module, inspect.isfunction):
        if name.startswith('_') or name == 'register' or name == 'main':
            continue

        # Ignore functions not defined in the module itself
        if func.__module__ != module.__name__:
            continue

        sig = inspect.signature(func)
        docstring = inspect.getdoc(func) or 'No description provided.'

        # Extract the main description from the docstring (first line).
        description = docstring.strip().split('\n')[0]

        parameters = {
            'type': 'object',
            'properties': {},
            'required': [],
        }

        for param_name, param in sig.parameters.items():
            if param.annotation is inspect.Parameter.empty:
                param_type = 'string'  # Default to string if no type hint
            else:
                param_type = TYPE_MAP.get(param.annotation, 'string')

            parameters['properties'][param_name] = {'type': param_type}
            if param.default is inspect.Parameter.empty:
                parameters['required'].append(param_name)

        tool: Tool = {
            'type': 'function',
            'function': {
                'name': name,
                'description': description,
                'parameters': parameters,
            }
        }
        tools.append(tool)

    return tools
