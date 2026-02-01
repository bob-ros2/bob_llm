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

import asyncio
import json
import os
import sys
from typing import Any, Dict

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# --- Configuration ---

# Define how to run the MCP server.
# Use sys.executable to ensure we use the same python interpreter.
python_executable = sys.executable

server_env = os.environ.copy()
# Provide a default collection name in case it's not set.
# Note: mcp-server-qdrant uses COLLECTION_NAME
server_env.setdefault('COLLECTION_NAME', 'llm_memory')

SERVER_PARAMS = StdioServerParameters(
    command='mcp-server-qdrant',
    args=[],
    env=server_env
)


# --- Internal Helpers ---

async def _run_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Start the Qdrant MCP server and execute the tool."""
    # Start the server process and connect
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Call the tool
            result = await session.call_tool(tool_name, arguments=arguments)

            # Parse and return the content
            output_text = []
            for content in result.content:
                if content.type == 'text':
                    output_text.append(content.text)

            return '\n'.join(output_text)


def _sync_wrapper(coro):
    """Run async tools in a sync context if needed."""
    try:
        return asyncio.run(coro)
    except RuntimeError:
        # If an event loop is already running
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)


# --- Exposed Tools for bob_llm ---

def save_memory(information: str, metadata: str = '{}') -> str:
    """
    Store information (memory) into the Qdrant vector database.

    :param information: The text content to store/memorize.
    :param metadata: A JSON string of metadata to associate with this memory.
    """
    # Parse metadata string to dict if necessary
    try:
        meta_dict = json.loads(metadata)
    except json.JSONDecodeError:
        meta_dict = {'raw_metadata': metadata}

    # The tool name in mcp-server-qdrant is typically 'qdrant-store'
    tool_name = 'qdrant-store'

    arguments = {
        'information': information,
        'metadata': meta_dict
    }

    return _sync_wrapper(_run_mcp_tool(tool_name, arguments))


def search_memory(query: str) -> str:
    """
    Search for relevant memories in the Qdrant vector database.

    :param query: The question or topic to search for in the memory.
    """
    tool_name = 'qdrant-find'

    arguments = {
        'query': query
    }

    return _sync_wrapper(_run_mcp_tool(tool_name, arguments))
