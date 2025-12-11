# qdrant_tools.py
import os
import sys
import asyncio
import json
from typing import Optional, Dict, Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# --- Configuration ---

# Define how to run the MCP server.
# Use sys.executable to ensure we use the same python interpreter as the parent process.
python_executable = sys.executable

server_env = os.environ.copy()
# Provide a default collection name in case it's not set.
# Note: mcp-server-qdrant uses COLLECTION_NAME (not QDRANT_COLLECTION_NAME)
server_env.setdefault('COLLECTION_NAME', 'llm_memory') 

SERVER_PARAMS = StdioServerParameters(
    command="mcp-server-qdrant",
    args=[], # The command is the executable, no extra python args needed
    env=server_env  # Pass current env vars (QDRANT_URL, etc.) to the server
)

# --- Internal Helpers ---

async def _run_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    """
    Starts the Qdrant MCP server, connects via stdio, executes the tool, 
    and returns the result text.
    """
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
                if content.type == "text":
                    output_text.append(content.text)
            
            return "\n".join(output_text)

def _sync_wrapper(coro):
    """Helper to run async tools in a sync context if needed."""
    try:
        return asyncio.run(coro)
    except RuntimeError:
        # If an event loop is already running (e.g. inside a jupyter notebook or existing loop)
        # you might need to use nest_asyncio or simply await it if the caller is async.
        # Assuming bob_llm handles async execution or is a standard script:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)

# --- Exposed Tools for bob_llm ---

def save_memory(information: str, metadata: str = "{}") -> str:
    """
    Stores information (memory) into the Qdrant vector database.
    
    Args:
        information: The text content to store/memorize.
        metadata: A JSON string of metadata to associate with this memory (optional).
    """
    # Parse metadata string to dict if necessary, usually the LLM passes a string
    try:
        meta_dict = json.loads(metadata)
    except json.JSONDecodeError:
        meta_dict = {"raw_metadata": metadata}

    # The tool name in mcp-server-qdrant is typically 'qdrant-store' 
    # (or 'qdrant-store-memory' depending on version). 
    # We use the most common identifier.
    tool_name = "qdrant-store" 
    
    arguments = {
        "information": information,
        "metadata": meta_dict
    }
    
    return _sync_wrapper(_run_mcp_tool(tool_name, arguments))

def search_memory(query: str) -> str:
    """
    Searches for relevant memories in the Qdrant vector database using semantic search.
    
    Args:
        query: The question or topic to search for in the memory.
    """
    tool_name = "qdrant-find"
    
    arguments = {
        "query": query
    }
    
    return _sync_wrapper(_run_mcp_tool(tool_name, arguments))