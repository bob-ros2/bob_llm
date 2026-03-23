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

"""Tool interface for Qdrant vector database using native qdrant-client."""

import json
import os
from typing import Any, List

from bob_llm.tool_utils import register as default_register
from bob_llm.tool_utils import Tool
from qdrant_client import QdrantClient
from qdrant_client.http import models
import rclpy


class _NodeContext:
    """Private class to hold the ROS node instance and Qdrant client."""

    node: rclpy.node.Node = None
    client: QdrantClient = None
    collection_name: str = 'bob_memory'


def register(module: Any, node: Any = None) -> List[Tool]:
    """
    Inspect a module and build a list of tool definitions.

    :param module: The module to inspect.
    :param node: The ROS node instance.
    :return: A list of tool definitions.
    """
    _NodeContext.node = node

    # Configuration via environment variables
    location = os.environ.get('LLM_QDRANT_LOCATION', ':memory:')
    api_key = os.environ.get('LLM_QDRANT_API_KEY', '')
    _NodeContext.collection_name = os.environ.get('LLM_QDRANT_COLLECTION', 'bob_memory')

    # Initialize Qdrant Client
    if location == ':memory:':
        _NodeContext.client = QdrantClient(location)
    elif location.startswith(('http://', 'https://')):
        _NodeContext.client = QdrantClient(url=location, api_key=api_key)
    else:
        # Assume it's a local path
        _NodeContext.client = QdrantClient(path=location)

    if node:
        node.get_logger().info(f'Qdrant initialized at: {location}')

    # Ensure collection exists
    if not _NodeContext.client.collection_exists(_NodeContext.collection_name):
        _NodeContext.client.create_collection(
            collection_name=_NodeContext.collection_name,
            vectors_config=models.VectorParams(
                size=1536,  # Default for many models (OpenAI compatible)
                distance=models.Distance.COSINE
            )
        )
    return default_register(module, node)


# --- Exposed Tools for bob_llm ---

def save_memory(information: str, metadata: str = '{}') -> str:
    """
    Store information (memory) into the Qdrant vector database.

    :param information: The text content to store/memorize.
    :param metadata: A JSON string of metadata to associate with this memory.
    """
    if not _NodeContext.client:
        return 'Error: Qdrant client not initialized.'

    try:
        meta_dict = json.loads(metadata)
    except json.JSONDecodeError:
        meta_dict = {'raw_metadata': metadata}

    try:
        _NodeContext.client.add(
            collection_name=_NodeContext.collection_name,
            documents=[information],
            metadata=[meta_dict]
        )
        return f'Successfully stored memory in collection {_NodeContext.collection_name}.'
    except Exception as e:
        return f'Error saving to Qdrant: {e}'


def search_memory(query: str, limit: int = 5) -> str:
    """
    Search for relevant memories in the Qdrant vector database.

    :param query: The question or topic to search for in the memory.
    :param limit: Number of results to return.
    """
    if not _NodeContext.client:
        return 'Error: Qdrant client not initialized.'

    try:
        results = _NodeContext.client.query(
            collection_name=_NodeContext.collection_name,
            query_text=query,
            limit=limit
        )

        if not results:
            return 'No relevant memories found.'

        output = ['Found the following relevant memories:']
        for res in results:
            output.append(f'- {res.document} (Metadata: {res.metadata})')

        return '\n'.join(output)
    except Exception as e:
        return f'Error searching Qdrant: {e}'
