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

"""Unit tests for LLMNode execution statistics and telemetry."""

import json
from unittest.mock import MagicMock, patch

from bob_llm.llm_node import LLMNode

import pytest
import rclpy


@pytest.fixture
def ros_init():
    """Ensure rclpy is initialized for the duration of the test."""
    if not rclpy.ok():
        rclpy.init()
    yield
    # Note: we do not shutdown here as other tests might share the context


def test_token_counting(ros_init):
    """Test that token counting functions produce reasonable estimates."""
    with patch('bob_llm.llm_node.LLMNode.add_on_set_parameters_callback'):
        node = LLMNode()

    # If tiktoken loaded, it uses encoding. Fallback is max(1, len//4).
    res = node._count_tokens('hello')
    assert res >= 1

    res_long = node._count_tokens('hello world')
    assert res_long >= 1

    history = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Hello, how are you?'}
    ]
    estimated = node._count_history_tokens(history)
    assert estimated > 0
    node.destroy_node()


def test_stats_formatting_and_publishing(ros_init):
    """Test JSON payload structure of execution stats."""
    with patch('bob_llm.llm_node.LLMNode.add_on_set_parameters_callback'):
        node = LLMNode()

    mock_pub = MagicMock()
    node.pub_stats = mock_pub

    # Override parameters correctly in ROS 2
    params = [
        rclpy.parameter.Parameter(
            'model_context_limit', rclpy.Parameter.Type.INTEGER, 8192
        ),
        rclpy.parameter.Parameter(
            'max_tokens', rclpy.Parameter.Type.INTEGER, 512
        )
    ]
    node.set_parameters(params)

    # Publish stats
    node._publish_stats(
        prompt_tokens=1000,
        completion_tokens=50,
        tokens_per_second=25.5,
        status='generating'
    )

    assert mock_pub.publish.called
    called_msg = mock_pub.publish.call_args[0][0]
    assert called_msg.__class__.__name__ == 'String'

    data = json.loads(called_msg.data)
    assert data['prompt_tokens'] == 1000
    assert data['context_limit'] == 8192
    assert data['context_percent'] == 12  # round(1000/8192 * 100) = 12
    assert data['completion_tokens'] == 50
    assert data['max_tokens'] == 512
    assert data['tokens_per_second'] == 25.5
    assert data['status'] == 'generating'

    # Check formatted string output
    assert 'Context: 1000/8192' in data['formatted']
    assert 'Output: 50/512' in data['formatted']
    assert '25.5 t/s' in data['formatted']

    node.destroy_node()


def test_stats_infinite_max_tokens(ros_init):
    """Test format output when max_tokens parameter is 0 (infinite)."""
    with patch('bob_llm.llm_node.LLMNode.add_on_set_parameters_callback'):
        node = LLMNode()

    mock_pub = MagicMock()
    node.pub_stats = mock_pub

    params = [
        rclpy.parameter.Parameter(
            'model_context_limit', rclpy.Parameter.Type.INTEGER, 4096
        ),
        rclpy.parameter.Parameter(
            'max_tokens', rclpy.Parameter.Type.INTEGER, 0
        )
    ]
    node.set_parameters(params)

    node._publish_stats(
        prompt_tokens=500,
        completion_tokens=120,
        tokens_per_second=15.0,
        status='completed'
    )

    called_msg = mock_pub.publish.call_args[0][0]
    data = json.loads(called_msg.data)
    assert 'Output: 120/∞' in data['formatted']

    node.destroy_node()
