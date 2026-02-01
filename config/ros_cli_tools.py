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

"""A tool interface for the LLM node providing ROS 2 CLI functionalities."""

import subprocess
from typing import Any, List

from bob_llm.tool_utils import register as default_register
from bob_llm.tool_utils import Tool
import rclpy
import yaml


class _NodeContext:
    """Private class to hold the ROS node instance."""

    node: rclpy.node = None


def register(module: Any, node: Any = None) -> List[Tool]:
    """
    Inspect a module and build a list of tool definitions.

    :param module: The module to inspect.
    :param node: The ROS node instance.
    :return: A list of tool definitions.
    """
    _NodeContext.node = node
    return default_register(module, node)


def _run_ros_command(command: List[str], timeout: float = 10.0) -> str:
    """
    Execute a ROS 2 CLI command and return its output.

    :param command: The list of command arguments.
    :param timeout: Maximum execution time in seconds.
    :return: The command output or an error message.
    """
    try:
        # Execute the command and capture output. A timeout is used to prevent hanging.
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        cmd_str = ' '.join(command)
        error_message = f"Command '{cmd_str}' failed with return code {e.returncode}.\n"
        if e.stderr:
            error_message += f'Stderr: {e.stderr.strip()}'
        return error_message
    except subprocess.TimeoutExpired:
        cmd_str = ' '.join(command)
        return f"Command '{cmd_str}' timed out after {timeout} seconds."
    except FileNotFoundError:
        return "Error: 'ros2' command not found. Please ensure ROS 2 is installed."


# --- Tool Definitions ---

def list_topics() -> str:
    """List all active ROS 2 topics."""
    return _run_ros_command(['ros2', 'topic', 'list'])


def list_nodes() -> str:
    """List all active ROS 2 nodes."""
    return _run_ros_command(['ros2', 'node', 'list'])


def get_topic_info(topic_name: str) -> str:
    """Display information about a specific ROS 2 topic."""
    return _run_ros_command(['ros2', 'topic', 'info', topic_name])


def publish_topic_message(topic_name: str, message_type: str, message_yaml: str) -> str:
    r"""
    Publish a message to a ROS 2 topic.

    Example: publish_topic_message('/chatter', 'std_msgs/msg/String', 'data: \'hello\'')
    """
    return _run_ros_command([
        'ros2', 'topic', 'pub', '--once', topic_name, message_type, message_yaml
    ])


def echo_topic(topic_name: str, count: int = 1) -> str:
    """
    Read (echo) messages from a ROS 2 topic.

    :param topic_name: The name of the topic to read.
    :param count: Number of messages to receive (default 1).
    """
    return _run_ros_command(['ros2', 'topic', 'echo', '--once', topic_name, '-n', str(count)])


def list_services() -> str:
    """List all active ROS 2 services."""
    return _run_ros_command(['ros2', 'service', 'list'])


def get_service_type(service_name: str) -> str:
    """Get the type of a specific ROS 2 service."""
    return _run_ros_command(['ros2', 'service', 'type', service_name])


def call_service(service_name: str, service_type: str, request_yaml: str) -> str:
    """
    Call a ROS 2 service.

    :param service_name: The name of the service to call.
    :param service_type: The type of the service.
    :param request_yaml: The service request encoded in YAML.
    """
    return _run_ros_command(['ros2', 'service', 'call', service_name, service_type, request_yaml])


def get_parameter(node_name: str, param_name: str) -> str:
    """
    Get the value of a parameter from a ROS 2 node.

    If the node is this LLM node, it queries the node directly for the latest value.
    """
    if _NodeContext.node and _NodeContext.node.get_fully_qualified_name() == node_name:
        param = _NodeContext.node.get_parameter(param_name)
        if param:
            return (f"Parameter '{param_name}' on node '{node_name}' is "
                    f'set to: {param.value}')
    return _run_ros_command(['ros2', 'param', 'get', node_name, param_name])


def list_parameters(node_name: str) -> str:
    """List all parameters of a specific ROS 2 node."""
    if _NodeContext.node and _NodeContext.node.get_fully_qualified_name() == node_name:
        params = _NodeContext.node._parameters.keys()
        params_str = ', '.join(params)
        return f"Parameters for node '{node_name}': {params_str}"
    return _run_ros_command(['ros2', 'param', 'list', node_name])


def set_parameter(node_name: str, param_name: str, value: str) -> str:
    """
    Set a parameter on a ROS 2 node.

    :param node_name: The target node name.
    :param param_name: The parameter name to set.
    :param value: The value to set (string).
    """
    if _NodeContext.node and _NodeContext.node.get_fully_qualified_name() == node_name:
        try:
            # Use yaml.safe_load to infer type (int, float, bool, etc.) from the string value
            typed_value = yaml.safe_load(value)
            new_param = rclpy.parameter.Parameter(
                param_name,
                rclpy.Parameter.Type.from_python_tag(type(typed_value)),
                typed_value
            )
            _NodeContext.node.set_parameters([new_param])
            return (f"Successfully set parameter '{param_name}' on "
                    f"node '{node_name}' to {value}.")
        except Exception as e:
            return f"An error occurred while setting parameter '{param_name}': {e}"

    return _run_ros_command(['ros2', 'param', 'set', node_name, param_name, value])
