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
import os
import subprocess
from typing import Any, List

import rclpy
import yaml

from bob_llm.tool_utils import Tool
from bob_llm.tool_utils import register as default_register


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


def _run_ros_command(command: List[str]) -> str:
    """Run a ROS 2 command and return the output."""
    try:
        # Execute the command and capture output. A timeout is used to prevent hanging.
        timeout = int(os.environ.get('ROS_CLI_TOOL_TIMEOUT', '15'))
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,  # Raise CalledProcessError for non-zero exit codes
            timeout=timeout,
            env=os.environ.copy()
        )
        return process.stdout.strip()
    except subprocess.CalledProcessError as e:
        # Return a formatted error message if the command fails
        error_message = f"Command '{' '.join(command)}' failed with return code {e.returncode}.\n"
        error_message += f"Stderr: {e.stderr.strip()}"
        return error_message
    except subprocess.TimeoutExpired:
        return f"Command '{' '.join(command)}' timed out after {timeout} seconds."
    except FileNotFoundError:
        return "Error: 'ros2' command not found. Please ensure ROS 2 is installed."


# === Topic Tools ===

def list_topics() -> str:
    """List all active ROS 2 topics."""
    return _run_ros_command(["ros2", "topic", "list"])


def get_topic_info(topic_name: str) -> str:
    """Get detailed information about a specific ROS 2 topic."""
    return _run_ros_command(["ros2", "topic", "info", topic_name])


def echo_topic_once(topic_name: str) -> str:
    """Retrieve a single message from a ROS 2 topic."""
    return _run_ros_command(["ros2", "topic", "echo", "--once", topic_name])


def publish_topic_message(topic_name: str, message_type: str, message_yaml: str) -> str:
    """
    Publish a single message to a ROS 2 topic using YAML format.

    Example: publish_topic_message("/chatter", "std_msgs/msg/String", "data: 'hello'")
    """
    return _run_ros_command(
        ["ros2", "topic", "pub", "--once", "--keep-alive", "1.0",
         topic_name, message_type, message_yaml]
    )


# === Node Tools ===

def list_nodes() -> str:
    """List all active ROS 2 nodes."""
    return _run_ros_command(["ros2", "node", "list"])


def get_node_info(node_name: str) -> str:
    """Get detailed information about a specific ROS 2 node."""
    return _run_ros_command(["ros2", "node", "info", node_name])


# === Service Tools ===

def list_services() -> str:
    """List all available ROS 2 services."""
    return _run_ros_command(["ros2", "service", "list"])


def get_service_type(service_name: str) -> str:
    """Get the type of a specific ROS 2 service."""
    return _run_ros_command(["ros2", "service", "type", service_name])


def call_service(service_name: str, service_type: str, request_yaml: str) -> str:
    """Call a ROS 2 service with a request in YAML format."""
    return _run_ros_command(
        ["ros2", "service", "call", service_name, service_type, request_yaml]
    )


# === Param Tools ===

def list_node_params(node_name: str) -> str:
    """List all parameters of a specific ROS 2 node."""
    # If the tool is asking about itself, use the fast internal API
    if _NodeContext.node and _NodeContext.node.get_fully_qualified_name() == node_name:
        param_descriptions = _NodeContext.node.describe_parameters(
            _NodeContext.node.get_parameters_by_prefix('').keys())
        return "\n".join(p.name for p in param_descriptions)

    return _run_ros_command(["ros2", "param", "list", node_name])


def get_param(node_name: str, param_name: str) -> str:
    """Get the value of a specific parameter from a ROS 2 node."""
    if _NodeContext.node and _NodeContext.node.get_fully_qualified_name() == node_name:
        try:
            param = _NodeContext.node.get_parameter(param_name)
            res = (f"Parameter '{param.name}' is of type '{param.type_.name}' "
                   f"with value: {param.value}")
            return res
        except rclpy.exceptions.ParameterNotDeclaredException:
            return f"Error: Parameter '{param_name}' not found on node '{node_name}'."

    return _run_ros_command(["ros2", "param", "get", node_name, param_name])


def set_param(node_name: str, param_name: str, value: str) -> str:
    """Set the value of a specific parameter on a ROS 2 node."""
    if _NodeContext.node and _NodeContext.node.get_fully_qualified_name() == node_name:
        try:
            # Use yaml.safe_load to infer type (int, float, bool, etc.) from the string value
            typed_value = yaml.safe_load(value)
            param = rclpy.Parameter(param_name, value=typed_value)
            # set_parameters returns a list of SetParametersResult
            result = _NodeContext.node.set_parameters([param])[0]
            if result.successful:
                return f"Successfully set parameter '{param_name}'."

            return f"Failed to set parameter '{param_name}': {result.reason}"
        except Exception as e:
            return f"An error occurred while setting parameter '{param_name}': {e}"

    return _run_ros_command(["ros2", "param", "set", node_name, param_name, value])
