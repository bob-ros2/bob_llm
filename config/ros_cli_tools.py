"""
A tool interface for the LLM node providing ROS 2 CLI functionalities.

This module contains functions that wrap common `ros2` command-line tools,
allowing an LLM to inspect and interact with a ROS 2 system.
"""
import os
import yaml
import subprocess
from typing import List, Any
import rclpy
from bob_llm.tool_utils import Tool
from bob_llm.tool_utils import register as default_register

class _NodeContext:
    """A private class to hold the ROS node instance."""
    node: rclpy.node = None

def register(module: Any, node: Any = None) -> List[Tool]:
    """
    Inspects a module and builds a list of OpenAI-compatible tool definitions
    from its functions. It ignores functions starting with '_' and 'register'.
    """
    _NodeContext.node = node
    return default_register(module, node)

def _run_ros_command(command: List[str]) -> str:
    """Helper function to execute a ROS 2 command and return the output."""
    try:
        # Execute the command and capture output. A timeout is used to prevent hanging.
        timeout = int(os.environ.get('ROS_CLI_TOOL_TIMEOUT', '15'))
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True, # Raise CalledProcessError for non-zero exit codes
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
        return "Error: 'ros2' command not found. Please ensure ROS 2 is installed and the environment is sourced."

# === Topic Tools ===

def list_topics() -> str:
    """Lists all active ROS 2 topics."""
    return _run_ros_command(["ros2", "topic", "list"])

def get_topic_info(topic_name: str) -> str:
    """Gets detailed information about a specific ROS 2 topic, including its type."""
    return _run_ros_command(["ros2", "topic", "info", topic_name])

def echo_topic_once(topic_name: str) -> str:
    """Retrieves a single message from a ROS 2 topic."""
    return _run_ros_command(["ros2", "topic", "echo", "--once", topic_name])

def publish_topic_message(topic_name: str, message_type: str, message_yaml: str) -> str:
    """Publishes a single message to a ROS 2 topic using YAML format.
    Example: publish_topic_message("/chatter", "std_msgs/msg/String", "data: 'hello world'")"""
    return _run_ros_command(["ros2", "topic", "pub", "--once", "--keep-alive", "1.0", topic_name, message_type, message_yaml])

# === Node Tools ===

def list_nodes() -> str:
    """Lists all active ROS 2 nodes."""
    return _run_ros_command(["ros2", "node", "list"])

def get_node_info(node_name: str) -> str:
    """Gets detailed information about a specific ROS 2 node."""
    return _run_ros_command(["ros2", "node", "info", node_name])

# === Service Tools ===

def list_services() -> str:
    """Lists all available ROS 2 services."""
    return _run_ros_command(["ros2", "service", "list"])

def get_service_type(service_name: str) -> str:
    """Gets the type of a specific ROS 2 service."""
    return _run_ros_command(["ros2", "service", "type", service_name])

def call_service(service_name: str, service_type: str, request_yaml: str) -> str:
    """Calls a ROS 2 service with a request in YAML format."""
    # Example: call_service("/add_two_ints", "example_interfaces/srv/AddTwoInts", "a: 1, b: 2")
    return _run_ros_command(["ros2", "service", "call", service_name, service_type, request_yaml])

# === Param Tools ===

def list_node_params(node_name: str) -> str:
    """Lists all parameters of a specific ROS 2 node."""
    # If the tool is asking about itself, use the fast internal API
    if _NodeContext.node and _NodeContext.node.get_fully_qualified_name() == node_name:
        param_descriptions = _NodeContext.node.describe_parameters(
            _NodeContext.node.get_parameters_by_prefix('').keys())
        return "\n".join(p.name for p in param_descriptions)
    else:
        return _run_ros_command(["ros2", "param", "list", node_name])

def get_param(node_name: str, param_name: str) -> str:
    """Gets the value of a specific parameter from a ROS 2 node."""
    if _NodeContext.node and _NodeContext.node.get_fully_qualified_name() == node_name:
        try:
            param = _NodeContext.node.get_parameter(param_name)
            return f"Parameter '{param.name}' is of type '{param.type_.name}' with value: {param.value}"
        except rclpy.exceptions.ParameterNotDeclaredException:
            return f"Error: Parameter '{param_name}' not found on node '{node_name}'."
    else:
        return _run_ros_command(["ros2", "param", "get", node_name, param_name])

def set_param(node_name: str, param_name: str, value: str) -> str:
    """Sets the value of a specific parameter on a ROS 2 node."""
    if _NodeContext.node and _NodeContext.node.get_fully_qualified_name() == node_name:
        try:
            # Use yaml.safe_load to infer type (int, float, bool, etc.) from the string value
            typed_value = yaml.safe_load(value)
            param = rclpy.Parameter(param_name, value=typed_value)
            # set_parameters returns a list of SetParametersResult
            result = _NodeContext.node.set_parameters([param])[0]
            if result.successful:
                return f"Successfully set parameter '{param_name}'."
            else:
                return f"Failed to set parameter '{param_name}': {result.reason}"
        except Exception as e:
            return f"An error occurred while setting parameter '{param_name}': {e}"
    else:
        return _run_ros_command(["ros2", "param", "set", node_name, param_name, value])