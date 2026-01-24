#!/usr/bin/env python3
import os
import json
import base64
import requests
import mimetypes
import logging
import traceback
import importlib
import importlib.util
from collections import deque
import rclpy
from rclpy.node import Node
from rclpy.logging import LoggingSeverity
from std_msgs.msg import String
from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import ParameterType
from rclpy.parameter import Parameter
from rclpy.executors import MultiThreadedExecutor
from ament_index_python.packages import get_package_share_directory
from rclpy.callback_groups import ReentrantCallbackGroup
from bob_llm.tool_utils import register as default_register
from bob_llm.backend_clients import OpenAICompatibleClient

class LLMNode(Node):
    """
    A ROS 2 node that interfaces with an OpenAI-compatible LLM.

    This node handles chat history, tool execution, and communication with an LLM
    backend, configured entirely through ROS parameters.
    """
    def __init__(self):
        super().__init__('llm')

        # Synchronize logging level with ROS logger verbosity for library output.
        logging.basicConfig(
            level = (logging.DEBUG
                if self.get_logger().get_effective_level() \
                    == LoggingSeverity.DEBUG \
                else logging.INFO),
            format="[%(levelname)s] [%(asctime)s.] [%(name)s]: %(message)s",
            datefmt="%s")

        self.get_logger().info("LLM Node starting up...")

        # Get the string list from environment or use the example as default
        interfaces_array = os.environ.get('LLM_TOOL_INTERFACES', 
            os.path.join(
                get_package_share_directory("bob_llm"),
                "config",
                "example_interface.py"))
        interfaces_array = interfaces_array.split(',')

        # ROS parameters

        self.declare_parameter('api_type', 
            os.environ.get('LLM_API_TYPE', 'openai_compatible'),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='The type of the LLM backend API (e.g., "openai_compatible").'
            )
        )
        self.declare_parameter('api_url',
            os.environ.get('LLM_API_URL', 'http://localhost:8000'),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='The base URL of the LLM backend API.'
            )
        )
        self.declare_parameter('api_key',
            os.environ.get('LLM_API_KEY', 'no_key'),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='The API key for authentication with the LLM backend.'
            )
        )
        self.declare_parameter('api_model',
            os.environ.get('LLM_API_MODEL', ''),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='The specific model name to use (e.g., "gpt-4", "llama3").'
            )
        )
        self.declare_parameter('system_prompt', 
            os.environ.get('LLM_SYSTEM_PROMPT', ''),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='The system prompt to set the LLM context.'
            )
        )
        self.declare_parameter('initial_messages_json',
            os.environ.get('LLM_INITIAL_MESSAGES_JSON', '[]'),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='A JSON string of initial messages for few-shot prompting.'
            )
        )
        self.declare_parameter('max_history_length', 
            int(os.environ.get('LLM_MAX_HISTORY_LENGTH', '10')),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description='Maximum number of conversational turns to keep in history.'
            )
        )
        self.declare_parameter('stream',
            os.environ.get('LLM_STREAM', 'true').lower() in ('true', '1', 'yes'),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description='Enable or disable streaming for the final LLM response.'
            )
        )
        self.declare_parameter('max_tool_calls', 
            int(os.environ.get('LLM_MAX_TOOL_CALLS', '5')),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description='Maximum number of consecutive tool calls before aborting.'
            )
        )
        self.declare_parameter('temperature',
            float(os.environ.get('LLM_TEMPERATURE', '0.7')),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description='Controls the randomness of the output. Lower is more deterministic.'
            )
        )
        self.declare_parameter('top_p',
            float(os.environ.get('LLM_TOP_P', '1.0')),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description='Nucleus sampling. Controls output diversity. Alter this or temperature, not both.'
            )
        )
        self.declare_parameter('max_tokens',
            int(os.environ.get('LLM_MAX_TOKENS', '0')),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description='Maximum number of tokens to generate. 0 means no limit.'
            )
        )
        self.declare_parameter('stop',
            os.environ.get('LLM_STOP', 'stop_llm').split(','),
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING_ARRAY,
                description='A list of sequences to stop generation at.'
            )
        )
        self.declare_parameter('presence_penalty',
            float(os.environ.get('LLM_PRESENCE_PENALTY', '0.0')),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description='Penalizes new tokens based on their presence in the text so far.'
            )
        )
        self.declare_parameter('frequency_penalty',
            float(os.environ.get('LLM_FREQUENCY_PENALTY', '0.0')),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description='Penalizes new tokens based on their frequency in the text so far.'
            )
        )
        self.declare_parameter('api_timeout',
            float(os.environ.get('LLM_API_TIMEOUT', '120.0')),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description='Timeout in seconds for API requests to the LLM backend.'
            )
        )
        self.declare_parameter('tool_interfaces', 
            interfaces_array,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING_ARRAY,
                description='A list of Python modules or file paths to load as tool interfaces.'
            )
        )
        self.declare_parameter('message_log',
            os.environ.get('LLM_MESSAGE_LOG', ''),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='If set, appends each user/assistant turn to this JSON file.'
            )
        )
        self.declare_parameter('process_image_urls',
            False,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description='If true, processes image_url in JSON prompts by base64 encoding the image.'
            )
        )
        self.declare_parameter('response_format',
            os.environ.get('LLM_RESPONSE_FORMAT', ''),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='JSON string defining the output format. E.g. {"type": "json_object"} or {"type": "json_schema", "json_schema": {...}}'
            )
        )

        self.chat_history = []
        self.load_llm_client()
        self._initialize_chat_history()
        self._prefix_history_len = len(self.chat_history)

        # Load tools and their corresponding functions
        self.tools, self.tool_functions = self._load_tools()
        if self.tools:
            self.get_logger().info(f"Successfully loaded {len(self.tools)} tools.")

        DEFAULT_QUEUE_SIZE = int(os.environ.get('LLM_QUEUE_SIZE', '1000'))

        self._is_generating = False
        self._cancel_requested = False
        
        # Prompt queue to handle incoming prompts when busy
        self._prompt_queue = deque()
        self._queue_timer = None
        self._queue_timer_period = 0.1  # Check queue every 100ms

        self.sub = self.create_subscription(
            String, 'llm_prompt', self.prompt_callback, DEFAULT_QUEUE_SIZE,
            callback_group=ReentrantCallbackGroup())

        self.pub_response = self.create_publisher(
            String, 'llm_response', DEFAULT_QUEUE_SIZE)

        self.pub_stream = self.create_publisher(
            String, 'llm_stream', DEFAULT_QUEUE_SIZE)

        self.pub_latest_turn = self.create_publisher(
            String, 'llm_latest_turn', DEFAULT_QUEUE_SIZE)

        self.get_logger().info(
            f"Node is ready. History has {len(self.chat_history)} initial messages.")

    def _initialize_chat_history(self):
        """
        Populates the initial chat history from ROS parameters.

        This method adds the system prompt and any few-shot examples provided in
        the 'system_prompt' and 'initial_messages_json' parameters, respectively,
        to guide the LLM's behavior.
        """
        system_prompt = self.get_parameter('system_prompt').value
        if system_prompt:
            self.chat_history.append({"role": "system", "content": system_prompt})
            self.get_logger().info("System prompt added.")
        initial_messages_str = self.get_parameter('initial_messages_json').value
        try:
            initial_messages = json.loads(initial_messages_str)
            if isinstance(initial_messages, list):
                self.chat_history.extend(initial_messages)
                self.get_logger().info(
                    f"Loaded {len(initial_messages)} initial messages from JSON.")
        except json.JSONDecodeError:
            self.get_logger().error(
                "Failed to parse 'initial_messages_json'.")

    def load_llm_client(self):
        """
        Loads and configures the LLM client based on ROS parameters.

        This method reads the 'api_*' and generation parameters (e.g., temperature,
        top_p) to instantiate and configure the appropriate backend client for
        communicating with the Large Language Model.
        """
        api_type = self.get_parameter('api_type').value
        api_url = self.get_parameter('api_url').value
        model = self.get_parameter('api_model').value

        if not api_url:
            self.get_logger().error(
                "LLM URL not configured. Please set 'api_url' parameter.")
            return

        self.get_logger().info(f"Connecting to LLM at {api_url} with model {model}")

        if api_type == 'openai_compatible':

            try:
                stop = self.get_parameter('stop').value
            except:
                stop = None

            # Parse response_format if provided
            response_format_str = self.get_parameter('response_format').value
            response_format = None
            if response_format_str:
                try:
                    response_format = json.loads(response_format_str)
                    self.get_logger().info(f"Using response_format: {response_format}")
                except json.JSONDecodeError as e:
                    self.get_logger().error(f"Failed to parse 'response_format' JSON: {e}")

            self.llm_client = OpenAICompatibleClient(
                api_url=          self.get_parameter('api_url').value,
                api_key=          self.get_parameter('api_key').value,
                model=            self.get_parameter('api_model').value,
                logger=           self.get_logger(),
                temperature=      self.get_parameter('temperature').value,
                top_p=            self.get_parameter('top_p').value,
                max_tokens=       self.get_parameter('max_tokens').value,
                stop=             stop,
                presence_penalty= self.get_parameter('presence_penalty').value,
                frequency_penalty=self.get_parameter('frequency_penalty').value,
                timeout=          self.get_parameter('api_timeout').value,
                response_format=  response_format
            )
        else:
            self.get_logger().error(f"Unsupported API type: {api_type}")

    def _load_tools(self) -> (list, dict):
        """
        Dynamically loads tool modules specified in 'tool_interfaces'.

        Supports loading from both Python module names (e.g., 'my_package.tools')
        and absolute file paths. It generates an OpenAI-compatible schema for each
        function and maps the function name to its callable object.

        Returns:
            A tuple containing a list of tool schemas for the LLM and a dictionary
            mapping function names to their callable objects.
        """
        try:
            tool_modules_paths = self.get_parameter('tool_interfaces').value
        except:
            return [], {}

        self.get_logger().info(
            f"Loading tool interfaces: {tool_modules_paths}")

        all_tools = []
        all_functions = {}
        for path_str in tool_modules_paths:
            try:
                # Check if the path is a file, otherwise treat it as a module
                if path_str.endswith('.py') and os.path.exists(path_str):
                    module_name = os.path.splitext(os.path.basename(path_str))[0]
                    spec = importlib.util.spec_from_file_location(module_name, path_str)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    self.get_logger().debug(
                        f"Imported module from file: {path_str}")
                else:
                    module = importlib.import_module(path_str)
                    self.get_logger().info(f"Imported module by name: {path_str}")

                # Use the module's register function if it exists, otherwise use the default
                if hasattr(module, 'register') and callable(getattr(module, 'register')):
                    self.get_logger().info(f"Using custom 'register' from {path_str}")
                    tools = module.register(module, self)
                else:
                    self.get_logger().info(f"Using default 'register' for {path_str}")
                    tools = default_register(module, self)

                all_tools.extend(tools)

                # Map function names from the schema to the actual callable functions
                for tool_def in tools:
                    func_name = tool_def['function']['name']
                    if hasattr(module, func_name):
                        all_functions[func_name] = getattr(module, func_name)

            except ImportError as e:
                self.get_logger().error(f"Failed to import tool module {path_str}: {e}")
            except Exception as e:
                self.get_logger().error(f"Error loading tools from {path_str}: {e}")

        return all_tools, all_functions

    def _publish_latest_turn(self, user_prompt: str, assistant_message: dict):
        """
        Processes the latest conversational turn for publishing and logging.

        Args:
            user_prompt: The string content of the user's latest prompt.
            assistant_message: The final message dictionary from the assistant,
                               e.g., `{'role': 'assistant', 'content': '...'}`.
        """
        try:
            user_msg = {"role": "user", "content": user_prompt}
            # The assistant_message is already a dict like {'role': 'assistant', 'content': ...}
            latest_turn_list = [user_msg, assistant_message]
            json_string = json.dumps(latest_turn_list)
            self.pub_latest_turn.publish(
                String(data=json_string))
        except TypeError as e:
            self.get_logger().error(f"Failed to serialize latest turn to JSON: {e}")

        # --- Log to file ---
        log_file_path = self.get_parameter('message_log').value
        if not log_file_path:
            return  # Do nothing if the parameter is not set

        try:
            log_data = []
            # Check if the file exists and has content before trying to read it
            if os.path.exists(log_file_path) and os.path.getsize(log_file_path) > 0:
                with open(log_file_path, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
                if not isinstance(log_data, list):
                    self.get_logger().warning(f"Log file '{log_file_path}' contained invalid data. Overwriting.")
                    log_data = []

            # If the log is empty, it's a new log. Prepend the system prompt if it exists.
            if not log_data:
                system_prompt = self.get_parameter('system_prompt').value
                if system_prompt:
                    log_data.append({"role": "system", "content": system_prompt})

            # Append new messages from the latest turn
            log_data.append({"role": "user", "content": user_prompt})
            log_data.append(assistant_message)

            # Write the updated array back to the file
            with open(log_file_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2)

        except (IOError, json.JSONDecodeError) as e:
            self.get_logger().error(f"Error processing message log file '{log_file_path}': {e}")
            
    def _get_truncated_history(self):
        """Returns a copy of chat history with long strings truncated for logging."""
        truncated_history = []
        for msg in self.chat_history:
            msg_copy = msg.copy()
            
            # Handle OpenAI Vision format (list of content parts)
            if isinstance(msg_copy.get("content"), list):
                new_content = []
                for part in msg_copy["content"]:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        part_copy = part.copy()
                        if "image_url" in part_copy and "url" in part_copy["image_url"]:
                            url = part_copy["image_url"]["url"]
                            if len(url) > 100:
                                part_copy["image_url"] = {
                                    "url": f"{url[:30]}...<truncated_base64>...{url[-30:]}"
                                }
                        new_content.append(part_copy)
                    else:
                        new_content.append(part)
                msg_copy["content"] = new_content
            
            truncated_history.append(msg_copy)
        return truncated_history

    def _trim_chat_history(self):
        """
        Prevents the chat history from growing indefinitely.

        It trims the oldest conversational turns to stay within the 'max_history_length'
        limit, preserving the system prompt and any initial few-shot examples.
        A turn is defined as a user message and all subsequent assistant/tool
        messages that follow it.
        """
        max_len = self.get_parameter('max_history_length').value

        # Separate the static prefix (system prompt, few-shot) from the conversation
        prefix = self.chat_history[:self._prefix_history_len]
        conversation = self.chat_history[self._prefix_history_len:]

        # Find the start of each conversational turn (marked by a 'user' message)
        user_message_indices = [
            i for i, msg in enumerate(conversation) if msg['role'] == 'user']
        
        num_turns = len(user_message_indices)

        if num_turns > max_len:
            # Determine the index of the first message of the first turn to keep
            first_turn_to_keep_idx = user_message_indices[num_turns - max_len]

            # Trim the conversation, keeping only the most recent turns
            trimmed_conversation = conversation[first_turn_to_keep_idx:]

            # Reconstruct the full chat history
            self.chat_history = prefix + trimmed_conversation
            self.get_logger().info(
                f"Trimmed {num_turns - max_len} old turn(s) from chat history.")
        
        self.get_logger().debug(f"History: {str(self._get_truncated_history())}")

    def prompt_callback(self, msg):
        """
        Processes an incoming prompt from the 'llm_prompt' topic.

        This is the core callback for the node. It manages the conversation flow
        by first entering a loop to handle potential tool calls from the LLM. Once
        the LLM decides to respond with text, it exits the loop and generates the
        final response, either by streaming it token-by-token or as a single
        message, based on the 'stream' parameter.

        Args:
            msg: The std_msgs/String message containing the user's prompt.
        """
        # --- Cancellation Check ---
        stop_list = self.get_parameter('stop').value
        if msg.data in stop_list:
            if self._is_generating:
                self.get_logger().warn(f"Cancellation requested via stop command: '{msg.data}'")
                self._cancel_requested = True
            else:
                self.get_logger().info(f"Received stop command '{msg.data}', but no generation is active.")
            return

        # --- Busy Check ---
        if self._is_generating:
            self._prompt_queue.append(msg.data)
            self.get_logger().info(f"Node is busy. Queued prompt. Queue size: {len(self._prompt_queue)}")
            
            # Start timer if not already running
            if self._queue_timer is None:
                self._queue_timer = self.create_timer(
                    self._queue_timer_period, 
                    self._process_queue_callback,
                    callback_group=ReentrantCallbackGroup()
                )
            return

        # --- Start Generation ---
        self._is_generating = True
        self._cancel_requested = False

        try:
            if not self.llm_client:
                self.get_logger().error("LLM client is not available. Cannot process prompt.")
                return

            self.get_logger().info(f"Prompt received")
            self.get_logger().debug(f"Promp: '{msg.data}'")

            # Try to parse as JSON first
            prompt_text_for_log = msg.data
            new_messages = []

            try:
                json_data = json.loads(msg.data)
                if isinstance(json_data, list):
                    new_messages = json_data
                    # Attempt to extract clean text for logging from the last user message
                    for m in reversed(new_messages):
                        if isinstance(m, dict) and m.get("role") == "user":
                            c = m.get("content", "")
                            if isinstance(c, str):
                                prompt_text_for_log = c
                            break
                elif isinstance(json_data, dict):
                    user_content = json_data
                    # It's a valid message structure
                    if json_data.get("role") == "user":
                        # Extract text for log
                        c = json_data.get("content", "")
                        if isinstance(c, str):
                            prompt_text_for_log = c

                        if self.get_parameter('process_image_urls').value and "image_url" in json_data:
                            image_url = json_data["image_url"]
                            try:
                                image_data = None
                                if image_url.startswith("file://"):
                                    file_path = image_url[7:]
                                    with open(file_path, "rb") as image_file:
                                        mime_type, _ = mimetypes.guess_type(file_path)
                                        if not mime_type:
                                            mime_type = "image/jpeg" # Default fallback
                                        base64_data = base64.b64encode(image_file.read()).decode('utf-8')
                                        image_data = f"data:{mime_type};base64,{base64_data}"
                                elif image_url.startswith("http"):
                                     response = requests.get(image_url)
                                     response.raise_for_status()
                                     mime_type = response.headers.get('Content-Type', 'image/jpeg')
                                     base64_data = base64.b64encode(response.content).decode('utf-8')
                                     image_data = f"data:{mime_type};base64,{base64_data}"

                                if image_data:
                                    # Construct the new message structure using OpenAI Vision format
                                    user_content = {
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "text",
                                                "text": json_data.get("content", "")
                                            },
                                            {
                                                "type": "image_url",
                                                "image_url": {
                                                    "url": image_data
                                                }
                                            }
                                        ]
                                    }
                                    self.get_logger().info(f"Processed image from {image_url}")
                                else:
                                     self.get_logger().warning(f"Could not process image url: {image_url}")
                                     user_content = json_data # Fallback to original json dict

                            except Exception as e:
                                self.get_logger().error(f"Failed to process image url {image_url}: {e}")
                                user_content = json_data # Fallback
                    
                    new_messages = [user_content]
                else:
                     # Parsed as JSON but not list or dict (e.g. number, bool)
                     new_messages = [{"role": "user", "content": str(json_data)}]
                     prompt_text_for_log = str(json_data)

            except json.JSONDecodeError:
                # Not JSON, treat as plain string
                new_messages = [{"role": "user", "content": msg.data}]
                prompt_text_for_log = msg.data

            self.chat_history.extend(new_messages)
            self._trim_chat_history()

            stream_enabled = self.get_parameter('stream').value
            max_calls = self.get_parameter('max_tool_calls').value
            tool_call_count = 0

            # Phase 1: Tool-handling loop (always non-streaming)
            while tool_call_count < max_calls:
                if self._cancel_requested:
                    self.get_logger().warn("Generation cancelled during tool loop.")
                    return

                success, response_message = self.llm_client.process_prompt(
                    self.chat_history,
                    self.tools if self.tools else None
                )

                if not success:
                    self.get_logger().error(f"Error during LLM request: {response_message}")
                    self.chat_history.pop() # Remove the user message that caused an error
                    return

                if response_message.get("tool_calls"):
                    # The LLM wants to use a tool, so we add its request to history
                    # Ensure content is a string (not null) for llama.cpp compatibility
                    if response_message.get("content") is None:
                        response_message["content"] = ""
                    self.chat_history.append(response_message)
                    self.get_logger().info(f"LLM requested a tool call: {response_message['tool_calls']}")
                    tool_call_count += 1
                    tool_calls = response_message["tool_calls"]
                    for tool_call in tool_calls:
                        if self._cancel_requested:
                            self.get_logger().warn("Generation cancelled before tool execution.")
                            return

                        function_name = tool_call['function']['name']
                        tool_call_id = tool_call['id']

                        func_to_call = self.tool_functions.get(function_name)
                        if not func_to_call:
                            error_msg = f"Error: Tool '{function_name}' not found."
                            self.get_logger().error(error_msg)
                            self.chat_history.append({
                                "tool_call_id": tool_call_id, "role": "tool",
                                "name": function_name, "content": error_msg
                            })
                            continue

                        try:
                            args_str = tool_call['function']['arguments']
                            args_dict = json.loads(args_str)

                            self.get_logger().info(f"Executing tool '{function_name}' with args: {args_dict}")
                            result = func_to_call(**args_dict)
                            # Serialize result to JSON string if possible
                            if isinstance(result, str):
                                content_str = result
                            else:
                                try:
                                    content_str = json.dumps(result, ensure_ascii=False)
                                except (TypeError, ValueError):
                                    content_str = str(result)

                            self.get_logger().debug(content_str)

                            self.chat_history.append({
                                "tool_call_id": tool_call_id, "role": "tool",
                                "name": function_name, "content": content_str
                            })
                        except Exception as e:
                            error_trace = traceback.format_exc()
                            error_msg = f"Error executing tool {function_name}: {e}"
                            self.get_logger().error(f"{error_msg}\n{error_trace}")
                            self.chat_history.append({
                                "tool_call_id": tool_call_id, "role": "tool",
                                "name": function_name, "content": error_msg
                            })
                    continue
                else:
                    # LLM is ready to generate a text response, so we break the loop.
                    # We do NOT add its empty message to history here.
                    break

            if tool_call_count >= max_calls:
                # Handle reaching the tool call limit
                error_msg = f"Max tool calls ({max_calls}) reached. Aborting."
                self.get_logger().warning(error_msg)
                self.pub_response.publish(String(
                    data="I seem to be stuck in a tool-use loop. Please rephrase your request."))
                return

            if self._cancel_requested:
                self.get_logger().warn("Generation cancelled before final response.")
                return

            # Phase 2: Final response generation (stream or single response)
            if stream_enabled:
                self.get_logger().info("Streaming final response...")
                full_response = ""

                for chunk in self.llm_client.stream_prompt(self.chat_history):
                    if self._cancel_requested:
                        self.get_logger().warn("Generation cancelled during streaming.")
                        self.pub_response.publish(String(data="[Cancelled]"))
                        return

                    if chunk:
                        full_response += chunk
                        self.pub_stream.publish(
                            String(data=chunk))

                self.pub_response.publish(
                    String(data=full_response))
                self.get_logger().debug(
                    f"Finished streaming. Full response: {full_response[:80]}...")
                assistant_message = {"role": "assistant", "content": full_response}
                self.chat_history.append(assistant_message)
                self._publish_latest_turn(prompt_text_for_log, assistant_message)
            else:
                self.get_logger().info("Generating non-streamed final response...")
                success, final_message = self.llm_client.process_prompt(self.chat_history)

                if self._cancel_requested:
                    self.get_logger().warn("Generation cancelled after non-streamed response.")
                    return

                if success and final_message.get("content"):
                    llm_response_text = final_message["content"]
                    self.pub_response.publish(
                        String(data=llm_response_text))
                    self.get_logger().info(
                        f"Published LLM response: {llm_response_text[:80]}...")
                    self.chat_history.append(final_message)
                    self._publish_latest_turn(prompt_text_for_log, final_message)
                else:
                    self.get_logger().error(
                        f"Failed to get final non-streamed response: {final_message}")
                    self.pub_response.publish(
                        String(data="Sorry, I encountered an error generating my final response."))
        finally:
            self._is_generating = False

    def _process_queue_callback(self):
        """
        Timer callback to process queued prompts when the node is not busy.
        
        This callback is triggered periodically to check if there are any queued
        prompts waiting to be processed. If the node is not currently generating
        and there are prompts in the queue, it processes the oldest one.
        """
        # If still generating, wait for next timer tick
        if self._is_generating:
            return
        
        # If queue is empty, stop the timer
        if not self._prompt_queue:
            if self._queue_timer is not None:
                self._queue_timer.cancel()
                self._queue_timer = None
            return
        
        # Process the next prompt from the queue
        prompt_data = self._prompt_queue.popleft()
        self.get_logger().info(f"Processing queued prompt. Remaining in queue: {len(self._prompt_queue)}")
        
        # Create a String message and process it
        msg = String()
        msg.data = prompt_data
        self.prompt_callback(msg)

def main(args=None):
    rclpy.init(args=args)
    llm_node = LLMNode()
    executor = MultiThreadedExecutor()
    executor.add_node(llm_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        llm_node.get_logger().info(
            "KeyboardInterrupt received, shutting down.")
    finally:
        llm_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()