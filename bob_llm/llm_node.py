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

import base64
import importlib
import importlib.util
import json
import logging
import mimetypes
import os
from collections import deque

import rclpy
from ament_index_python.packages import get_package_share_directory
from bob_llm.backend_clients import OpenAICompatibleClient
from bob_llm.tool_utils import register as default_register
from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import ParameterType
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.logging import LoggingSeverity
from rclpy.node import Node
from std_msgs.msg import String


class LLMNode(Node):
    """
    ROS 2 node that provides an interface to LLMs and VLMs.

    This node handles prompts, manages conversation history, and executes tools.
    """

    def __init__(self):
        super().__init__('llm')

        # Synchronize logging level with ROS logger verbosity for library output.
        logging.basicConfig(
            level=(logging.DEBUG
                   if self.get_logger().get_effective_level()
                   == LoggingSeverity.DEBUG
                   else logging.INFO),
            format="[%(levelname)s] [%(asctime)s.] [%(name)s]: %(message)s",
            datefmt="%s")

        self.get_logger().info("LLM Node starting up...")

        # Get the string list from environment or use description
        share_dir = get_package_share_directory("bob_llm")
        interfaces_array = os.environ.get(
            'LLM_TOOL_INTERFACES',
            os.path.join(share_dir, "config", "example_interface.py"))
        interfaces_array = interfaces_array.split(',')

        # ROS parameters

        self.declare_parameter(
            'api_type',
            os.environ.get('LLM_API_TYPE', 'openai_compatible'),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='The type of the LLM backend API (e.g., "openai_compatible").'
            )
        )
        self.declare_parameter(
            'api_url',
            os.environ.get('LLM_API_URL', 'http://localhost:8000'),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='The base URL of the LLM backend API.'
            )
        )
        self.declare_parameter(
            'api_key',
            os.environ.get('LLM_API_KEY', 'no_key'),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='The API key for authentication with the LLM backend.'
            )
        )
        self.declare_parameter(
            'api_model',
            os.environ.get('LLM_API_MODEL', ''),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='The specific model name to use (e.g., "gpt-4", "llama3").'
            )
        )
        self.declare_parameter(
            'system_prompt',
            os.environ.get('LLM_SYSTEM_PROMPT', ''),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='The system prompt to set the LLM context.'
            )
        )
        self.declare_parameter(
            'initial_messages_json',
            os.environ.get('LLM_INITIAL_MESSAGES_JSON', '[]'),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='A JSON string of initial messages for few-shot prompting.'
            )
        )
        self.declare_parameter(
            'max_history_length',
            int(os.environ.get('LLM_MAX_HISTORY_LENGTH', '10')),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description='Maximum number of conversational turns to keep in history.'
            )
        )
        self.declare_parameter(
            'stream',
            os.environ.get('LLM_STREAM', 'true').lower() in ('true', '1', 'yes'),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description='Enable or disable streaming for the final LLM response.'
            )
        )
        self.declare_parameter(
            'max_tool_calls',
            int(os.environ.get('LLM_MAX_TOOL_CALLS', '5')),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description='Maximum number of consecutive tool calls before aborting.'
            )
        )
        self.declare_parameter(
            'temperature',
            float(os.environ.get('LLM_TEMPERATURE', '0.7')),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description='Controls the randomness of the output. Lower is more deterministic.'
            )
        )
        self.declare_parameter(
            'top_p',
            float(os.environ.get('LLM_TOP_P', '1.0')),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description='Nucleus sampling. Controls output diversity.'
            )
        )
        self.declare_parameter(
            'max_tokens',
            int(os.environ.get('LLM_MAX_TOKENS', '0')),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description='Maximum number of tokens to generate. 0 means no limit.'
            )
        )
        self.declare_parameter(
            'stop',
            os.environ.get('LLM_STOP', 'stop_llm').split(','),
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING_ARRAY,
                description='A list of sequences to stop generation at.'
            )
        )
        self.declare_parameter(
            'presence_penalty',
            float(os.environ.get('LLM_PRESENCE_PENALTY', '0.0')),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description='Penalizes new tokens based on their presence in the text so far.'
            )
        )
        self.declare_parameter(
            'frequency_penalty',
            float(os.environ.get('LLM_FREQUENCY_PENALTY', '0.0')),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description='Penalizes new tokens based on their frequency in the text.'
            )
        )
        self.declare_parameter(
            'api_timeout',
            float(os.environ.get('LLM_API_TIMEOUT', '120.0')),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description='Timeout in seconds for API requests to the LLM backend.'
            )
        )
        self.declare_parameter(
            'tool_interfaces',
            interfaces_array,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING_ARRAY,
                description='A list of Python modules or file paths to load as tools.'
            )
        )
        self.declare_parameter(
            'message_log',
            os.environ.get('LLM_MESSAGE_LOG', ''),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='If set, appends each user/assistant turn to this JSON file.'
            )
        )
        self.declare_parameter(
            'process_image_urls',
            False,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description='If true, processes image_url in JSON prompts.'
            )
        )
        self.declare_parameter(
            'response_format',
            os.environ.get('LLM_RESPONSE_FORMAT', ''),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='JSON string defining the output format.'
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
        """Populate the initial chat history from ROS parameters."""
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
        """Load and configure the LLM client based on ROS parameters."""
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
            except Exception:
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
                api_url=self.get_parameter('api_url').value,
                api_key=self.get_parameter('api_key').value,
                model=self.get_parameter('api_model').value,
                logger=self.get_logger(),
                temperature=self.get_parameter('temperature').value,
                top_p=self.get_parameter('top_p').value,
                max_tokens=self.get_parameter('max_tokens').value,
                stop=stop,
                presence_penalty=self.get_parameter('presence_penalty').value,
                frequency_penalty=self.get_parameter('frequency_penalty').value,
                timeout=self.get_parameter('api_timeout').value,
                response_format=response_format
            )
        else:
            self.get_logger().error(f"Unsupported API type: {api_type}")

    def _load_tools(self) -> (list, dict):
        """
        Dynamically load tool modules specified in 'tool_interfaces'.

        :return: A tuple containing (all_tools, all_functions).
        """
        try:
            tool_modules_paths = self.get_parameter('tool_interfaces').value
        except Exception:
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
        Process the latest conversational turn for publishing and logging.

        :param user_prompt: The string content of the user's latest prompt.
        :param assistant_message: The final message dictionary from the assistant.
        """
        try:
            user_msg = {"role": "user", "content": user_prompt}
            latest_turn_list = [user_msg, assistant_message]
            json_string = json.dumps(latest_turn_list)
            self.pub_latest_turn.publish(String(data=json_string))
        except TypeError as e:
            self.get_logger().error(f"Failed to serialize latest turn to JSON: {e}")

        # --- Log to file ---
        log_file_path = self.get_parameter('message_log').value
        if not log_file_path:
            return

        try:
            log_data = []
            if os.path.exists(log_file_path) and os.path.getsize(log_file_path) > 0:
                with open(log_file_path, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
                if not isinstance(log_data, list):
                    self.get_logger().warning(
                        f"Log file '{log_file_path}' contained invalid data. Overwriting.")
                    log_data = []

            if not log_data:
                system_prompt = self.get_parameter('system_prompt').value
                if system_prompt:
                    log_data.append({"role": "system", "content": system_prompt})

            log_data.append({"role": "user", "content": user_prompt})
            log_data.append(assistant_message)

            with open(log_file_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2)

        except (IOError, json.JSONDecodeError) as e:
            self.get_logger().error(f"Error processing message log file '{log_file_path}': {e}")

    def _get_truncated_history(self):
        """Return a copy of chat history with long strings truncated."""
        truncated_history = []
        for msg in self.chat_history:
            msg_copy = msg.copy()
            if isinstance(msg_copy.get("content"), list):
                new_content = []
                for part in msg_copy["content"]:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        part_copy = part.copy()
                        if "image_url" in part_copy and "url" in part_copy["image_url"]:
                            url = part_copy["image_url"]["url"]
                            if len(url) > 100:
                                part_copy["image_url"] = {
                                    "url": f"{url[:30]}...<truncated>...{url[-30:]}"
                                }
                        new_content.append(part_copy)
                    else:
                        new_content.append(part)
                msg_copy["content"] = new_content
            truncated_history.append(msg_copy)
        return truncated_history

    def _trim_chat_history(self):
        """
        Prevent the chat history from growing indefinitely.

        It trims the oldest conversational turns to stay within the limit.
        """
        max_len = self.get_parameter('max_history_length').value
        prefix = self.chat_history[:self._prefix_history_len]
        conversation = self.chat_history[self._prefix_history_len:]
        user_message_indices = [
            i for i, msg in enumerate(conversation) if msg['role'] == 'user']
        num_turns = len(user_message_indices)

        if num_turns > max_len:
            first_turn_to_keep_idx = user_message_indices[num_turns - max_len]
            trimmed_conversation = conversation[first_turn_to_keep_idx:]
            self.chat_history = prefix + trimmed_conversation
            self.get_logger().info(
                f"Trimmed {num_turns - max_len} old turn(s) from chat history.")

        self.get_logger().debug(f"History: {str(self._get_truncated_history())}")

    def prompt_callback(self, msg):
        """Process an incoming prompt from the 'llm_prompt' topic."""
        # --- Cancellation Check ---
        stop_list = self.get_parameter('stop').value
        if msg.data in stop_list:
            if self._is_generating:
                self.get_logger().warn(f"Cancellation requested: '{msg.data}'")
                self._cancel_requested = True
            else:
                self.get_logger().info(f"Stop command '{msg.data}' received.")
            return

        # --- Busy Check ---
        if self._is_generating:
            self._prompt_queue.append(msg.data)
            self.get_logger().info(f"Queued prompt. Queue size: {len(self._prompt_queue)}")
            if self._queue_timer is None:
                self._queue_timer = self.create_timer(
                    self._queue_timer_period,
                    self._process_queue_callback,
                    callback_group=ReentrantCallbackGroup()
                )
            return

        self._is_generating = True
        self._cancel_requested = False

        try:
            if not self.llm_client:
                self.get_logger().error("LLM client not available.")
                return

            self.get_logger().info("Prompt received")
            prompt_text_for_log = msg.data
            new_messages = []

            try:
                json_data = json.loads(msg.data)
                if isinstance(json_data, list):
                    new_messages = json_data
                    for m in reversed(new_messages):
                        if isinstance(m, dict) and m.get("role") == "user":
                            c = m.get("content", "")
                            if isinstance(c, str):
                                prompt_text_for_log = c
                            break
                elif isinstance(json_data, dict):
                    user_content = json_data
                    if json_data.get("role") == "user":
                        c = json_data.get("content", "")
                        if isinstance(c, str):
                            prompt_text_for_log = c

                        process_img = self.get_parameter('process_image_urls').value
                        if process_img and "image_url" in json_data:
                            image_url = json_data["image_url"]
                            try:
                                image_data = None
                                if image_url.startswith("file://"):
                                    file_path = image_url[7:]
                                    with open(file_path, "rb") as image_file:
                                        mime_type, _ = mimetypes.guess_type(file_path)
                                        if not mime_type:
                                            mime_type = "image/jpeg"
                                        b64 = base64.b64encode(image_file.read()).decode('utf-8')
                                        image_data = f"data:{mime_type};base64,{b64}"
                                elif image_url.startswith("http"):
                                    import requests
                                    response = requests.get(image_url, timeout=10.0)
                                    response.raise_for_status()
                                    m_t = response.headers.get('Content-Type', 'image/jpeg')
                                    b64 = base64.b64encode(response.content).decode('utf-8')
                                    image_data = f"data:{m_t};base64,{b64}"

                                if image_data:
                                    user_content = {
                                        "role": "user",
                                        "content": [
                                            {"type": "text", "text": json_data.get("content", "")},
                                            {"type": "image_url", "image_url": {"url": image_data}}
                                        ]
                                    }
                            except Exception as e:
                                self.get_logger().error(f"Image processing failed: {e}")

                    new_messages = [user_content]
                else:
                    new_messages = [{"role": "user", "content": str(json_data)}]
                    prompt_text_for_log = str(json_data)
            except json.JSONDecodeError:
                new_messages = [{"role": "user", "content": msg.data}]
                prompt_text_for_log = msg.data

            self.chat_history.extend(new_messages)
            self._trim_chat_history()

            stream_enabled = self.get_parameter('stream').value
            max_calls = self.get_parameter('max_tool_calls').value
            tool_call_count = 0

            while tool_call_count < max_calls:
                if self._cancel_requested:
                    return

                success, response_message = self.llm_client.process_prompt(
                    self.chat_history,
                    self.tools if self.tools else None
                )

                if not success:
                    self.get_logger().error(f"LLM request error: {response_message}")
                    self.chat_history.pop()
                    return

                if response_message.get("tool_calls"):
                    if response_message.get("content") is None:
                        response_message["content"] = ""
                    self.chat_history.append(response_message)
                    tool_call_count += 1
                    for tool_call in response_message["tool_calls"]:
                        if self._cancel_requested:
                            return

                        func_name = tool_call['function']['name']
                        tool_call_id = tool_call['id']
                        func_to_call = self.tool_functions.get(func_name)

                        if not func_to_call:
                            err = f"Tool '{func_name}' not found."
                            self.get_logger().error(err)
                            self.chat_history.append({
                                "tool_call_id": tool_call_id, "role": "tool",
                                "name": func_name, "content": err})
                            continue

                        try:
                            args = json.loads(tool_call['function']['arguments'])
                            self.get_logger().info(f"Calling tool '{func_name}'")
                            result = func_to_call(**args)
                            c_str = (result if isinstance(result, str)
                                     else json.dumps(result, ensure_ascii=False))
                            self.chat_history.append({
                                "tool_call_id": tool_call_id, "role": "tool",
                                "name": func_name, "content": c_str})
                        except Exception as e:
                            self.get_logger().error(f"Tool {func_name} failed: {e}")
                            self.chat_history.append({
                                "tool_call_id": tool_call_id, "role": "tool",
                                "name": func_name, "content": str(e)})
                    continue
                else:
                    break

            if tool_call_count >= max_calls:
                err = f"Max tool calls ({max_calls}) reached."
                self.get_logger().warning(err)
                self.pub_response.publish(String(data="Too many tool calls."))
                return

            if self._cancel_requested:
                return

            if stream_enabled:
                full_response = ""
                for chunk in self.llm_client.stream_prompt(self.chat_history):
                    if self._cancel_requested:
                        self.pub_response.publish(String(data="[Cancelled]"))
                        return
                    if chunk:
                        full_response += chunk
                        self.pub_stream.publish(String(data=chunk))

                self.pub_response.publish(String(data=full_response))
                assistant_message = {"role": "assistant", "content": full_response}
                self.chat_history.append(assistant_message)
                self._publish_latest_turn(prompt_text_for_log, assistant_message)
            else:
                success, final_message = self.llm_client.process_prompt(self.chat_history)
                if self._cancel_requested:
                    return
                if success and final_message.get("content"):
                    res_text = final_message["content"]
                    self.pub_response.publish(String(data=res_text))
                    self.chat_history.append(final_message)
                    self._publish_latest_turn(prompt_text_for_log, final_message)
                else:
                    self.get_logger().error(f"Failed response: {final_message}")
                    self.pub_response.publish(String(data="Error generating response."))
        finally:
            self._is_generating = False

    def _process_queue_callback(self):
        """Timer callback to process queued prompts."""
        if self._is_generating:
            return
        if not self._prompt_queue:
            if self._queue_timer is not None:
                self._queue_timer.cancel()
                self._queue_timer = None
            return
        prompt_data = self._prompt_queue.popleft()
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
        llm_node.get_logger().info("Shutting down node...")
    finally:
        llm_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
