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
from collections import deque
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError
)
import importlib
import importlib.util
import json
import logging
import mimetypes
import os

from ament_index_python.packages import get_package_share_directory
from bob_llm.backend_clients import OpenAICompatibleClient
from bob_llm.tool_utils import register as default_register
from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import ParameterType
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.logging import LoggingSeverity
from rclpy.node import Node
import requests
from std_msgs.msg import String


class LLMNode(Node):
    """
    ROS 2 node that provides an interface to LLMs and VLMs.

    This node handles prompts, manages conversation history, and executes tools.
    """

    def _declare_param(self, name, env_name, default, param_type, description):
        """Declare a ROS parameter with an environment variable fallback."""
        env_value = os.environ.get(env_name)
        if env_value is not None:
            try:
                if param_type == ParameterType.PARAMETER_INTEGER:
                    final_value = int(env_value)
                elif param_type == ParameterType.PARAMETER_DOUBLE:
                    final_value = float(env_value)
                elif param_type == ParameterType.PARAMETER_BOOL:
                    final_value = env_value.lower() in ('true', '1', 'yes')
                elif param_type == ParameterType.PARAMETER_STRING_ARRAY:
                    final_value = env_value.split(',')
                else:
                    final_value = env_value
            except ValueError:
                self.get_logger().warn(
                    f"Environment variable '{env_name}' has invalid value '{env_value}' "
                    f'for type {param_type}. Using default: {default}'
                )
                final_value = default
        else:
            final_value = default

        # Append [ENV: ...] tag to description if not already present
        env_tag = f'[ENV: {env_name}]'
        if env_tag not in description:
            description = f'{description} {env_tag}'

        self.declare_parameter(
            name,
            final_value,
            ParameterDescriptor(type=param_type, description=description)
        )
        return self.get_parameter(name).value

    def __init__(self):
        super().__init__('llm')
        self.llm_client = None

        # Synchronize logging level with ROS logger verbosity for library output.
        logging.basicConfig(
            level=(logging.DEBUG
                   if self.get_logger().get_effective_level()
                   == LoggingSeverity.DEBUG
                   else logging.INFO),
            format='[%(levelname)s] [%(asctime)s.] [%(name)s]: %(message)s',
            datefmt='%s')

        self.get_logger().info('LLM Node starting up...')

        # ROS parameters

        self._declare_param(
            'api_type', 'LLM_API_TYPE', 'openai_compatible',
            ParameterType.PARAMETER_STRING,
            'The type of the LLM backend API (e.g., "openai_compatible").')

        self._declare_param(
            'api_url', 'LLM_API_URL', 'http://localhost:8000/v1',
            ParameterType.PARAMETER_STRING,
            'The base URL of the LLM backend API. '
            'The node appends "/chat/completions" automatically.')

        self._declare_param(
            'api_key', 'LLM_API_KEY', 'no_key',
            ParameterType.PARAMETER_STRING,
            'The API key for authentication with the LLM backend.')

        self._declare_param(
            'api_model', 'LLM_API_MODEL', '',
            ParameterType.PARAMETER_STRING,
            "The specific model name to use (e.g., 'gpt-4', 'llama3').")

        self._declare_param(
            'system_prompt', 'LLM_SYSTEM_PROMPT', '',
            ParameterType.PARAMETER_STRING,
            'The system prompt to set the LLM context. '
            'If this is a valid file path, the content of the file will be used.')

        self._declare_param(
            'system_prompt_file', 'LLM_SYSTEM_PROMPT_FILE', '',
            ParameterType.PARAMETER_STRING,
            'Path to a file containing the system prompt. '
            'Takes precedence over system_prompt.')

        self._declare_param(
            'initial_messages_json', 'LLM_INITIAL_MESSAGES_JSON', '[]',
            ParameterType.PARAMETER_STRING,
            'A JSON string of initial messages for few-shot prompting.')

        self._declare_param(
            'max_history_length', 'LLM_MAX_HISTORY_LENGTH', 10,
            ParameterType.PARAMETER_INTEGER,
            'Maximum number of conversational turns to keep in history.')

        self._declare_param(
            'stream', 'LLM_STREAM', True,
            ParameterType.PARAMETER_BOOL,
            'Enable or disable streaming for the final LLM response.')

        self._declare_param(
            'max_tool_calls', 'LLM_MAX_TOOL_CALLS', 5,
            ParameterType.PARAMETER_INTEGER,
            'Maximum number of consecutive tool calls before aborting.')

        self._declare_param(
            'temperature', 'LLM_TEMPERATURE', 0.7,
            ParameterType.PARAMETER_DOUBLE,
            'Controls the randomness of the output.')

        self._declare_param(
            'top_p', 'LLM_TOP_P', 1.0,
            ParameterType.PARAMETER_DOUBLE,
            'Nucleus sampling diversity control.')

        self._declare_param(
            'max_tokens', 'LLM_MAX_TOKENS', 0,
            ParameterType.PARAMETER_INTEGER,
            'Maximum number of tokens to generate. 0 means no limit.')

        self._declare_param(
            'stop', 'LLM_STOP', ['stop_llm'],
            ParameterType.PARAMETER_STRING_ARRAY,
            'A list of sequences to stop generation at.')

        self._declare_param(
            'presence_penalty', 'LLM_PRESENCE_PENALTY', 0.0,
            ParameterType.PARAMETER_DOUBLE,
            'Penalizes new tokens based on presence.')

        self._declare_param(
            'frequency_penalty', 'LLM_FREQUENCY_PENALTY', 0.0,
            ParameterType.PARAMETER_DOUBLE,
            'Penalizes new tokens based on frequency.')

        self._declare_param(
            'api_timeout', 'LLM_API_TIMEOUT', 120.0,
            ParameterType.PARAMETER_DOUBLE,
            'Timeout in seconds for API requests.')

        self._declare_param(
            'tool_interfaces', 'LLM_TOOL_INTERFACES', [''],
            ParameterType.PARAMETER_STRING_ARRAY,
            'A list of Python modules or file paths to load as tools.')

        self._declare_param(
            'message_log', 'LLM_MESSAGE_LOG', '',
            ParameterType.PARAMETER_STRING,
            'If set, appends conversation turns to this JSON file.')

        self._declare_param(
            'process_image_urls', 'LLM_PROCESS_IMAGE_URLS', False,
            ParameterType.PARAMETER_BOOL,
            'If true, processes image_url in JSON prompts.')

        self._declare_param(
            'response_format', 'LLM_RESPONSE_FORMAT', '',
            ParameterType.PARAMETER_STRING,
            'JSON string defining the output format.')

        self._declare_param(
            'eof', 'LLM_EOF', '',
            ParameterType.PARAMETER_STRING,
            'Optional string to publish on llm_stream when '
            'generation is finished.')

        self._declare_param(
            'tool_choice', 'LLM_TOOL_CHOICE', 'auto',
            ParameterType.PARAMETER_STRING,
            "Tool calling behavior ('auto', 'none', 'required').")

        self._declare_param(
            'tool_timeout', 'LLM_TOOL_TIMEOUT', 60.0,
            ParameterType.PARAMETER_DOUBLE,
            'Maximum time in seconds to wait for a tool to execute.')

        share_dir = get_package_share_directory('bob_llm')
        default_skill_dir = os.path.join(share_dir, 'config', 'skills')

        self._declare_param(
            'skill_dir', 'LLM_SKILL_DIR', default_skill_dir,
            ParameterType.PARAMETER_STRING,
            'Directory where skills are stored.')

        # Cloak API Key: Read from parameter, store in private variable, and clear parameter.
        self._api_key = self.get_parameter('api_key').value
        if self._api_key and self._api_key != 'no_key':
            new_param = rclpy.parameter.Parameter('api_key', rclpy.Parameter.Type.STRING, '')
            self.set_parameters([new_param])
            self.get_logger().info('API key has been cloaked.')

        self.chat_history = []
        self._user_turns_count = 0
        self.load_llm_client()
        self._initialize_chat_history()
        self._prefix_history_len = len(self.chat_history)

        # Load tools and their corresponding functions
        self.tools, self.tool_functions = self._load_tools()
        if self.tools:
            self.get_logger().info(
                f'Successfully loaded {len(self.tools)} tools.')

        DEFAULT_QUEUE_SIZE = int(os.environ.get('LLM_QUEUE_SIZE', '1000'))

        self._is_generating = False
        self._cancel_requested = False
        self._prompt_queue = deque()
        self._stream_buffer = ''
        self._executor = ThreadPoolExecutor(max_workers=5)

        self.sub = self.create_subscription(
            String, 'llm_prompt', self.prompt_callback, DEFAULT_QUEUE_SIZE,
            callback_group=ReentrantCallbackGroup())

        self.pub_response = self.create_publisher(
            String, 'llm_response', DEFAULT_QUEUE_SIZE)

        self.pub_stream = self.create_publisher(
            String, 'llm_stream', DEFAULT_QUEUE_SIZE)

        self.pub_reasoning = self.create_publisher(
            String, 'llm_reasoning', DEFAULT_QUEUE_SIZE)

        self.pub_latest_turn = self.create_publisher(
            String, 'llm_latest_turn', DEFAULT_QUEUE_SIZE)
        self.pub_tool_calls = self.create_publisher(
            String, 'llm_tool_calls', DEFAULT_QUEUE_SIZE)

        # Register parameter update callback
        self.add_on_set_parameters_callback(self.on_params_changed)

        self.get_logger().info(
            f'Node is ready. History has {len(self.chat_history)} initial messages.')

    def _load_system_prompt(self):
        """Load the system prompt from parameter or file."""
        system_prompt_file = self.get_parameter('system_prompt_file').value
        system_prompt = self.get_parameter('system_prompt').value

        # Priority 1: system_prompt_file
        if system_prompt_file and os.path.isfile(system_prompt_file):
            try:
                with open(system_prompt_file, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            except Exception as e:
                self.get_logger().error(
                    f"Failed to read system_prompt_file '{system_prompt_file}': {e}")

        # Priority 2: system_prompt (check if it's a file path)
        if system_prompt and os.path.isfile(system_prompt):
            try:
                with open(system_prompt, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            except Exception as e:
                self.get_logger().error(
                    f"Failed to read system_prompt as file '{system_prompt}': {e}")

        # Priority 3: system_prompt as string
        return system_prompt

    def _initialize_chat_history(self):
        """Populate the initial chat history from ROS parameters."""
        system_prompt = self._load_system_prompt()
        if system_prompt:
            self.chat_history.append({'role': 'system', 'content': system_prompt})
            self.get_logger().info('System prompt added.')
        initial_messages_str = self.get_parameter('initial_messages_json').value
        try:
            initial_messages = json.loads(initial_messages_str)
            if isinstance(initial_messages, list):
                self.chat_history.extend(initial_messages)
                self._user_turns_count = sum(
                    1 for m in initial_messages if m.get('role') == 'user')
                self.get_logger().info(
                    f'Loaded {len(initial_messages)} messages '
                    f'({self._user_turns_count} user turns).')
                # Log the last message for context
                if initial_messages:
                    last_msg = initial_messages[-1]
                    text = self._get_message_text(last_msg.get('content', ''))
                    self.get_logger().info(f'Last message: {text[:100]}...')
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

        self.get_logger().info(
            f'Connecting to LLM at {api_url} with model {model}')

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
                    self.get_logger().info(f'Using response_format: {response_format}')
                except json.JSONDecodeError as e:
                    self.get_logger().error(
                        f"Failed to parse 'response_format' JSON: {e}")

            self.llm_client = OpenAICompatibleClient(
                api_url=self.get_parameter('api_url').value,
                api_key=self._api_key,
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
            self.get_logger().error(f'Unsupported API type: {api_type}')

    def _load_tools(self) -> tuple:
        """
        Dynamically load tool modules specified in 'tool_interfaces'.

        :return: A tuple containing (all_tools, all_functions).
        """
        try:
            tool_modules_paths = self.get_parameter('tool_interfaces').value
        except Exception:
            return [], {}

        self.get_logger().info(
            f'Loading tool interfaces: {tool_modules_paths}')

        all_tools = []
        all_functions = {}
        for path_str in tool_modules_paths:
            if not path_str:
                continue
            try:
                # Check if the path is a file, otherwise treat it as a module
                if path_str.endswith('.py') and os.path.exists(path_str):
                    module_name = os.path.splitext(os.path.basename(path_str))[0]
                    spec = importlib.util.spec_from_file_location(module_name, path_str)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    self.get_logger().debug(
                        f'Imported module from file: {path_str}')
                else:
                    module = importlib.import_module(path_str)
                    self.get_logger().info(f'Imported module by name: {path_str}')

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
                self.get_logger().error(f'Failed to import tool module {path_str}: {e}')
            except Exception as e:
                self.get_logger().error(
                    f'Error loading tools from {path_str}: {e}')

        return all_tools, all_functions

    def _publish_latest_turn(self, user_prompt: str, assistant_message: dict):
        """
        Process the latest conversational turn for publishing and logging.

        :param user_prompt: The string content of the user's latest prompt.
        :param assistant_message: The final message dictionary from the assistant.
        """
        try:
            user_msg = {'role': 'user', 'content': user_prompt}
            latest_turn_list = [user_msg, assistant_message]
            json_string = json.dumps(latest_turn_list)
            self.pub_latest_turn.publish(String(data=json_string))
        except TypeError as e:
            self.get_logger().error(f'Failed to serialize latest turn to JSON: {e}')

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
                    log_data.append({'role': 'system', 'content': system_prompt})

            log_data.append({'role': 'user', 'content': user_prompt})
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
            if isinstance(msg_copy.get('content'), list):
                new_content = []
                for part in msg_copy['content']:
                    if isinstance(part, dict) and part.get('type') == 'image_url':
                        part_copy = part.copy()
                        if 'image_url' in part_copy and 'url' in part_copy['image_url']:
                            url = part_copy['image_url']['url']
                            if len(url) > 100:
                                part_copy['image_url'] = {
                                    'url': f'{url[:30]}...<truncated>...{url[-30:]}'
                                }
                        new_content.append(part_copy)
                    else:
                        new_content.append(part)
                msg_copy['content'] = new_content
            truncated_history.append(msg_copy)
        return truncated_history

    def _trim_chat_history(self):
        """
        Prevent the chat history from growing indefinitely.

        Trims oldest turns after the prefix to stay within max_history_length.
        """
        max_len = self.get_parameter('max_history_length').value
        if max_len <= 0 or self._user_turns_count <= max_len:
            return

        prefix = self.chat_history[:self._prefix_history_len]
        conversation = self.chat_history[self._prefix_history_len:]

        # Find the N-th user message from the back
        user_turns_found = 0
        trim_at = 0

        for i in range(len(conversation) - 1, -1, -1):
            if conversation[i]['role'] == 'user':
                user_turns_found += 1
                if user_turns_found == max_len:
                    trim_at = i
                    break

        if user_turns_found >= max_len:
            trimmed_conversation = conversation[trim_at:]
            self.chat_history = prefix + trimmed_conversation
            self.get_logger().info(
                f'Trimmed {self._user_turns_count - max_len} old turn(s) from chat history.')
            self._user_turns_count = max_len

    def _get_message_text(self, content):
        """Extract text from message content (string or multimodal list)."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get('type') == 'text':
                        texts.append(item.get('text', ''))
                    elif item.get('type') == 'image_url':
                        texts.append('[Image]')
            return ' '.join(texts).strip()
        return str(content)

    def _process_image_url(self, image_url, text_content):
        """Process an image URL (file or http) and return a multimodal message content."""
        try:
            image_data = None
            if image_url.startswith('file://'):
                file_path = image_url[7:]
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as image_file:
                        mime_type, _ = mimetypes.guess_type(file_path)
                        mime_type = mime_type or 'image/jpeg'
                        b64 = base64.b64encode(image_file.read()).decode('utf-8')
                        image_data = f'data:{mime_type};base64,{b64}'
            elif image_url.startswith('http'):
                response = requests.get(image_url, timeout=10.0)
                response.raise_for_status()
                mime_type = response.headers.get('Content-Type', 'image/jpeg')
                b64 = base64.b64encode(response.content).decode('utf-8')
                image_data = f'data:{mime_type};base64,{b64}'

            if image_data:
                return [
                    {'type': 'text', 'text': text_content},
                    {'type': 'image_url', 'image_url': {'url': image_data}}
                ]
        except Exception as e:
            self.get_logger().error(f'Image processing failed for {image_url}: {e}')

        return text_content

    def _prepare_messages(self, raw_data):
        """Parse raw input data and return (new_messages, prompt_text_for_log)."""
        prompt_text_for_log = raw_data
        new_messages = []

        try:
            json_data = json.loads(raw_data)
            if isinstance(json_data, list):
                new_messages = json_data
                for m in reversed(new_messages):
                    if isinstance(m, dict) and m.get('role') == 'user':
                        c = m.get('content', '')
                        if isinstance(c, str):
                            prompt_text_for_log = c
                        break
            elif isinstance(json_data, dict):
                if 'role' in json_data:
                    user_content = json_data
                else:
                    user_content = {'role': 'user', 'content': raw_data}

                # Extract log text
                c = user_content.get('content', '')
                if isinstance(c, str):
                    prompt_text_for_log = c

                # Image processing
                process_img = self.get_parameter('process_image_urls').value
                if process_img and 'image_url' in json_data:
                    user_content['content'] = self._process_image_url(
                        json_data['image_url'], json_data.get('content', '')
                    )

                new_messages = [user_content]
            else:
                new_messages = [{'role': 'user', 'content': str(json_data)}]
                prompt_text_for_log = str(json_data)
        except json.JSONDecodeError:
            new_messages = [{'role': 'user', 'content': raw_data}]

        # Update prompt_text_for_log based on the actual new messages
        if new_messages:
            # Find the last user message in the new batch
            for m in reversed(new_messages):
                if m.get('role') == 'user':
                    prompt_text_for_log = self._get_message_text(m.get('content', ''))
                    break

        # Increment user turn count for new messages
        for m in new_messages:
            if m.get('role') == 'user':
                self._user_turns_count += 1

        return new_messages, prompt_text_for_log

    def _execute_tool_calls(self, tool_calls):
        """
        Execute a list of tool calls and append results to history.

        Returns True if at least one tool call succeeded.
        """
        any_success = False
        for tool_call in tool_calls:
            if self._cancel_requested:
                return

            func_name = tool_call['function']['name']
            tool_call_id = tool_call['id']
            args_raw = tool_call['function']['arguments']
            func_to_call = self.tool_functions.get(func_name)

            if not func_to_call:
                err = f"Tool '{func_name}' not found."
                self.get_logger().error(err)
                self.chat_history.append({
                    'tool_call_id': tool_call_id,
                    'role': 'tool',
                    'name': func_name,
                    'content': err
                })
                continue

            try:
                args = json.loads(args_raw)
                self.get_logger().info(f"Calling tool '{func_name}' with args: {args_raw}")
                self.pub_tool_calls.publish(String(data=json.dumps({
                    'name': func_name,
                    'arguments': args_raw,
                    'id': tool_call_id
                })))

                # Execute tool with timeout
                timeout = self.get_parameter('tool_timeout').value
                future = self._executor.submit(func_to_call, **args)
                try:
                    result = future.result(timeout=timeout)
                    content_str = (result if isinstance(result, str)
                                   else json.dumps(result, ensure_ascii=False, default=str))
                except FutureTimeoutError:
                    content_str = f"Error: Tool '{func_name}' timed out after {timeout}s."
                    self.get_logger().error(content_str)

                if not content_str.startswith('Error:'):
                    any_success = True

                self.chat_history.append({
                    'tool_call_id': tool_call_id,
                    'role': 'tool',
                    'name': func_name,
                    'content': content_str
                })
            except Exception as e:
                err_msg = f'Error: Tool execution failed: {e}'
                self.get_logger().error(err_msg)
                self.chat_history.append({
                    'tool_call_id': tool_call_id,
                    'role': 'tool',
                    'name': func_name,
                    'content': err_msg
                })

        return any_success

    def _generate_stream(self, tool_choice):
        """
        Process LLM generation in streaming mode.

        Returns (resp, reasoning, tools).
        """
        full_response = ''
        full_reasoning = ''
        tool_calls_chunks = {}  # index -> {id, name, arguments_str}

        retry_count = 0
        max_retries = 3  # Increased for better initial connection robustness

        while retry_count <= max_retries:
            try:
                async_stream = self.llm_client.process_prompt_stream(
                    self.chat_history,
                    tools=self.tools if self.tools else None,
                    tool_choice=tool_choice
                )

                for chunk in async_stream:
                    if self._cancel_requested:
                        return None, None, None

                    if isinstance(chunk, dict):
                        content = chunk.get('content')
                        reasoning = chunk.get('reasoning')
                        t_calls = chunk.get('tool_calls')

                        if reasoning is not None:
                            full_reasoning += reasoning
                            self.pub_reasoning.publish(String(data=reasoning))
                        if content is not None:
                            full_response += content
                            self._stream_buffer += content

                            # Publish if we hit a boundary (space, punctuation, newline)
                            # or buffer gets too long (e.g. 15 chars)
                            if any(c in content for c in ' \t\n.,!?;:') or \
                                    len(self._stream_buffer) > 15:
                                self.pub_stream.publish(String(data=self._stream_buffer))
                                self._stream_buffer = ''

                        if t_calls:
                            for tc in t_calls:
                                idx = tc.get('index', 0)
                                if idx not in tool_calls_chunks:
                                    tool_calls_chunks[idx] = {
                                        'id': '', 'name': '', 'args': ''}
                                if tc.get('id'):
                                    tool_calls_chunks[idx]['id'] += tc['id']
                                if tc.get('function'):
                                    f = tc['function']
                                    if f.get('name'):
                                        tool_calls_chunks[idx]['name'] += f['name']
                                    if f.get('arguments'):
                                        tool_calls_chunks[idx]['args'] += f['arguments']

                    elif isinstance(chunk, str):
                        if chunk.startswith('[ERROR:'):
                            # If we haven't received anything yet, we can retry
                            should_retry = not full_response and not full_reasoning \
                                and retry_count < max_retries
                            if should_retry:
                                self.get_logger().warn(
                                    f'Stream error, retrying ({retry_count + 1}/{max_retries})...')
                                break
                            self.get_logger().error(chunk)
                            # Append error to response if it's already mid-stream
                            if full_response:
                                full_response += f'\n\n{chunk}'
                        else:
                            full_response += chunk
                            self.pub_stream.publish(String(data=chunk))

                # Success or non-retriable error
                break

            except Exception as e:
                self.get_logger().error(f'Critical stream processing error: {e}')
                if not full_response and not full_reasoning and retry_count < max_retries:
                    retry_count += 1
                    continue
                break
            finally:
                retry_count += 1

        # Convert chunks to standard tool_calls format
        tool_calls = []
        for idx in sorted(tool_calls_chunks.keys()):
            tc = tool_calls_chunks[idx]
            tool_calls.append({
                'id': tc['id'],
                'type': 'function',
                'function': {
                    'name': tc['name'],
                    'arguments': tc['args']
                }
            })

        # Final flush of stream buffer
        if self._stream_buffer:
            self.pub_stream.publish(String(data=self._stream_buffer))
            self._stream_buffer = ''

        return full_response, full_reasoning, tool_calls

    def _generate_sync(self, tool_choice):
        """Process LLM generation in synchronous mode. Returns (resp, reasoning, tools)."""
        success, response_message = self.llm_client.process_prompt(
            self.chat_history,
            self.tools if self.tools else None,
            tool_choice=tool_choice
        )

        if not success:
            self.get_logger().error(f'LLM request error: {response_message}')
            return None, None, None

        full_response = response_message.get('content', '')
        full_reasoning = (response_message.get('reasoning_content') or
                          response_message.get('reasoning') or '')
        tool_calls = response_message.get('tool_calls', [])

        return full_response, full_reasoning, tool_calls

    def prompt_callback(self, msg):
        """Handle incoming prompts via a non-blocking queue."""
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
            # Safety: Keep only the latest prompts to avoid massive backlog
            if len(self._prompt_queue) > 5:
                self.get_logger().warn('Prompt queue too long, dropping oldest message.')
                self._prompt_queue.popleft()

            self._prompt_queue.append(msg.data)
            self.get_logger().info(f'Queued prompt. Queue size: {len(self._prompt_queue)}')
            return

        self._is_generating = True
        self._cancel_requested = False

        try:
            current_prompt_data = msg.data

            while current_prompt_data:
                if not self.llm_client:
                    self.get_logger().error('LLM client not available.')
                    break

                self.get_logger().info('Processing prompt...')

                # 1. Prepare messages and history
                new_messages, prompt_text_for_log = self._prepare_messages(current_prompt_data)
                self.chat_history.extend(new_messages)
                self._trim_chat_history()

                # 2. Main Generation Loop
                stream_enabled = self.get_parameter('stream').value
                max_calls = self.get_parameter('max_tool_calls').value
                tool_choice = self.get_parameter('tool_choice').value
                tool_call_count = 0
                consecutive_errors = 0

                while tool_call_count < max_calls:
                    if self._cancel_requested or consecutive_errors > 3:
                        if consecutive_errors > 3:
                            self.get_logger().error('Too many consecutive tool errors. Aborting.')
                        break

                    if stream_enabled:
                        full_response, full_reasoning, tool_calls = \
                            self._generate_stream(tool_choice)
                    else:
                        full_response, full_reasoning, tool_calls = \
                            self._generate_sync(tool_choice)

                    if full_response is None and not tool_calls:
                        # Error or cancellation in generation
                        break

                    # Update history with assistant turn
                    assistant_message = {
                        'role': 'assistant',
                        'content': full_response,
                        'reasoning_content': full_reasoning
                    }
                    if tool_calls:
                        assistant_message['tool_calls'] = tool_calls

                    self.chat_history.append(assistant_message)

                    if not tool_calls:
                        # Final turn (no tools)
                        eof_str = self.get_parameter('eof').value
                        if eof_str:
                            self.pub_stream.publish(String(data=eof_str))

                        if not stream_enabled:
                            # In sync mode, we must publish reasoning/response now
                            if full_reasoning:
                                self.pub_reasoning.publish(String(data=full_reasoning))
                            if full_response:
                                self.pub_response.publish(String(data=full_response))
                        else:
                            # In stream mode, final full response is published here
                            self.pub_response.publish(String(data=full_response))

                        self._publish_latest_turn(prompt_text_for_log, assistant_message)
                        break

                    # Process Tool Calls
                    success = self._execute_tool_calls(tool_calls)
                    if success:
                        tool_call_count += 1
                        consecutive_errors = 0
                    else:
                        consecutive_errors += 1
                    # Continue loop to get next model turn after tool results

                if tool_call_count >= max_calls:
                    self.get_logger().warning(f'Max tool calls ({max_calls}) reached.')

                # Check if there is another prompt in the queue
                if self._prompt_queue and not self._cancel_requested:
                    current_prompt_data = self._prompt_queue.popleft()
                    self.get_logger().info(
                        f'Processing next queued prompt. Remaining: {len(self._prompt_queue)}')
                else:
                    current_prompt_data = None

        finally:
            self._is_generating = False

    def on_params_changed(self, params):
        """Handle parameter changes at runtime."""
        from rcl_interfaces.msg import SetParametersResult
        from rclpy.parameter import Parameter

        result = SetParametersResult(successful=True)
        system_prompt_updated = False
        client_params_updated = False

        for param in params:
            if param.name == 'stream' and param.type_ == Parameter.Type.BOOL:
                self.get_logger().info(
                    f"Streaming {'enabled' if param.value else 'disabled'}")
            elif (param.name == 'max_tool_calls' and
                    param.type_ == Parameter.Type.INTEGER):
                self.get_logger().info(
                    f'Max tool calls set to {param.value}')
            elif (param.name == 'queue_size' and
                    param.type_ == Parameter.Type.INTEGER):
                self.get_logger().info(f'Queue size set to {param.value}')
            elif param.name in ['system_prompt', 'system_prompt_file']:
                system_prompt_updated = True
            elif param.name in [
                'temperature', 'top_p', 'max_tokens', 'presence_penalty',
                'frequency_penalty', 'api_timeout', 'stop', 'api_url',
                'api_key', 'api_model', 'eof'
            ]:
                client_params_updated = True
            elif param.name == 'response_format':
                try:
                    if param.value:
                        json.loads(param.value)
                    client_params_updated = True
                except json.JSONDecodeError as e:
                    result.successful = False
                    result.reason = f'Invalid JSON for response_format: {e}'
                    return result

        if system_prompt_updated:
            new_prompt = self._load_system_prompt()
            if self.chat_history and self.chat_history[0]['role'] == 'system':
                self.chat_history[0]['content'] = new_prompt
                self.get_logger().info('System prompt updated in history.')
            else:
                self.chat_history.insert(
                    0, {'role': 'system', 'content': new_prompt})
                self._prefix_history_len += 1
                self.get_logger().info('System prompt added to history.')

        if client_params_updated and hasattr(self, 'llm_client'):
            for param in params:
                if param.name == 'temperature':
                    self.llm_client.temperature = param.value
                elif param.name == 'top_p':
                    self.llm_client.top_p = param.value
                elif param.name == 'max_tokens':
                    self.llm_client.max_tokens = param.value
                elif param.name == 'presence_penalty':
                    self.llm_client.presence_penalty = param.value
                elif param.name == 'frequency_penalty':
                    self.llm_client.frequency_penalty = param.value
                elif param.name == 'api_timeout':
                    self.llm_client.timeout = param.value
                elif param.name == 'stop':
                    self.llm_client.stop = list(param.value)
                elif param.name == 'api_url':
                    self.llm_client.api_url = (
                        param.value.rstrip('/') + '/chat/completions')
                elif param.name == 'api_model':
                    self.llm_client.model = param.value
                elif param.name == 'api_key':
                    self.llm_client.api_key = param.value
                    if param.value:
                        self.llm_client.headers['Authorization'] = (
                            f'Bearer {param.value}')
                    elif 'Authorization' in self.llm_client.headers:
                        del self.llm_client.headers['Authorization']
                elif param.name == 'response_format':
                    if param.value:
                        self.llm_client.response_format = json.loads(
                            param.value)
                    else:
                        self.llm_client.response_format = None
            self.get_logger().info('LLM client parameters updated.')

        return result

    def destroy_node(self):
        """Clean up resources before shutting down."""
        if hasattr(self, '_executor'):
            self.get_logger().info('Shutting down tool executor...')
            self._executor.shutdown(wait=False)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    llm_node = LLMNode()
    executor = MultiThreadedExecutor()
    executor.add_node(llm_node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        llm_node.get_logger().info('Shutting down node...')
    finally:
        llm_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
