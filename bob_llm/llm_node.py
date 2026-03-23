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

"""ROS 2 node for Large Language Models."""

import asyncio
import base64
from collections import deque
import importlib
import importlib.util
import json
import logging
import mimetypes
import os

from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.logging import LoggingSeverity
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.msg import SetParametersResult
import requests
from std_msgs.msg import String

from bob_llm.backend_clients import OpenAICompatibleClient
from bob_llm.tool_utils import register as default_register


class LLMNode(Node):
    """
    ROS 2 node that provides an interface to LLMs and VLMs.

    This node handles prompts, manages conversation history, and executes tools.
    """

    def __init__(self):
        super().__init__('llm_node')

        # Configure logging
        logging.basicConfig(
            level=(logging.DEBUG
                   if self.get_logger().get_effective_level()
                   == LoggingSeverity.DEBUG
                   else logging.INFO),
            format='[%(levelname)s] [%(asctime)s.] [%(name)s]: %(message)s',
            datefmt='%s')

        self.get_logger().info('LLM Node starting up...')

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
            os.environ.get('LLM_API_URL', 'http://localhost:8000/v1'),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description=(
                    'The base URL of the LLM backend API. '
                    'The node appends "/chat/completions" automatically.'
                )
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
                description="The specific model name to use (e.g., 'gpt-4', 'llama3')."
            )
        )
        self.declare_parameter(
            'system_prompt',
            os.environ.get('LLM_SYSTEM_PROMPT', ''),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description=(
                    'The system prompt to set the LLM context. '
                    'If this is a valid file path, the content of the file will be used.'
                )
            )
        )
        self.declare_parameter(
            'system_prompt_file',
            os.environ.get('LLM_SYSTEM_PROMPT_FILE', ''),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description=(
                    'Path to a file containing the system prompt. '
                    'Takes precedence over system_prompt.'
                )
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
            [''],
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
        self.declare_parameter(
            'eof',
            os.environ.get('LLM_EOF', ''),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='Optional string to publish on llm_stream when generation is finished.'
            )
        )
        self.declare_parameter(
            'tool_choice',
            os.environ.get('LLM_TOOL_CHOICE', 'auto'),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Tool calling behavior ('auto', 'none', 'required', or tool object)."
            )
        )
        share_dir = get_package_share_directory('bob_llm')
        default_skill_dir = os.path.join(share_dir, 'config', 'skills')

        self.declare_parameter(
            'skill_dir',
            os.environ.get('LLM_SKILL_DIR', default_skill_dir),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='Directory where skills are stored.'
            )
        )

        # Cloak API Key: Read from parameter, store in private variable, and clear parameter.
        self._api_key = self.get_parameter('api_key').value
        if self._api_key and self._api_key != 'no_key':
            new_param = rclpy.parameter.Parameter('api_key', rclpy.Parameter.Type.STRING, '')
            self.set_parameters([new_param])
            self.get_logger().info('API key has been cloaked.')

        self.chat_history = []
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
                self.get_logger().info(
                    f'Loaded {len(initial_messages)} initial messages from JSON.')
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

            except Exception as e:
                self.get_logger().error(
                    f"Failed to load tool interface '{path_str}': {e}")

        return all_tools, all_functions

    def on_params_changed(self, params):
        """Handle dynamic parameter updates."""
        reconfigured = False
        reinit_llm = False
        for param in params:
            if param.name in (
                'api_url', 'api_model', 'temperature', 'top_p', 'max_tokens',
                'stop', 'presence_penalty', 'frequency_penalty', 'api_timeout'
            ):
                reinit_llm = True
            if param.name == 'system_prompt':
                self.get_logger().info('System prompt update requested.')
                new_all_history = []
                new_all_history.append({'role': 'system', 'content': self._load_system_prompt()})
                # Attempt to keep user/assistant history but replace the system message
                # This is simple; we only replace the very first message if it's system.
                if self.chat_history and self.chat_history[0]['role'] == 'system':
                    self.chat_history[0]['content'] = param.value
                else:
                    # Prepend if no system prompt was there
                    self.chat_history.insert(0, {'role': 'system', 'content': param.value})
                reconfigured = True

        if reinit_llm:
            self.load_llm_client()
            reconfigured = True

        return SetParametersResult(successful=reconfigured)

    async def get_llm_response(self, messages, tools=None, tool_choice='auto'):
        """Send a request to the LLM backend and return the response."""
        if not self.llm_client:
            self.get_logger().error('LLM client not loaded.')
            return None

        return await self.llm_client.generate_response(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            stream=self.get_parameter('stream').value
        )

    def _append_to_log(self, role, content):
        """Append a message to the JSON log file if configured."""
        log_file = self.get_parameter('message_log').value
        if not log_file:
            return

        try:
            log_data = []
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    try:
                        log_data = json.load(f)
                    except json.JSONDecodeError:
                        pass

            log_data.append({
                'timestamp': self.get_clock().now().to_msg().sec,
                'role': role,
                'content': content
            })

            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2)
        except Exception as e:
            self.get_logger().error(f'Failed to write to message log: {e}')

    async def _process_tool_calls(self, response_msg):
        """Handle tool calls from the LLM response."""
        if not response_msg.tool_calls:
            return None

        # Add the assistant's request for tool calls to history
        self.chat_history.append(response_msg.to_dict())

        for tool_call in response_msg.tool_calls:
            func_name = tool_call.function_name
            arguments = tool_call.arguments
            call_id = tool_call.id

            self.get_logger().info(f"Calling tool '{func_name}' with args: {arguments}")

            if func_name in self.tool_functions:
                try:
                    # Execute the function
                    result = self.tool_functions[func_name](**arguments)

                    # Convert result to string if it isn't
                    if not isinstance(result, str):
                        result = json.dumps(result)

                    # Log a snippet of the result
                    preview = result[:100] + '...' if len(result) > 100 else result
                    self.get_logger().info(f"Tool '{func_name}' returned: {preview}")

                    # Add tool response to history
                    self.chat_history.append({
                        'role': 'tool',
                        'tool_call_id': call_id,
                        'name': func_name,
                        'content': result
                    })
                except Exception as e:
                    error_msg = f"Error executing tool '{func_name}': {e}"
                    self.get_logger().error(error_msg)
                    self.chat_history.append({
                        'role': 'tool',
                        'tool_call_id': call_id,
                        'name': func_name,
                        'content': error_msg
                    })
            else:
                error_msg = f"Tool '{func_name}' not found."
                self.get_logger().error(error_msg)
                self.chat_history.append({
                    'role': 'tool',
                    'tool_call_id': call_id,
                    'name': func_name,
                    'content': error_msg
                })

        # Return history to continue generation
        return self.chat_history

    def _publish_stream(self, text, is_final=False):
        """Publish a chunk of text to the llm_stream topic."""
        msg = String()
        msg.data = text
        self.pub_stream.publish(msg)

        if is_final:
            eof_str = self.get_parameter('eof').value
            if eof_str:
                eof_msg = String()
                eof_msg.data = eof_str
                self.pub_stream.publish(eof_msg)

    async def _handle_prompt(self, prompt_text):
        """Orchestrate the flow from prompt to final response, including tool calls."""
        if self._is_generating:
            self._prompt_queue.append(prompt_text)
            self.get_logger().info('Node is busy. Prompt queued.')
            return

        self._is_generating = True
        self._cancel_requested = False

        try:
            # Handle JSON prompts (for image support or specific roles)
            try:
                json_prompt = json.loads(prompt_text)
                if isinstance(json_prompt, dict):
                    if 'role' not in json_prompt:
                        # Wrap as user message if role is missing
                        user_msg = {'role': 'user', 'content': prompt_text}
                    else:
                        user_msg = json_prompt

                    process_img = self.get_parameter('process_image_urls').value
                    if process_img and 'image_url' in json_prompt:
                        image_url = json_prompt['image_url']
                        try:
                            image_data = None
                            if image_url.startswith('file://'):
                                file_path = image_url[7:]
                                with open(file_path, 'rb') as image_file:
                                    mime_type, _ = mimetypes.guess_type(file_path)
                                    if not mime_type:
                                        mime_type = 'image/jpeg'
                                    b64 = base64.b64encode(
                                        image_file.read()).decode('utf-8')
                                    image_data = f'data:{mime_type};base64,{b64}'
                            elif image_url.startswith('http'):
                                response = requests.get(image_url, timeout=10.0)
                                response.raise_for_status()
                                m_t = response.headers.get(
                                    'Content-Type', 'image/jpeg')
                                b64 = base64.b64encode(
                                    response.content).decode('utf-8')
                                image_data = f'data:{m_t};base64,{b64}'

                            if image_data:
                                user_msg = {
                                    'role': 'user',
                                    'content': [
                                        {'type': 'text', 'text': json_prompt.get('content', '')},
                                        {'type': 'image_url', 'image_url': {'url': image_data}}
                                    ]
                                }
                        except Exception as e:
                            self.get_logger().error(f'Image processing failed: {e}')

                else:
                    user_msg = {'role': 'user', 'content': prompt_text}
            except json.JSONDecodeError:
                user_msg = {'role': 'user', 'content': prompt_text}

            # Add user prompt to history
            self.chat_history.append(user_msg)
            self._append_to_log('user', prompt_text)

            # Trim history to max_history_length (keep system prompt)
            max_len = self.get_parameter('max_history_length').value
            if len(self.chat_history) > max_len:
                # Always keep the system prompt at index 0 and initial few-shot messages
                # We only trim messages added during the current session
                session_history = self.chat_history[self._prefix_history_len:]
                if len(session_history) > (max_len - self._prefix_history_len):
                    trimmed_session = session_history[
                        -(max_len - self._prefix_history_len):
                    ]
                    self.chat_history = (
                        self.chat_history[:self._prefix_history_len] +
                        trimmed_session
                    )

            tool_call_count = 0
            max_calls = self.get_parameter('max_tool_calls').value

            while tool_call_count < max_calls and not self._cancel_requested:
                # Request response from LLM
                response = await self.get_llm_response(
                    messages=self.chat_history,
                    tools=self.tools if self.tools else None,
                    tool_choice=self.get_parameter('tool_choice').value
                )

                if response is None:
                    break

                if response.is_streaming:
                    full_text = ""
                    async for chunk in response.stream_content:
                        if self._cancel_requested:
                            break
                        full_text += chunk
                        self._publish_stream(chunk)

                    self._publish_stream("", is_final=True)

                    # Finalize the assistant message
                    self.chat_history.append({'role': 'assistant', 'content': full_text})
                    self._append_to_log('assistant', full_text)

                    # Publish the full message
                    resp_msg = String()
                    resp_msg.data = full_text
                    self.pub_response.publish(resp_msg)
                    self.pub_latest_turn.publish(resp_msg)
                    break
                else:
                    # Non-streaming response
                    if response.tool_calls:
                        # Handle tool calls and repeat
                        await self._process_tool_calls(response)
                        tool_call_count += 1
                        continue
                    else:
                        # Final textual response
                        content = response.content
                        self.chat_history.append({'role': 'assistant', 'content': content})
                        self._append_to_log('assistant', content)

                        # Publish
                        resp_msg = String()
                        resp_msg.data = content
                        self.pub_response.publish(resp_msg)
                        self.pub_latest_turn.publish(resp_msg)
                        break

            if tool_call_count >= max_calls:
                self.get_logger().warn('Maximum number of consecutive tool calls reached.')

        except Exception as e:
            self.get_logger().error(f'Error handling prompt: {e}')
        finally:
            self._is_generating = False
            self._check_queue()

    def _check_queue(self):
        """Process the next prompt in the queue if available."""
        if self._prompt_queue and not self._is_generating:
            next_prompt = self._prompt_queue.popleft()
            # Start through a one-shot timer to stay in the main event loop
            self._queue_timer = self.create_timer(
                self._queue_timer_period,
                lambda: self._handle_queued_prompt(next_prompt),
                callback_group=ReentrantCallbackGroup()
            )

    def _handle_queued_prompt(self, prompt):
        """Handle a prompt that was taken from the queue."""
        if self._queue_timer:
            self._queue_timer.cancel()
            self._queue_timer = None

        # Use existing async infrastructure
        loop = (rclpy.get_global_executor()._loop if rclpy.get_global_executor()
                else asyncio.get_event_loop())
        asyncio.run_coroutine_threadsafe(self._handle_prompt(prompt), loop)

    def prompt_callback(self, msg):
        """Handle incoming prompts from the subscriber."""
        self.get_logger().info(f"Received prompt: '{msg.data}'")
        # Run the async handler in the executor's loop
        asyncio.ensure_future(self._handle_prompt(msg.data))


def main(args=None):
    """Run the main entry point for the LLM node."""
    rclpy.init(args=args)
    node = LLMNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
