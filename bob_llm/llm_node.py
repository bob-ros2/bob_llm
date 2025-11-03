#!/usr/bin/env python3
import os
import yaml
import json
import traceback
import importlib
import importlib.util
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import ParameterType
from rclpy.parameter import Parameter
from ament_index_python.packages import get_package_share_directory
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
        self.declare_parameter('tool_interfaces', 
            interfaces_array,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING_ARRAY,
                description='A list of Python modules or file paths to load as tool interfaces.'
            )
        )

        self.chat_history = []
        self.load_llm_client()
        self._initialize_chat_history()

        # Load tools and their corresponding functions
        self.tools, self.tool_functions = self._load_tools()
        if self.tools:
            self.get_logger().info(f"Successfully loaded {len(self.tools)} tools.")
        
        self.sub = self.create_subscription(
            String, 'llm_prompt', self.prompt_callback, 10)

        self.pub_response = self.create_publisher(
            String, 'llm_response', 10)

        self.pub_stream = self.create_publisher(
            String, 'llm_stream', 10)

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
                frequency_penalty=self.get_parameter('frequency_penalty').value
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
                    tools = module.register(module)
                else:
                    self.get_logger().info(f"Using default 'register' for {path_str}")
                    tools = default_register(module)

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

    def _trim_chat_history(self):
        """
        Prevents the chat history from growing indefinitely.

        It trims the oldest user/assistant messages to stay within the
        'max_history_length' limit, preserving the system prompt and any
        initial few-shot examples.
        """
        max_len = self.get_parameter('max_history_length').value
        start_index = 0

        for i, msg in enumerate(self.chat_history):
            if msg['role'] in ['user', 'assistant']:
                start_index = i
                break

        conversational_history = self.chat_history[start_index:]
        if len(conversational_history) > max_len * 2:
            messages_to_remove = len(conversational_history) - (max_len * 2)
            del self.chat_history[start_index : start_index + messages_to_remove]
            self.get_logger().info(
                f"Trimmed {messages_to_remove} old message(s) from chat history.")

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
        if not self.llm_client:
            self.get_logger().error("LLM client is not available. Cannot process prompt.")
            return

        self.get_logger().info(f"Received prompt: '{msg.data}'")
        self.chat_history.append({"role": "user", "content": msg.data})
        self._trim_chat_history()

        stream_enabled = self.get_parameter('stream').value
        max_calls = self.get_parameter('max_tool_calls').value
        tool_call_count = 0

        # Phase 1: Tool-handling loop (always non-streaming)
        while tool_call_count < max_calls:
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
                self.chat_history.append(response_message)
                self.get_logger().info(f"LLM requested a tool call: {response_message['tool_calls']}")
                tool_call_count += 1
                tool_calls = response_message["tool_calls"]
                for tool_call in tool_calls:
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

                        self.chat_history.append({
                            "tool_call_id": tool_call_id, "role": "tool",
                            "name": function_name, "content": str(result)
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

        # Phase 2: Final response generation (stream or single response)
        if stream_enabled:
            self.get_logger().info("Streaming final response...")
            full_response = ""

            for chunk in self.llm_client.stream_prompt(self.chat_history):
                if chunk:
                    full_response += chunk
                    self.pub_stream.publish(
                        String(data=chunk))

            self.pub_response.publish(
                String(data=full_response))
            self.get_logger().info(
                f"Finished streaming. Full response: {full_response[:80]}...")
            self.chat_history.append(
                {"role": "assistant", "content": full_response})
        else:
            self.get_logger().info("Generating non-streamed final response...")
            success, final_message = self.llm_client.process_prompt(self.chat_history)

            if success and final_message.get("content"):
                llm_response_text = final_message["content"]
                self.pub_response.publish(
                    String(data=lm_response_text))
                self.get_logger().info(
                    f"Published LLM response: {llm_response_text[:80]}...")
                self.chat_history.append(final_message)
            else:
                self.get_logger().error(
                    f"Failed to get final non-streamed response: {final_message}")
                self.pub_response.publish(
                    String(data="Sorry, I encountered an error generating my final response."))

def main(args=None):
    rclpy.init(args=args)
    llm_node = LLMNode()

    try:
        rclpy.spin(llm_node)
    except KeyboardInterrupt:
        llm_node.get_logger().info(
            "KeyboardInterrupt received, shutting down.")
    finally:
        llm_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()