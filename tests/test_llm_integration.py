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

"""Integration tests for the bob_llm node."""

import json
import os
import subprocess
import time

import pytest
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

# Optional: rich for beautiful terminal output
try:
    from rich.console import Console
    from rich.markdown import Markdown
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False


def load_env(file_path):
    """Load environment variables from a file."""
    if not os.path.exists(file_path):
        template_path = os.path.join(os.path.dirname(file_path),
                                     '.env.template')
        msg = (
            '\n[bold red]❌ ERROR: Configuration file ".env" '
            'not found![/bold red]\n'
            'To run this integration test, you must create a '
            '[bold].env[/bold] file in:\n'
            f'[blue]{os.path.dirname(file_path)}[/blue]\n\n'
            '1. Copy the template:\n'
            f'   [cyan]cp {template_path} {file_path}[/cyan]\n\n'
            '2. Edit the [bold].env[/bold] file and set your '
            '[yellow]LLM_API_KEY[/yellow].\n'
            '3. Ensure [yellow]LLM_TOOL_INTERFACES[/yellow] points to a '
            'valid absolute path.\n'
        )
        if HAS_RICH:
            console.print(msg)
        else:
            # Strip rich tags for plain print
            import re
            plain_msg = re.sub(r'\[.*?\]', '', msg)
            print(plain_msg)

        pytest.exit('Missing .env configuration file. See instructions above.',
                    returncode=1)

    env = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    if '=' in line:
                        key, value = line.split('=', 1)
                        env[key.strip()] = value.strip('"').strip("'")
                except ValueError:
                    continue
    return env


class LLMTestNode(Node):
    """Test node for inspecting LLM topics during integration tests."""

    def __init__(self, verbose=False):
        """Initialize the test node."""
        super().__init__('llm_test_inspector')
        self.verbose = verbose

        self.responses = []
        self.stream_tokens = []
        self.reasoning_chunks = []
        self.tool_calls = []

        self.sub_response = self.create_subscription(
            String, 'llm_response', self.response_cb, 10)
        self.sub_stream = self.create_subscription(
            String, 'llm_stream', self.stream_cb, 100)
        self.sub_reasoning = self.create_subscription(
            String, 'llm_reasoning', self.reasoning_cb, 100)
        self.sub_tool_calls = self.create_subscription(
            String, 'llm_tool_calls', self.tool_calls_cb, 10)

        self.pub_prompt = self.create_publisher(String, 'llm_prompt', 10)

    def response_cb(self, msg):
        """Handle incoming final responses."""
        self.responses.append(msg.data)

    def stream_cb(self, msg):
        """Handle incoming stream tokens."""
        self.stream_tokens.append(msg.data)

    def reasoning_cb(self, msg):
        """Handle incoming reasoning chunks."""
        self.reasoning_chunks.append(msg.data)

    def tool_calls_cb(self, msg):
        """Handle incoming tool call notifications."""
        if self.verbose:
            if HAS_RICH:
                console.print(f'[bold cyan]🔧 TOOL CALL:[/bold cyan] '
                              f'[yellow]{msg.data}[/yellow]')
            else:
                print(f'\n[LIVE TOOL CALL] -> {msg.data}')
        try:
            self.tool_calls.append(json.loads(msg.data))
        except Exception as e:
            self.get_logger().error(f'Failed to parse tool call JSON: {e}')


@pytest.fixture
def test_config():
    """Load the test configuration from .env."""
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    return load_env(env_path)


@pytest.fixture
def llm_node_process(test_config):
    """Launch the llm_node in a separate process with isolation."""
    test_env = os.environ.copy()
    test_env.update(test_config)
    test_env['ROS_DOMAIN_ID'] = '88'

    cmd = ['ros2', 'run', 'bob_llm', 'llm']
    process = subprocess.Popen(
        cmd, env=test_env, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True
    )

    time.sleep(5)
    yield process

    process.terminate()
    try:
        process.communicate(timeout=2.0)
    except subprocess.TimeoutExpired:
        process.kill()


def test_full_mandatory_flow(ros_context, llm_node_process, test_config):
    """Comprehensive test of all mandatory LLM topics."""
    verbose = test_config.get('TEST_VERBOSE') == '1'
    is_reasoner = test_config.get('LLM_IS_REASONER') == 'true'

    os.environ['ROS_DOMAIN_ID'] = '88'
    test_node = LLMTestNode(verbose=verbose)
    executor = MultiThreadedExecutor()
    executor.add_node(test_node)

    # Prompt to trigger stream, tool calls, and final response
    prompt = (
        'Check the live ROS topic list right now and tell me exactly how many '
        'topics are active. Do NOT rely on your memory, I need fresh data.'
    )

    print('\n')
    if HAS_RICH:
        console.print(
            '[bold magenta]───────────────── STARTING INTEGRATION TEST '
            '(Domain 88) ─────────────────[/bold magenta]'
        )
        console.print(f'[bold]Prompt:[/bold] {prompt}')
        mode_str = 'REASONING' if is_reasoner else 'STANDARD'
        console.print(f'[bold]Mode:[/bold]   {mode_str}')
        console.print(
            '[bold magenta]───────────────────────────────────────────────'
            '──────────────────[/bold magenta]'
        )
    else:
        print('\n' + '=' * 50)
        print('--- STARTING INTEGRATION TEST (Domain 77) ---')
        print(f'Prompt: {prompt}')
        print('=' * 50 + '\n')

    test_node.pub_prompt.publish(String(data=prompt))

    start_time = time.time()
    timeout = 120.0
    response_received_time = None

    while time.time() - start_time < timeout:
        executor.spin_once(timeout_sec=0.1)
        if len(test_node.responses) > 0 and response_received_time is None:
            response_received_time = time.time()
        if response_received_time and (
                time.time() - response_received_time > 3.0):
            break

    # Defer verbose output until everything is collected
    if verbose:
        if HAS_RICH:
            console.print('\n')
            if test_node.reasoning_chunks:
                reasoning_text = ''.join(test_node.reasoning_chunks)
                console.print('\n[bold blue]🧠 LLM Reasoning[/bold blue]')
                console.print(Markdown(reasoning_text))

            if test_node.stream_tokens:
                stream_content = ''.join(test_node.stream_tokens)
                console.print('\n[bold green]📝 Streamed Output (Markdown)'
                              '[/bold green]')
                console.print(Markdown(stream_content))

            if test_node.responses:
                console.print('\n[bold white]✅ Final Response Topic'
                              '[/bold white]')
                console.print(Markdown(test_node.responses[-1]))
        else:
            print('\n' + '-' * 50)
            print('--- FULL CONTENT LOG (DEFERRED OUTPUT) ---')
            if test_node.reasoning_chunks:
                txt = ''.join(test_node.reasoning_chunks)
                print('\n[REASONING CONTENT]:\n' + txt)
            if test_node.stream_tokens:
                txt = ''.join(test_node.stream_tokens)
                print('\n[STREAMED TOKENS]:\n' + txt)
            if test_node.responses:
                print('\n[FINAL RESPONSE]:\n' + test_node.responses[-1])
            print('-' * 50)

    # CHECKLIST
    print('\n--- TEST RESULTS CHECKLIST ---')

    stream_ok = len(test_node.stream_tokens) > 0
    res_len = len(test_node.stream_tokens)
    print(f'[CHECK] llm_stream (Tokens):   '
          f'{"PASSED" if stream_ok else "FAILED"} ({res_len} tokens)')

    response_ok = len(test_node.responses) > 0
    print(f'[CHECK] llm_response (Final):  '
          f'{"PASSED" if response_ok else "FAILED"}')

    reasoning_ok = len(test_node.reasoning_chunks) > 0
    if is_reasoner:
        print(f'[CHECK] llm_reasoning (MANDATORY): '
              f'{"PASSED" if reasoning_ok else "FAILED"}')
    else:
        status = 'RECEIVED' if reasoning_ok else 'NOT'
        print(f'[INFO]  llm_reasoning (Optional):  {status}')

    tools_ok = len(test_node.tool_calls) > 0
    print(f'[CHECK] llm_tool_calls (Tools): '
          f'{"PASSED" if tools_ok else "FAILED"}')

    print('------------------------------\n')

    # MANDATORY ASSERTIONS
    assert stream_ok, 'MANDATORY: No tokens received on llm_stream'
    assert response_ok, 'MANDATORY: No final response received on llm_response'
    assert tools_ok, 'MANDATORY: No tool calls received'
    if is_reasoner:
        assert reasoning_ok, 'MANDATORY: Model is reasoner but no reasoning'

    test_node.destroy_node()
