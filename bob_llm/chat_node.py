#!/usr/bin/env python3
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

import argparse
import json
import os
import sys
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.styles import Style
    from rich.box import ROUNDED
    from rich.console import Console, Group
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.text import Text
except ImportError:
    print('Error: Required libraries not found.')
    print('Please install them using:')
    print('pip install rich prompt_toolkit')
    sys.exit(1)


class BobChatClient(Node):
    def __init__(
        self,
        topic_in='llm_prompt',
        topic_out='llm_stream',
        topic_response='llm_response',
        topic_tools='llm_tool_calls',
        topic_reasoning='llm_reasoning',
        panels=False
    ):
        super().__init__('bob_chat_client')
        self.panels = panels

        # Declare parameters with environment variable fallbacks
        default_queue_size = int(os.environ.get('CHAT_QUEUE_SIZE', '1000'))
        self.declare_parameter('queue_size', default_queue_size)
        queue_size = self.get_parameter('queue_size').value

        default_idle_timeout = float(os.environ.get('CHAT_IDLE_TIMEOUT', '10.0'))
        self.declare_parameter('idle_timeout', default_idle_timeout)
        self.idle_timeout = self.get_parameter('idle_timeout').value

        self.pub_prompt = self.create_publisher(String, topic_in, queue_size)
        self.sub_stream = self.create_subscription(
            String, topic_out, self.stream_callback, queue_size
        )
        self.sub_reasoning = self.create_subscription(
            String, topic_reasoning, self.reasoning_callback, queue_size
        )
        self.sub_response = self.create_subscription(
            String, topic_response, self.response_callback, queue_size
        )
        self.sub_tools = self.create_subscription(
            String, topic_tools, self.tool_callback, queue_size
        )

        self.console = Console()
        self.live = None
        self.full_content = ''
        self.full_reasoning = ''
        self.is_receiving = False
        self.is_reasoning = False
        self.waiting_for_response = False
        self.last_stream_time = 0.0
        self.last_ui_update_time = 0.0
        self.ui_update_rate_limit = 0.1  # Max 10 FPS (1/0.1)

    def _update_live_display(self, force=False):
        if not self.live:
            return

        # Simple rate limiting to avoid re-rendering Markdown on every single token
        # which is very CPU expensive for long responses.
        current_time = time.time()
        if not force and (current_time - self.last_ui_update_time < self.ui_update_rate_limit):
            return
        self.last_ui_update_time = current_time

        parts = []
        if self.panels:
            if self.full_reasoning:
                parts.append(Panel(Markdown(self.full_reasoning), title='Reasoning',
                                   border_style='dim', box=ROUNDED))
            if self.full_content:
                parts.append(Panel(Markdown(self.full_content), title='LLM',
                                   border_style='blue', box=ROUNDED))
        else:
            if self.full_reasoning:
                parts.append(Text.from_markup('[dim]REASONING:[/]\n'))
                parts.append(Markdown(self.full_reasoning))
                parts.append(Text('\n'))
            if self.full_content:
                parts.append(Text.from_markup('[bold blue]LLM:[/]\n'))
                parts.append(Markdown(self.full_content))

        if parts:
            self.live.update(Group(*parts), refresh=True)

    def stream_callback(self, msg):
        self.last_stream_time = time.time()
        chunk = msg.data
        if not self.is_receiving:
            self.full_content = ''
            self.is_receiving = True
            if not self.live:
                self.live = Live(
                    Markdown(''),
                    console=self.console,
                    auto_refresh=False,
                    vertical_overflow='visible'
                )
                self.live.start()

        self.full_content += chunk
        self._update_live_display()

    def reasoning_callback(self, msg):
        self.last_stream_time = time.time()
        chunk = msg.data
        if not self.is_reasoning:
            self.full_reasoning = ''
            self.is_reasoning = True
            if not self.live:
                self.live = Live(
                    Markdown(''),
                    console=self.console,
                    auto_refresh=False,
                    vertical_overflow='visible'
                )
                self.live.start()

        self.full_reasoning += chunk
        self._update_live_display()

    def response_callback(self, msg):
        # ...
        if self.is_receiving or self.is_reasoning:
            if self.live:
                # Force one final update to ensure everything is rendered
                self._update_live_display(force=True)
                self.live.stop()
                self.live = None
            self.is_receiving = False
            self.is_reasoning = False
            self.console.print('')
        self.waiting_for_response = False

    def tool_callback(self, msg):
        try:
            call = json.loads(msg.data)
            name = call.get('name', 'unknown')
            args = call.get('arguments', '{}')
            if self.panels:
                self.console.print(
                    Panel(
                        f'[yellow]Calling: {name}({args})[/]',
                        title='SKILL',
                        border_style='yellow',
                        box=ROUNDED
                    )
                )
            else:
                self.console.print(
                    f'[yellow][*] SKILL: {name}({args})[/]'
                )
            # Reset idle timer because tools took time
            self.last_stream_time = time.time()
        except Exception as e:
            self.get_logger().error(f'Failed to parse tool call message: {e}')

    def send_prompt(self, text):
        msg = String()
        msg.data = text
        self.pub_prompt.publish(msg)


def main(args=None):
    parser = argparse.ArgumentParser(description='Bob LLM Chat Client')
    parser.add_argument(
        '--topic_in', default='llm_prompt',
        help='ROS Topic to send prompts to (default: llm_prompt)'
    )
    parser.add_argument(
        '--topic_out', default='llm_stream',
        help='ROS Topic to receive streamed responses (default: llm_stream)'
    )
    parser.add_argument(
        '--topic_response', default='llm_response',
        help='ROS Topic to receive final complete responses (default: llm_response)'
    )
    parser.add_argument(
        '--topic_tools', default='llm_tool_calls',
        help='ROS Topic to receive tool call notifications (default: llm_tool_calls)'
    )
    parser.add_argument(
        '--topic_reasoning', default='llm_reasoning',
        help='ROS Topic to receive model reasoning content (default: llm_reasoning)'
    )
    parser.add_argument(
        '--panels', action='store_true',
        help='Enable boxed UI (default: off)'
    )
    parsed_args, ros_args = parser.parse_known_args(sys.argv[1:])

    rclpy.init(args=ros_args)
    client_node = BobChatClient(
        topic_in=parsed_args.topic_in,
        topic_out=parsed_args.topic_out,
        topic_response=parsed_args.topic_response,
        topic_tools=parsed_args.topic_tools,
        topic_reasoning=parsed_args.topic_reasoning,
        panels=parsed_args.panels
    )

    # Spin the ROS node in a separate background thread
    spin_thread = threading.Thread(
        target=rclpy.spin,
        args=(client_node,),
        daemon=True
    )
    spin_thread.start()

    console = client_node.console
    console.print(
        '[bold green]Chat for https://github.com/bob-ros2/bob_llm[/]'
    )
    console.print(
        '[dim]Usage: Press [bold yellow]Enter[/] to send, '
        'or [bold yellow]Alt+Enter[/] (Esc then Enter) for a new line.[/dim]'
    )
    console.print("[dim]Type 'exit' or 'quit' to end the session.\n[/dim]")

    bindings = KeyBindings()

    @bindings.add('enter')
    def _(event):
        event.current_buffer.validate_and_handle()

    @bindings.add('escape', 'enter')
    def _(event):
        event.current_buffer.insert_text('\n')

    style = Style.from_dict({
        'prompt': 'ansicyan bold',
    })

    session = PromptSession(
        message=[('class:prompt', '❯ ')],
        multiline=True,
        key_bindings=bindings,
        style=style
    )

    try:
        while rclpy.ok():
            try:
                user_input = session.prompt()
            except KeyboardInterrupt:
                break
            except EOFError:
                break

            cleaned_input = user_input.strip()
            if not cleaned_input:
                continue

            if cleaned_input.lower() in ['exit', 'quit']:
                break

            # Clear the raw input line from terminal to avoid "double prompt"
            # \033[F moves cursor up, \033[K clears the line
            sys.stdout.write('\033[F\033[K')
            sys.stdout.flush()

            # Print the input
            if client_node.panels:
                console.print(
                    Panel(
                        cleaned_input,
                        title='YOU',
                        border_style='green',
                        box=ROUNDED
                    )
                )
            else:
                console.print(f'\n[bold green]YOU:[/] {cleaned_input}')

            client_node.send_prompt(cleaned_input)
            client_node.waiting_for_response = True

            try:
                # Wait for response to finish
                while client_node.waiting_for_response and rclpy.ok():
                    time.sleep(0.1)
                    # Auto-fallback if the final response message was lost
                    # or the user forgot to set --topic_response correctly
                    if client_node.is_receiving:
                        idle_time = time.time() - client_node.last_stream_time
                        if idle_time > client_node.idle_timeout:
                            client_node.response_callback(None)
            except KeyboardInterrupt:
                client_node.waiting_for_response = False
                if client_node.is_receiving:
                    if client_node.live:
                        client_node.live.stop()
                    client_node.is_receiving = False
                console.print('\n[dim]Waiting cancelled.[/dim]')
                continue

    finally:
        # Ensure the live display is stopped
        if client_node.live:
            client_node.live.stop()

        console.print('\n[bold red]Session ended. Goodbye![/]')

        # Graceful ROS 2 shutdown
        if rclpy.ok():
            rclpy.shutdown()

        # Wait for the spin thread to finish its cleanup
        if spin_thread.is_alive():
            spin_thread.join(timeout=1.0)


if __name__ == '__main__':
    main()
