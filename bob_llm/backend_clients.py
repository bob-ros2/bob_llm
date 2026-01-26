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

import json
import requests


class OpenAICompatibleClient:
    """Client for OpenAI-compatible APIs."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str,
        logger,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 0,
        stop: list = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        timeout: float = 60.0,
        response_format: dict = None
    ):
        """
        Initialize the OpenAICompatibleClient.

        :param api_url: The base URL of the OpenAI-compatible API.
        :param api_key: An optional API key for authentication.
        :param model: The name of the model to use for completions.
        :param logger: A ROS logger instance for logging messages.
        :param temperature: Controls the randomness of the output.
        :param top_p: Nucleus sampling parameter.
        :param max_tokens: The maximum number of tokens to generate.
        :param stop: A list of sequences to stop generation at.
        :param presence_penalty: Penalty for new tokens.
        :param frequency_penalty: Penalty for new tokens.
        :param timeout: Timeout in seconds for API requests.
        :param response_format: Optional dict for structured output.
        """
        self.api_url = api_url.rstrip('/') + "/v1/chat/completions"
        self.api_key = api_key
        self.model = model
        self.logger = logger
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stop = stop
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.timeout = timeout
        self.response_format = response_format

    def _build_payload(self, history: list, tools: list = None, stream: bool = False) -> dict:
        """
        Construct the JSON payload for an API request.

        :param history: The list of messages in the chat history.
        :param tools: An optional list of tool definitions.
        :param stream: A boolean indicating whether to enable streaming.
        :return: A dictionary representing the complete JSON payload.
        """
        # Sanitize history: ensure all messages have content as string
        sanitized_history = []
        for msg in history:
            msg_copy = msg.copy()
            content = msg_copy.get("content")
            if content is None:
                msg_copy["content"] = ""
            elif not isinstance(content, str):
                if isinstance(content, (list, dict)):
                    msg_copy["content"] = json.dumps(content)
                else:
                    msg_copy["content"] = str(content)
            sanitized_history.append(msg_copy)

        payload = {
            "model": self.model,
            "messages": sanitized_history,
        }

        # Add parameters. Honor the "don't set both temp and top_p" convention.
        if self.top_p < 1.0:
            payload['top_p'] = self.top_p
        else:
            payload['temperature'] = self.temperature

        if self.max_tokens > 0:
            payload['max_tokens'] = self.max_tokens
        if self.stop:
            payload['stop'] = self.stop
        if self.presence_penalty != 0.0:
            payload['presence_penalty'] = self.presence_penalty
        if self.frequency_penalty != 0.0:
            payload['frequency_penalty'] = self.frequency_penalty

        if tools:
            payload['tools'] = tools
            payload['tool_choice'] = 'auto'

        if stream:
            payload['stream'] = True

        if self.response_format:
            payload['response_format'] = self.response_format

        return payload

    def process_prompt(self, history: list, tools: list = None) -> (bool, dict):
        """
        Send a non-streaming request to the LLM to get a complete response.

        :param history: The list of messages in the chat history.
        :param tools: An optional list of tool definitions.
        :return: A tuple containing (success, message).
        """
        payload = self._build_payload(history, tools, stream=False)

        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            message = response.json()['choices'][0]['message']

            # Return the entire message dictionary to handle both text and tool calls.
            return True, message

        except requests.exceptions.RequestException as e:
            error_msg = f"API request failed: {e}"
            if self.logger:
                self.logger.error(error_msg)
            return False, error_msg

    def stream_prompt(self, history: list, tools: list = None):
        """
        Send a streaming request to the LLM and yield response chunks.

        :param history: The list of messages in the chat history.
        :param tools: An optional list of tool definitions.
        :yield: String chunks of the generated text content.
        """
        payload = self._build_payload(history, tools, stream=True)

        try:
            with requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                stream=True,
                timeout=self.timeout
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            json_str = line_str[6:]
                            if json_str.strip() == '[DONE]':
                                break
                            try:
                                data = json.loads(json_str)
                                if 'choices' in data and data['choices']:
                                    delta = data['choices'][0].get('delta', {})
                                    content = delta.get('content')
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                if self.logger:
                                    self.logger.warning(
                                        f"Could not decode JSON from stream: {json_str}")
        except requests.exceptions.RequestException as e:
            error_msg = f"API stream request failed: {e}"
            if self.logger:
                self.logger.error(error_msg)
            yield f"[ERROR: {error_msg}]"
