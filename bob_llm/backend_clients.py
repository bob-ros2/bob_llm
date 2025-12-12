import json
import requests
import traceback

class OpenAICompatibleClient:
    """
    A client for OpenAI-compatible APIs that supports chat, tool use, and streaming.

    This class encapsulates the logic for sending requests to an LLM backend,
    handling both standard (blocking) and streaming responses. It is configured
    with various generation parameters to control the LLM's output.
    """
    def __init__(self,
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
        response_format: dict = None):
        """
        Initializes the OpenAICompatibleClient.

        Args:
            api_url: The base URL of the OpenAI-compatible API.
            api_key: An optional API key for authentication (Bearer token).
            model: The name of the model to use for completions.
            logger: A ROS logger instance for logging messages.
            temperature: Controls the randomness of the output.
            top_p: Nucleus sampling parameter.
            max_tokens: The maximum number of tokens to generate.
            stop: A list of sequences to stop generation at.
            presence_penalty: Penalty for new tokens based on their presence.
            frequency_penalty: Penalty for new tokens based on their frequency.
            timeout: Timeout in seconds for API requests.
            response_format: Optional dict for structured output, e.g. {"type": "json_object"}.
        """

        self.api_url = api_url.rstrip('/') + "/v1/chat/completions"
        self.api_key = api_key
        self.model = model
        self.logger = logger
        self.headers = {"Content-Type": "application/json"}
        if api_key: self.headers["Authorization"] = f"Bearer {api_key}"

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
        Constructs the JSON payload for an API request.

        This helper method assembles the request body, including the model name,
        message history, generation parameters, tool definitions, and stream flag.
        It enforces the convention of using either 'temperature' or 'top_p', but
        not both.

        Args:
            history: The list of messages in the chat history.
            tools: An optional list of tool definitions.
            stream: A boolean indicating whether to enable streaming.

        Returns:
            A dictionary representing the complete JSON payload.
        """
        # Sanitize history: ensure all messages have content as string (some backends require this)
        sanitized_history = []
        for msg in history:
            msg_copy = msg.copy()
            content = msg_copy.get("content")
            if content is None:
                msg_copy["content"] = ""
            elif not isinstance(content, str):
                # If content is a list (multimodal) or other type, convert to string
                msg_copy["content"] = json.dumps(content) if isinstance(content, (list, dict)) else str(content)
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
        Sends a non-streaming request to the LLM to get a complete response.

        This is used for single-shot responses, especially when expecting a tool call
        from the model.

        Args:
            history: The list of messages in the chat history.
            tools: An optional list of tool definitions.

        Returns:
            A tuple containing a boolean for success and either the response
            message dictionary or an error string on failure.
        """
        payload = self._build_payload(history, tools, stream=False)

        try:
            response = requests.post(
                self.api_url, 
                headers=self.headers, 
                json=payload, 
                timeout=self.timeout)
            response.raise_for_status()
            message = response.json()['choices'][0]['message']

            # Return the entire message dictionary to handle both text and tool calls.
            return True, message

        except requests.exceptions.RequestException as e:
            error_trace = traceback.format_exc()
            error_msg = f"API request failed: {e}"
            if self.logger: self.logger.error(f"{error_msg}\n{error_trace}")
            return False, error_msg

    def stream_prompt(self, history: list, tools: list = None):
        """
        Sends a streaming request to the LLM and yields response chunks.

        This method is used for generating the final text response token-by-token.

        Args:
            history: The list of messages in the chat history.
            tools: An optional list of tool definitions (rarely used with streaming).

        Yields:
            String chunks of the generated text content as they are received.
            An error message string is yielded if the request fails.
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
                                if 'choices' in data and len(data['choices']) > 0:
                                    delta = data['choices'][0].get('delta', {})
                                    content = delta.get('content')
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                if self.logger: self.logger.warning(f"Could not decode JSON from stream: {json_str}")
        except requests.exceptions.RequestException as e:
            error_trace = traceback.format_exc()
            error_msg = f"API stream request failed: {e}"
            if self.logger: self.logger.error(f"{error_msg}\n{error_trace}")
            # Yield an error message to the caller
            yield f"[ERROR: {error_msg}]"