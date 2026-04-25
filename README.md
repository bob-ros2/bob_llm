[![ROS 2 CI](https://github.com/bob-ros2/bob_llm/actions/workflows/ros_ci.yml/badge.svg)](https://github.com/bob-ros2/bob_llm/actions/workflows/ros_ci.yml)
[![amd64](https://img.shields.io/github/actions/workflow/status/bob-ros2/bob_llm/docker.yml?label=amd64&logo=docker)](https://github.com/bob-ros2/bob_llm/actions/workflows/docker.yml)
[![arm64](https://img.shields.io/github/actions/workflow/status/bob-ros2/bob_llm/docker.yml?label=arm64&logo=docker)](https://github.com/bob-ros2/bob_llm/actions/workflows/docker.yml)

# ROS Package [bob_llm](https://github.com/bob-ros2/bob_llm)

The `bob_llm` package provides a ROS 2 node (`llm node`) that acts as a powerful interface to an external Large Language Model (LLM). It operates as a stateful service that maintains a conversation, connects to any OpenAI-compatible API, and features a robust tool execution system.

## Features

-   **OpenAI-Compatible:** Connects to any LLM backend that exposes an OpenAI-compatible API endpoint (e.g., `Ollama`, `vLLM`, `llama-cpp-python`, commercial APIs).
-   **Stateful Conversation:** Maintains chat history to provide conversational context to the LLM.
-   **Dynamic Tool System:** Dynamically loads Python functions from user-provided files and makes them available to the LLM. The LLM can request to call these functions to perform actions or gather information.
-   **Anthropic Agent Skills:** Full support for the [Anthropic Agent Skills](https://agentskills.io) specification, enabling modular, self-contained capabilities with documentation and execution logic.
-   **High Performance Streaming:** Optimized byte-stream parsing ensures zero-latency delivery of reasoning tokens and response chunks directly from the socket (no internal buffering).
-   **Reasoning/Thinking Support:** Real-time extraction and publishing of model reasoning (e.g., from Gemma 2 or DeepSeek) to a dedicated topic.
-   **Interactive Chat CLI:** Includes a premium terminal interface with Markdown rendering and multi-line support.
-   **Multi-modality:** Supports multimodal input (e.g., images) via JSON prompts.
-   **Lightweight:** The node core requires only standard Python libraries (`requests`, `rich`, `prompt_toolkit`).
-   **Multi-arch Docker Support:** Ready-to-use Docker images for `amd64` and `arm64`, fully configurable via environment variables for easy deployment.

## Docker Usage

The `bob_llm` node is available as a multi-arch Docker image. All ROS parameters can be configured via environment variables (prefixed with `LLM_`).

### Running with Docker

```bash
docker run -it --rm \
  --name bob-llm \
  -e LLM_API_URL="http://192.168.1.100:8000/v1" \
  -e LLM_API_KEY="your_secret_token" \
  -e LLM_API_MODEL="llama3" \
  -e LLM_TEMPERATURE="0.5" \
  ghcr.io/bob-ros2/bob-llm:latest
```

### Running with Docker Compose

```yaml
services:
  llm:
    image: ghcr.io/bob-ros2/bob-llm:latest
    container_name: bob-llm
    environment:
      - LLM_API_URL=http://llm-backend:8000/v1
      - LLM_API_KEY=sk-12345
      - LLM_API_MODEL=gpt-4
      - LLM_SYSTEM_PROMPT="You are a helpful robot assistant named Bob."
      - LLM_TEMPERATURE=0.8
    restart: always
```


## Installation

1.  **Clone the Repository**
    Navigate to your ROS 2 workspace's `src` directory and clone the repository:
    ```bash
    cd ~/ros2_ws/src
    git clone https://github.com/bob-ros2/bob_llm.git
    ```

2.  **Install Dependencies**
    The node requires a few Python packages. It is recommended to install these within a virtual environment.
    ```bash
    pip install requests PyYAML rich prompt_toolkit
    ```

3.  **Build and Source**
    ```bash
    cd ~/ros2_ws
    colcon build --packages-select bob_llm
    source install/setup.bash
    ```

## Usage

### 1. Start the Brain (LLM Node)

Ensure your LLM server is active and the `api_url` in your params file is correct.

```bash
ros2 run bob_llm llm --ros-args --params-file /path/to/your/ros2_ws/src/bob_llm/config/node_params.yaml
```

### 2. Enter Interactive Chat

Interact with Bob through a dedicated, interactive terminal client.

```bash
# Start standard chat
ros2 run bob_llm chat

# Start with premium boxed UI (visual panels)
ros2 run bob_llm chat --panels
```

#### CLI Arguments for `chat`

| Option | Default | Description |
| :--- | :--- | :--- |
| `--topic_in` | `llm_prompt` | ROS Topic to send prompts to. |
| `--topic_out` | `llm_stream` | ROS Topic to receive streamed chunks. |
| `--topic_reasoning` | `llm_reasoning` | ROS Topic to receive model reasoning content. |
| `--topic_response` | `llm_response` | ROS Topic to receive final complete responses. |
| `--topic_tools` | `llm_tool_calls` | Topic for skill execution feedback. |
| `--panels` | `False` | Enable decorative boxes around messages. |

#### Chat Configuration

The chat client supports the following ROS parameters and environment variables:

- **`queue_size` (Integer)**: ROS parameter to control the subscription queue depth. 
- **`CHAT_QUEUE_SIZE` (Environment Variable)**: Default value for the `queue_size` parameter (default: `1000`).

Example usage:
```bash
export CHAT_QUEUE_SIZE=2000
ros2 run bob_llm chat --topic_in /user_query --topic_out /llm_stream --panels
```

#### Chat Example

```text
Chat for https://github.com/bob-ros2/bob_llm
Usage: Press Enter to send, or Alt+Enter for a new line.

YOU: Was kannst du über dieses System sagen?

[*] SKILL: list_nodes({})

LLM: Ich sehe folgende aktive Komponenten im System:
- /llm (Das Gehirn)
- /bob_chat_client (Dieser Chat)
- /eva/logic (Zustandssteuerung)
```

### 3. Advanced Input & Multi-modality

The node supports advanced input formats beyond simple text. If the input message on `/llm_prompt` is valid JSON, it is parsed as a message object.

**Generic JSON Input:**
You can pass any valid JSON dictionary. If it contains a `role` field (e.g., `user`), it is treated as a standard message object and appended to the history.

**Image Helper:**
If `process_image_urls` is enabled, the node automatically base64-encodes images from `file://` or `http://` URLs.

```bash
ros2 topic pub /llm_prompt std_msgs/msg/String "data: '{\"role\": \"user\", \"content\": \"Describe this\", \"image_url\": \"file:///tmp/cam.jpg\"}'" -1
```

## ROS 2 API

| Topic | Type | Description |
| :--- | :--- | :--- |
| `/llm_prompt` | `std_msgs/msg/String` | **(Subscribed)** Receives user prompts. |
| `/llm_response` | `std_msgs/msg/String` | **(Published)** Final, complete response from the LLM. |
| `/llm_stream` | `std_msgs/msg/String` | **(Published)** token-by-token chunks of the response. |
| `/llm_reasoning` | `std_msgs/msg/String` | **(Published)** Live reasoning/thinking content from the model. |
| `/llm_tool_calls` | `std_msgs/msg/String` | **(Published)** JSON info about tool execution for clients. |
| `/llm_latest_turn`| `std_msgs/msg/String` | **(Published)** Latest turn as JSON array of messages. |


## Configuration

The node is configured through a ROS parameters YAML file. Most parameters support **dynamic reconfiguration** at runtime.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `api_type` | string | `openai_compatible` | The type of the LLM backend API (e.g., "openai_compatible"). [ENV: LLM_API_TYPE] |
| `api_url` | string | `http://localhost:8000/v1` | The base URL of the LLM backend API. [ENV: LLM_API_URL] |
| `api_key` | string | `no_key` | The API key for authentication with the LLM backend. [ENV: LLM_API_KEY] |
| `api_model` | string | `""` | The specific model name to use (e.g., 'gpt-4', 'llama3'). [ENV: LLM_API_MODEL] |
| `system_prompt` | string | `""` | The system prompt to set the LLM context. [ENV: LLM_SYSTEM_PROMPT] |
| `system_prompt_file` | string | `""` | Path to a file containing the system prompt. [ENV: LLM_SYSTEM_PROMPT_FILE] |
| `max_history_length` | integer | `10` | Max turns to keep in history. [ENV: LLM_MAX_HISTORY_LENGTH] Range: [0, 1000] |
| `max_tool_calls` | integer | `5` | Max consecutive tool calls allowed. [ENV: LLM_MAX_TOOL_CALLS] Range: [0, 50] |
| `stream` | bool | `true` | Enable/disable streaming for the final LLM response. [ENV: LLM_STREAM] |
| `temperature` | double | `0.7` | Controls the randomness of the output. [ENV: LLM_TEMPERATURE] Range: [0.0, 2.0] |
| `top_p` | double | `1.0` | Nucleus sampling diversity control. [ENV: LLM_TOP_P] Range: [0.0, 1.0] |
| `max_tokens` | integer | `0` | Max tokens to generate. 0 means no limit. [ENV: LLM_MAX_TOKENS] |
| `presence_penalty` | double | `0.0` | Penalizes new tokens based on presence. [ENV: LLM_PRESENCE_PENALTY] Range: [-2.0, 2.0]|
| `frequency_penalty`| double | `0.0` | Penalizes new tokens based on frequency. [ENV: LLM_FREQUENCY_PENALTY] Range: [-2.0, 2.0]|
| `tool_interfaces` | string array | `[]` | A list of Python modules or file paths to load as tools. [ENV: LLM_TOOL_INTERFACES] |
| `skill_dir` | string | `./config/skills` | Directory where skills are stored. [ENV: LLM_SKILL_DIR] |
| `message_log` | string | `""` | If set, appends conversation turns to this JSON file. [ENV: LLM_MESSAGE_LOG] |
| `response_format` | string | `""` | JSON string defining the output format. [ENV: LLM_RESPONSE_FORMAT] |
| `eof` | string | `""` | Optional string to publish on llm_stream when generation is finished. [ENV: LLM_EOF] |

### Security Note: API Key Cloaking

For security reasons, the `api_key` parameter is **"cloaked"** immediately after the node initializes. Once the key has been read into the node's internal memory, the public ROS parameter is cleared. This prevents the key from being accidentally exposed or read via `ros2 param get /llm api_key`.

> [!TIP]
> For production environments, it is best practice to use an **API Gateway** or Reverse Proxy to inject authentication tokens. This avoids passing sensitive keys through the ROS parameter system entirely.

### Structured JSON Output

The `response_format` parameter enables structured output, forcing the LLM to respond with valid JSON.

#### JSON Object Mode
Force the LLM to output valid JSON by setting `response_format` to `{"type": "json_object"}`. Note: Your prompt **must** mention the word "JSON".

#### JSON Schema Mode (Strict)
Define an exact schema the LLM must follow:
```yaml
response_format: |
  {
    "type": "json_schema",
    "json_schema": {
      "name": "robot_command",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {
          "action": {"type": "string", "enum": ["move", "stop"]},
          "speed": {"type": "number"}
        },
        "required": ["action"]
      }
    }
  }
```

### Conversation Logging

Set the `message_log` parameter to an absolute file path (e.g., `/home/user/chat.json`) to save the entire conversation history.

---

## Tool System

### Creating a Tool File
A tool file is a standard Python script containing functions with docstrings. The system automatically mirrors these functions as tools for the LLM.

### Skill System (Agentskills)
The `bob_llm` node implements the [Anthropic Agent Skills](https://agentskills.io/specification) specification.

#### 1. Skill Discovery
Add the path of `config/skill_tools.py` to your `tool_interfaces` to enable the skill discovery API (`load_skill_info`, `execute_skill_script`, etc.).

**Configuration:** Ensure the `skill_dir` parameter points to a valid directory where your Agentskills are stored. A sample collection of skills is provided in the `./config/skills` directory of the package.

#### 2. Ready-to-use Tools
- **ROS CLI Tools (`config/ros_cli_tools.py`):** Inspect and control the ROS system.
- **Qdrant Memory Tools (`config/qdrant_tools.py`):** Semantic long-term memory.
    - Configured via `LLM_QDRANT_LOCATION`, `LLM_QDRANT_COLLECTION`, etc.