[![ROS 2 CI](https://github.com/bob-ros2/bob_llm/actions/workflows/ros_ci.yml/badge.svg)](https://github.com/bob-ros2/bob_llm/actions/workflows/ros_ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

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
| `api_type` | string | `openai` | LLM backend API type. |
| `api_url` | string | `http://...` | Base URL of the LLM backend. |
| `api_key` | string | `no_key` | API key for authentication. |
| `api_model` | string | `""` | Specific model name (e.g., "gpt-4"). |
| `system_prompt` | string | `""` | System context instructions. |
| `max_history_length` | integer | `10` | Max conversation turns to remember. |
| `max_tool_calls` | integer | `5` | Max consecutive tool calls allowed. |
| `stream` | bool | `true` | Enable/disable token streaming. |
| `temperature` | double | `0.7` | Output randomness. |
| `tool_interfaces` | string array | `[]` | Paths to Python tool files. |
| `skill_dir` | string | `./config/skills` | Directory where skills are stored. |

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

#### 2. Ready-to-use Tools
- **ROS CLI Tools (`config/ros_cli_tools.py`):** Inspect and control the ROS system.
- **Qdrant Memory Tools (`config/qdrant_tools.py`):** Semantic long-term memory.
    - Configured via `LLM_QDRANT_LOCATION`, `LLM_QDRANT_COLLECTION`, etc.