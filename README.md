[![ROS 2 CI](https://github.com/bob-ros2/bob_llm/actions/workflows/ros_ci.yml/badge.svg)](https://github.com/bob-ros2/bob_llm/actions/workflows/ros_ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# ROS Package [bob_llm](https://github.com/bob-ros2/bob_llm)

The `bob_llm` package provides a ROS 2 node (`llm node`) that acts as a powerful interface to an external Large Language Model (LLM). It operates as a stateful service that maintains a conversation, connects to any OpenAI-compatible API, and features a robust tool execution system.

## Features

-   **OpenAI-Compatible:** Connects to any LLM backend that exposes an OpenAI-compatible API endpoint (e.g., `Ollama`, `vLLM`, `llama-cpp-python`, commercial APIs).
-   **Stateful Conversation:** Maintains chat history to provide conversational context to the LLM.
-   **Dynamic Tool System:** Dynamically loads Python functions from user-provided files and makes them available to the LLM. The LLM can request to call these functions to perform actions or gather information.
-   **Streaming Support:** Can stream the LLM's final response token-by-token for real-time feedback.
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
ros2 run bob_llm llm --ros-args --params-file src/bob_llm/config/node_params.yaml
```

### 2. Enter Interactive Chat

Interact with Bob through a dedicated, interactive terminal client.

```bash
# Start standard chat
ros2 run bob_llm chat

# Start with premium boxed UI
ros2 run bob_llm chat --panels
```

#### CLI Arguments for `chat`

| Option | Default | Description |
| :--- | :--- | :--- |
| `--topic_in` | `llm_prompt` | ROS Topic to send prompts to. |
| `--topic_out` | `llm_stream` | ROS Topic to receive streamed chunks. |
| `--topic_tools` | `llm_tool_calls` | Topic for skill execution feedback. |
| `--panels` | `False` | Enable decorative boxes around messages. |

#### Example Session

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

**Image Helper:**
If `process_image_urls` is enabled, the node automatically base64-encodes images from `file://` or `http://` URLs.

```bash
ros2 topic pub /llm_prompt std_msgs/msg/String "data: '{\"role\": \"user\", \"content\": \"Describe this\", \"image_url\": \"file:///tmp/cam.jpg\"}'" -1
```

## ROS 2 API

| Topic | Type | Path |
| :--- | :--- | :--- |
| `/llm_prompt` | `std_msgs/msg/String` | Incoming prompts (Sub) |
| `/llm_response` | `std_msgs/msg/String` | Final complete string (Pub) |
| `/llm_stream` | `std_msgs/msg/String` | Real-time token stream (Pub) |
| `/llm_tool_calls` | `std_msgs/msg/String` | JSON info about used skills (Pub) |
| `/llm_latest_turn`| `std_msgs/msg/String` | Last User/Bot pair as JSON (Pub) |


## Configuration (node_params.yaml)

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `api_url` | `http://localhost:8000/v1` | Your LLM server URL. |
| `api_model` | `gpt-4` | Model name. |
| `max_history_length`| `10` | Conversation memory (in turns). |
| `max_tool_calls` | `5` | Recursion limit for skills. |
| `tool_interfaces` | `[]` | List of Python files containing skills. |
| `skill_dir` | `./config/skills` | Location for Anthropic standard skills. |

---

## Skill System (Agentskills)

The `bob_llm` node implements the [Anthropic Agent Skills](https://agentskills.io/specification) specification.

### 1. Skill Tools (`config/skill_tools.py`)
This module allows the LLM to **discover**, **learn**, and **execute** skills from a directory.

### 2. Usage
Add the absolute path of `config/skill_tools.py` to your `tool_interfaces` and set the `skill_dir`:

```yaml
llm:
  ros__parameters:
    tool_interfaces: ["/path/to/skill_tools.py"]
    skill_dir: "/path/to/my/skills"
```

The LLM will automatically discover skills by reading `SKILL.md` files and can run their scripts using the `execute_skill_script` tool.