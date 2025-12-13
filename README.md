# ROS Package [bob_llm](https://github.com/bob-ros2/bob_llm)

The `bob_llm` package provides a ROS 2 node (`llm node`) that acts as a powerful interface to an external Large Language Model (LLM). It operates as a stateful service that maintains a conversation, connects to any OpenAI-compatible API, and features a robust tool execution system.

## Features

-   **OpenAI-Compatible:** Connects to any LLM backend that exposes an OpenAI-compatible API endpoint (e.g., `Ollama`, `vLLM`, `llama-cpp-python`, commercial APIs).
-   **Stateful Conversation:** Maintains chat history to provide conversational context to the LLM.
-   **Dynamic Tool System:** Dynamically loads Python functions from user-provided files and makes them available to the LLM. The LLM can request to call these functions to perform actions or gather information.
-   **Streaming Support:** Can stream the LLM's final response token-by-token for real-time feedback.
-   **Fully Parameterized:** All configuration, from API endpoints to LLM generation parameters, is handled through a single ROS parameters file.
-   **Multi-modality:** Supports multimodal input (e.g., images) via JSON prompts.
-   **Lightweight:** The node is simple and has minimal dependencies, requiring only a few standard Python libraries (`requests`, `PyYAML`) on top of ROS 2.


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
    pip install requests PyYAML
    ```
    The required ROS 2 dependencies (`rclpy`, `std_msgs`) will be resolved by the build system.

3.  **Build the Workspace**
    Navigate to the root of your workspace and build the package:
    ```bash
    cd ~/ros2_ws
    colcon build --packages-select bob_llm
    ```

4.  **Source the Workspace**
    Before running the node, remember to source your workspace's setup file:
    ```bash
    source install/setup.bash
    ```

## Usage

### 1. Run the Node

Before running, ensure your LLM server is active and the `api_url` in your params file is correct.

```bash
# Make sure your workspace is sourced
# source install/setup.bash

# Run the node with your parameters file
ros2 run bob_llm llm --ros-args --params-file /path/to/your/ros2_ws/src/bob_llm/config/node_params.yaml
```

### 2. Interact with the Node

The package includes a convenient helper script, `scripts/query.sh`, for interacting with the node directly from the command line.

Once the `llm node` is running, open a new terminal (with the workspace sourced) and run the script:

```bash
$ ros2 run bob_llm query.sh
--- Listening for results on llm_response ---
--- Enter your prompt below (Press Ctrl+C to exit) ---
> What is the status of the robot?
Robot status: Battery is at 85%. All systems are nominal. Currently idle.
>
```

### 3. Advanced Input & Multi-modality

The node supports advanced input formats beyond simple text strings. If the input message on `/llm_prompt` is a valid JSON string, it is parsed and treated as a message object.

**Generic JSON Input:**
You can pass any valid JSON dictionary. If it contains a `role` field (e.g., `user`), it is treated as a standard message object and appended to the history. This allows you to send custom content structures supported by your specific LLM backend (e.g., complex multimodal inputs, custom fields).

**Image Handling Helper:**
For convenience, the node includes a helper for handling images. If `process_image_urls` is set to `true`, the node looks for an `image_url` field in your JSON input. It will automatically fetch the image (from `file://` or `http://` URLs), base64 encode it, and format the message according to the OpenAI Vision API specification.

**Example (Image Helper):**
```bash
ros2 topic pub /llm_prompt std_msgs/msg/String "data: '{\"role\": \"user\", \"content\": \"Describe this image\", \"image_url\": \"file:///path/to/image.jpg\"}'" -1
```

## Conversation Flow

1.  A user publishes a prompt to the `/llm_prompt` topic.
2.  The `llm node` adds the prompt to its internal chat history.
3.  The node sends the history and a list of available tools to the LLM backend.
4.  The LLM decides whether to respond directly or use a tool.
    -   **If Tool:** The LLM returns a request to call a specific function. The `llm node` executes the function, appends the result to the history, and sends the updated history back to the LLM. This loop can repeat multiple times.
    -   **If Text:** The LLM generates a final, natural language response.
5.  The `llm node` publishes the final response. If streaming is enabled, it's sent token-by-token to `/llm_stream` and the full message is sent to `/llm_response` upon completion. Otherwise, the full response is sent directly to `/llm_response`.

## ROS 2 API

The node uses the following topics for communication:

| Topic             | Type                  | Description                                                                                                        |
| ----------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `/llm_prompt`     | `std_msgs/msg/String` | **(Subscribed)** Receives user prompts to be processed by the LLM.                                                     |
| `/llm_response`   | `std_msgs/msg/String` | **(Published)** Publishes the final, complete response from the LLM.                                                   |
| `/llm_stream`     | `std_msgs/msg/String` | **(Published)** Publishes token-by-token chunks of the LLM's response if streaming is enabled.                           |
| `/llm_latest_turn`| `std_msgs/msg/String` | **(Published)** Publishes the latest user/assistant turn as a JSON string: `[{"role": "user", ...}, {"role": "assistant", ...}]`. |


## Configuration

The node is configured entirely through a ROS parameters YAML file (e.g., `config/node_params.yaml`).

### ROS Parameters

All parameters can be set via a YAML file, command-line arguments, or environment variables. The order of precedence is: command-line arguments > parameters file > environment variables > coded defaults. For array-type parameters, environment variables should be comma-separated strings (e.g., `LLM_STOP="stop1,stop2"`).

| Parameter                 | Type         | Default                        | Environment Variable          | Description                                                                          |
| ------------------------- | ------------ | ------------------------------ | ----------------------------- | ------------------------------------------------------------------------------------ |
| `api_type`                | string       | `openai_compatible`            | `LLM_API_TYPE`                | The type of the LLM backend API. Currently only `openai_compatible` is supported.    |
| `api_url`                 | string       | `http://localhost:8000`        | `LLM_API_URL`                 | The base URL of the LLM backend. The node appends `/v1/chat/completions` automatically.|
| `api_key`                 | string       | `no_key`                       | `LLM_API_KEY`                 | The API key (Bearer token) for authentication with the LLM backend, if required.       |
| `api_model`               | string       | `""`                           | `LLM_API_MODEL`               | The specific model name to use (e.g., "gpt-4", "llama3").                              |
| `api_timeout`             | double       | `120.0`                        | `LLM_API_TIMEOUT`             | Timeout in seconds for API requests to the LLM backend.                                |
| `system_prompt`           | string       | `""`                           | `LLM_SYSTEM_PROMPT`           | The system prompt to set the LLM's context, personality, and instructions.           |
| `initial_messages_json`   | string       | `[]`                           | `LLM_INITIAL_MESSAGES_JSON`   | A JSON string of initial messages for few-shot prompting to guide the LLM.             |
| `max_history_length`      | integer      | `10`                           | `LLM_MAX_HISTORY_LENGTH`      | Maximum number of user/assistant conversational turns to keep in history.              |
| `message_log`             | string       | `""`                           | `LLM_MESSAGE_LOG`             | If set to a file path, appends each conversational turn to a persistent JSON log file. |
| `stream`                  | bool         | `true`                         | `LLM_STREAM`                  | Enable or disable streaming for the final LLM response.                                |
| `process_image_urls`      | bool         | `false`                        | `LLM_PROCESS_IMAGE_URLS`      | If true, processes `image_url` in JSON prompts by base64 encoding the image.           |
| `response_format`         | string       | `""`                           | `LLM_RESPONSE_FORMAT`         | JSON string defining the output format. See [Structured JSON Output](#structured-json-output). |
| `max_tool_calls`          | integer      | `5`                            | `LLM_MAX_TOOL_CALLS`          | Maximum number of consecutive tool calls before aborting to prevent loops.           |
| `temperature`             | double       | `0.7`                          | `LLM_TEMPERATURE`             | Controls the randomness of the output. Lower is more deterministic.                    |
| `top_p`                   | double       | `1.0`                          | `LLM_TOP_P`                   | Nucleus sampling. Controls output diversity. Alter this or temperature, not both.    |
| `max_tokens`              | integer      | `0`                            | `LLM_MAX_TOKENS`              | Maximum number of tokens to generate. `0` means use the server's default limit.        |
| `stop`                    | string array | `["stop_llm"]`                 | `LLM_STOP`                    | A list of sequences where the API will stop generating further tokens. Also, sending any of these strings to `llm_prompt` cancels the current generation. |
| `presence_penalty`        | double       | `0.0`                          | `LLM_PRESENCE_PENALTY`        | Penalizes new tokens based on whether they appear in the text so far.                  |
| `frequency_penalty`       | double       | `0.0`                          | `LLM_FREQUENCY_PENALTY`       | Penalizes new tokens based on their existing frequency in the text so far.             |
| `tool_interfaces`         | string array | Path to `example_interface.py` | `LLM_TOOL_INTERFACES`         | A list of absolute paths to Python files containing tool functions. The node will attempt to load functions from each file path provided. |

### Structured JSON Output

The `response_format` parameter enables structured output from the LLM, forcing it to respond with valid JSON that conforms to a specified schema. This is useful for parsing responses programmatically, building automation pipelines, or ensuring consistent output formats.

#### JSON Object Mode

The simplest form forces the LLM to output valid JSON:

```yaml
llm:
  ros__parameters:
    response_format: '{"type": "json_object"}'
    system_prompt: "You are a robot assistant. Always respond with JSON containing 'action' and 'reasoning' fields."
```

> **Important:** When using `json_object` mode, your system prompt or user message **must** mention the word "JSON" — most LLM backends require this or will reject the request.

#### JSON Schema Mode (Strict)

For more control, you can define an exact JSON schema that the LLM must follow. This is supported by OpenAI and compatible backends like vLLM:

```yaml
llm:
  ros__parameters:
    response_format: |
      {
        "type": "json_schema",
        "json_schema": {
          "name": "robot_command",
          "strict": true,
          "schema": {
            "type": "object",
            "properties": {
              "action": {"type": "string", "enum": ["move", "stop", "rotate"]},
              "target": {"type": "string"},
              "speed": {"type": "number"}
            },
            "required": ["action"]
          }
        }
      }
```

#### Example: Command Parsing

With the above schema, the LLM will always respond with valid JSON:

```bash
$ ros2 topic pub /llm_prompt std_msgs/msg/String "data: 'Move forward slowly'" -1
```

Response on `/llm_response`:
```json
{"action": "move", "target": "forward", "speed": 0.3}
```

This makes it easy to parse the response in downstream nodes without complex text parsing.

> [!WARNING]
> **llama.cpp Compatibility:** The `response_format` parameter works with llama.cpp using the format `{"type": "json_object", "schema": {...}}`. However, there is a **bug** in llama.cpp server when combining `response_format` with tool calls — it crashes with `Content path must be a string`. If you need both features, consider using vLLM or another backend until this is fixed upstream.

### Conversation Logging

The node can optionally save the entire conversation to a JSON file, which is useful for debugging, analysis, or creating datasets for fine-tuning models.

To enable logging, set the `message_log` parameter to an absolute file path (e.g., `/home/user/conversation.json`). The node will append each user prompt and the corresponding assistant response to this file. If the file does not exist, it will be created. On the first write to a new file, the `system_prompt` (if configured) will be automatically added as the first entry.

The resulting file will be a flat JSON array of message objects, like this:
```json
[
  {
    "role": "system",
    "content": "You are a helpful robot assistant."
  },
  {
    "role": "user",
    "content": "What is the status of the robot?"
  },
  {
    "role": "assistant",
    "content": "Robot status: Battery is at 85%. All systems are nominal. Currently idle."
  }
]
```

## Tool System

The standout feature of this node is its ability to use dynamically loaded Python functions as tools. The LLM can request to call these functions to perform actions or gather information.

### Creating a Tool File

A tool file is a standard Python script containing one or more functions. The system automatically generates the necessary API schema for the LLM from your function's signature (including type hints) and its docstring. The first line of the docstring is used as the function's description for the LLM.

**Example: `config/example_interface.py`**
```python
def get_weather(location: str, unit: str = "celsius") -> str:
    """
    Get the current weather in a given location.

    This is an example function and will return a fixed string.
    """
    if "tokyo" in location.lower():
        return f"The weather in Tokyo is 10 degrees {unit} and sunny."
    elif "san francisco" in location.lower():
        return f"The weather in San Francisco is 15 degrees {unit} and foggy."
    else:
        return f"Sorry, I don't have the weather for {location}."

def get_robot_status() -> str:
    """
    Retrieves the current status of the robot.

    This function checks the robot's battery level, joint states, and current task.
    """
    # In a real scenario, this would query robot topics or services
    return "Robot status: Battery is at 85%. All systems are nominal. Currently idle."
```

### Configuring Tools

To make your tools available to the LLM, you must provide a list of absolute paths to your Python tool files in the `tool_interfaces` parameter.

Open `config/node_params.yaml` and edit the `tool_interfaces` list. You must replace any placeholder path with the full, absolute path to the tool file on your system.

For example:
```yaml
# In config/node_params.yaml
llm:
  ros__parameters:
    # ... other parameters

    # A list of Python modules to load as tool interfaces.
    # Replace the path below with the absolute path on your machine.
    tool_interfaces:
      - "/home/user/ros2_ws/src/bob_llm/config/example_interface.py"
      # You can add more tool files here
      # - "/home/user/ros2_ws/src/my_robot_tools/my_robot_tools/tools.py"

    # ... other parameters
```

### Inbuilt Tools

The package comes with several ready-to-use tool modules in the `config/` directory.

#### 1. ROS CLI Tools (`config/ros_cli_tools.py`)

This module provides a comprehensive set of tools that wrap standard ROS 2 command-line interface (CLI) functionalities. It allows the LLM to inspect the system (list nodes, topics, services) and interact with it (publish messages, call services, get/set parameters).

**Dependencies:**
-   None (uses standard ROS 2 libraries and CLI tools).

**Usage:**
Add the absolute path to `config/ros_cli_tools.py` to your `tool_interfaces` parameter.

#### 2. Qdrant Memory Tools (`config/qdrant_tools.py`)

This module enables long-term memory for the LLM using the Qdrant vector database. It uses the Model Context Protocol (MCP) to communicate with a Qdrant MCP server.

**Features:**
-   `save_memory`: Stores information with optional metadata.
-   `search_memory`: Semantically searches for relevant information in the database.

**Prerequisites:**
1.  **Qdrant Server:** You must have a running Qdrant server instance. See [Qdrant Quickstart](https://qdrant.tech/) for installation instructions.
2.  **MCP Python Package:** Install the `mcp` library:
    ```bash
    pip install mcp
    ```
3.  **Qdrant MCP Server:** Install the Qdrant MCP server (see [mcp-server-qdrant](https://github.com/qdrant/mcp-server-qdrant)). Ensure `mcp-server-qdrant` is in your PATH.

**Environment Variables:**
The `qdrant_tools.py` module requires the following environment variables to connect to your Qdrant instance:

```bash
export QDRANT_URL="http://localhost:6333"
export QDRANT_API_KEY="your_key"
export COLLECTION_NAME="my_knowledge_base"
```

**Usage:**
Add the absolute path to `config/qdrant_tools.py` to your `tool_interfaces` parameter.
The node will load all specified tool files at startup.