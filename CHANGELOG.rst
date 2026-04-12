^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package bob_llm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1.0.3 (2026-04-13)
------------------
* Optimized streaming latency by implementing raw byte stream parsing using a unified loop
* Integrated tool call detection directly into reasoning stream to eliminate pre-check delay
* Restored 100% flake8/PEP8 compliance for ROS 2 Rolling and Humble CI (enforced single quotes)
* Refactored main interaction loop to ensure robust synchronous execution without ROS executor deadlocks
* Fixed prompt input to handle JSON dictionaries without a role by wrapping them as user messages and improved log extraction
* Added support for loading system_prompt from files and new system_prompt_file parameter
* Implemented dynamic parameter reconfiguration for LLM client and system prompt
* Added optional `eof` parameter to signal the end of a stream on `llm_stream`
* Added `tool_choice` parameter to dynamically control tool calling behavior
* Enhanced tool execution logging with result previews for better debugging
* Removed prefix v1 from chat API path
* Added support for [Agentskills](https://agentskills.io/) specification
* Added native Qdrant vector database tools with environment variable configuration
* Refactored Agent Skills implementation to strictly follow progressive disclosure
* Fixed Race Condition in LLMNode initialization by pre-initializing llm_client
* Implemented soft limit for tool calls with system hint for final response
* Added `llm_reasoning` topic to support live reasoning/thinking content from models (e.g., Gemma 2, DeepSeek)
* Updated `OpenAICompatibleClient` to extract `reasoning_content` from both stream chunks and blocking responses
* Enhanced tool safety in `ros_cli_tools.py` by enforcing mandatory discovery of topics, services, and parameters in docstrings
* Improved type safety in `backend_clients` with proper Tuple annotations and fixed linter issues
* Added premium interactive terminal chat client with Markdown and optional boxed UI
* Cleaned up legacy scripts and modernized README documentation
* Contributors: Bob Ros

1.0.2 (2026-02-01)
------------------
* Full ROS 2 Rolling and Humble compliance (fixed linter issues)
* Standardized import ordering and quote usage
* Contributors: Bob Ros

1.0.1 (2026-01-26)
------------------
* Fix 270+ linter and style issues for ROS2 compliance
* Fix package.xml schema validation
* Standardize docstrings and copyright headers
* Contributors: Bob Ros

1.0.0 (2025-11-25)
------------------
* Initial release of bob_llm
* Contributors: Bob Ros
