^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package bob_llm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1.0.3 (2026-02-10)
------------------
* Fixed prompt input to handle JSON dictionaries without a role by wrapping them as user messages and improved log extraction
* Added robust skill folder handling (scans for subdirectories with handler.py or __init__.py)
* Implemented collision protection for tool registration with mandatory logging
* Improved tool loading logic to comply with ROS 2 (Ament) linting standards (E501)
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
* Fix package.xml schema validation
* Standardize docstrings and copyright headers
* Contributors: Bob Ros

1.0.0 (2025-11-25)
------------------
* Initial release of bob_llm
* Contributors: Bob Ros
