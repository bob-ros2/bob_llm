^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package bob_llm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1.0.3 (2026-02-10)
------------------
* Fixed prompt input to handle JSON dictionaries without a role by wrapping them as user messages and improved log extraction
* Added support for loading system_prompt from files and new system_prompt_file parameter
* Implemented dynamic parameter reconfiguration for LLM client and system prompt
* Added optional `eof` parameter to signal the end of a stream on `llm_stream`
* Added `tool_choice` parameter to dynamically control tool calling behavior
* Enhanced tool execution logging with result previews for better debugging
* Removed prefix v1 from chat API path
* Added support for [Agentskills](https://agentskills.io/) specification
* Added native Qdrant vector database tools with environment variable configuration

1.0.2 (2026-02-01)
------------------
* Full ROS 2 Rolling and Humble compliance (fixed linter issues)
* Standardized import ordering and quote usage

1.0.1 (2026-01-26)
------------------
* Fix 270+ linter and style issues for ROS2 compliance
* Fix package.xml schema validation
* Standardize docstrings and copyright headers
* Contributors: Bob Ros

1.0.0 (2026-01-26)
------------------
* Initial release of bob_llm
* Contributors: Bob Ros
