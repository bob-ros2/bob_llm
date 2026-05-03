^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package bob_llm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


1.0.3 (2026-04-30)
------------------
* Refactor monolithic prompt processing into event-driven architecture (removed polling timer).
* Implement O(n) chat history trimming for improved performance during long conversations.
* Add robust stream recovery with automatic retry for failed API connections.
* Implement token buffering for smoother streaming and reduced ROS message overhead.
* Integrate ThreadPoolExecutor for isolated tool execution with configurable timeouts.
* Add `tool_timeout` parameter (default 60s) for robust skill execution.
* Improve multimodal content preservation (images) and smart text extraction for logging.
* Refine tool budgeting to count only successful calls against the limit.
* Add clean node shutdown logic with executor and thread cleanup.
* Add comprehensive integration test suite for LLM flow verification.
* Fix chat history reasoning issues and improve turns management.
* Implement definitive zero-latency SSE streaming via iter_lines.
* Fix UTF-8 encoding for special characters in raw byte streams.
* Optimize chat UI refresh rate for better human perception.
* Integrate tool call detection in reasoning stream for faster response.
* Restore 100% flake8/PEP8 compliance and single quote enforcement.
* Refactor main interaction loop for robust synchronous execution.
* Fix JSON prompt handling and enhance system prompt file support.
* Add support for dynamic system_prompt_file parameter loading.
* Implement dynamic parameter reconfiguration for LLM client.
* Add optional eof parameter to signal end of stream on llm_stream.
* Add tool_choice parameter for dynamic tool call control.
* Enhance tool execution logging with result previews.
* Remove prefix v1 from chat API path for standard compatibility.
* Add support for Agentskills specification and modular tools.
* Add native Qdrant vector database tools with env configuration.
* Refactor Agent Skills to follow progressive disclosure patterns.
* Fix Race Condition in LLMNode node-to-client initialization.
* Implement soft limit for tool calls with final response hint.
* Add llm_reasoning topic for live thinking content support.
* Update OpenAI client to extract reasoning_content from chunks.
* Enforce mandatory discovery in tool docstrings for safety.
* Improve type safety in backend_clients with proper annotations.
* Add premium interactive terminal chat client with boxed UI.
* Clean up legacy scripts and modernize README documentation.
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
