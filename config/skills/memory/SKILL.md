---
name: memory
description: Stores and retrieves information over time in a local JSON file. Use this when asked to remember facts, user preferences, or when searching for past context.
---

# memory

This skill allows the agent to maintain a persistent memory using a local JSON file. 

## Instructions

Whenever you need to remember an important fact, user preference, or specific context, use the `save.py` script.
Whenever you need to recall past context, use the `search.py` script.

The scripts are accessed via the `execute_skill_script` tool provided in your workspace.

### 1. Saving Memory
To save information, call `scripts/save.py`. It takes the content as a string argument. You can also provide optional metadata as a JSON string using the `--metadata` flag.

**Example usage via `execute_skill_script`**:
```python
execute_skill_script(
    skill_name="memory",
    script_path="scripts/save.py",
    args="'The user prefers dark mode.' --metadata '{\"category\": \"preference\"}'"
)
```

### 2. Searching Memory
To read information, call `scripts/search.py`. It takes a search query as a string argument. You can limit the results with `--limit`.

**Example usage via `execute_skill_script`**:
```python
execute_skill_script(
    skill_name="memory",
    script_path="scripts/search.py",
    args="'dark mode' --limit 2"
)
```

## Directory Structure
- `scripts/save.py`: Saves data to `data/memory.json`.
- `scripts/search.py`: Queries data from `data/memory.json`.
- `data/`: Auto-generated folder containing `memory.json`.
