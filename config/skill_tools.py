# Copyright 2026 Bob Ros
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A tool interface for the LLM node providing Skill functionalities."""

import os
import shlex
import subprocess
from typing import Any, List

from bob_llm.tool_utils import register as default_register
from bob_llm.tool_utils import Tool
import rclpy


class _NodeContext:
    """Private class to hold the ROS node instance."""

    node: rclpy.node.Node = None


def register(module: Any, node: Any = None) -> List[Tool]:
    """
    Inspect a module and build a list of tool definitions.

    :param module: The module to inspect.
    :param node: The ROS node instance.
    :return: A list of tool definitions.
    """
    _NodeContext.node = node
    return default_register(module, node)


def _get_skill_dirs() -> List[str]:
    """Get the list of skill directories from ROS parameters or use default."""
    paths_str = './config/skills'
    if _NodeContext.node:
        try:
            param = _NodeContext.node.get_parameter('skill_dir')
            if param and param.value:
                paths_str = param.value
        except rclpy.exceptions.ParameterNotDeclaredException:
            pass

    # Split by comma and remove empty strings/whitespace
    return [p.strip() for p in paths_str.split(',') if p.strip()]


# --- Tool Definitions ---

def list_skills() -> str:
    """List all available skills across all skill directories."""
    skill_dirs = _get_skill_dirs()
    all_skills = set()

    # Collect unique skills from all configured directories
    for skill_dir in skill_dirs:
        if os.path.exists(skill_dir):
            try:
                for d in os.listdir(skill_dir):
                    if os.path.isdir(os.path.join(skill_dir, d)):
                        all_skills.add(d)
            except Exception:
                pass

    if not all_skills:
        return f'No skills found in directories: {", ".join(skill_dirs)}'

    # Return sorted list for consistent output
    return f'Available skills: {", ".join(sorted(all_skills))}'


def read_skill_file(skill_name: str, filename: str) -> str:
    r"""
    Read a file from a specific skill. Useful for reading SKILL.md or scripts.

    :param skill_name: The name of the skill.
    :param filename: The name of the file to read (e.g. \'SKILL.md\').
    """
    skill_dirs = _get_skill_dirs()

    # Search through directories in order; return the first match
    for skill_dir in skill_dirs:
        path = os.path.join(skill_dir, skill_name, filename)
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                return f'Error reading file: {e}'

    return (f'Error: File {filename} not found in skill {skill_name} '
            'across any configured directory.')


def write_skill_file(skill_name: str, filename: str, content: str) -> str:
    """
    Create or overwrite a file in a specific skill directory.

    This can be used to generate new skills.

    :param skill_name: The name of the skill.
    :param filename: The name of the file to write.
    :param content: The content to write to the file.
    """
    skill_dirs = _get_skill_dirs()
    if not skill_dirs:
        return 'Error: No skill directories configured.'

    # If multiple paths exist, protect the 'core' skills from being overwritten.
    # The agent is only allowed to write to the very last directory in the list.
    if len(skill_dirs) > 1:
        for core_dir in skill_dirs[:-1]:
            core_path = os.path.join(core_dir, skill_name)
            if os.path.exists(core_path):
                return 'Error: Core skills are protected and cannot be modified.'

    # Always write to the last configured directory
    target_dir = skill_dirs[-1]
    path = os.path.join(target_dir, skill_name)

    # Check for write permissions in the target directory
    if os.path.exists(target_dir) and not os.access(target_dir, os.W_OK):
        return (f"Error: Directory '{target_dir}' is READ-ONLY. "
                'To enable writing skills, either change permissions or '
                "set the 'skill_dir' parameter to a writable path.")

    try:
        os.makedirs(path, exist_ok=True)
        # Handle subdirectories in filename
        file_path = os.path.join(path, filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        # Make scripts executable
        if filename.endswith('.sh') or filename.endswith('.py') or filename.startswith('scripts/'):
            os.chmod(file_path, 0o755)

        return f'Successfully wrote {filename} in skill {skill_name}.'
    except Exception as e:
        return f'Error writing file: {e}'


def execute_skill_script(skill_name: str, script_path: str, args: str = '') -> str:
    r"""
    Execute a script belonging to a skill.

    :param skill_name: The name of the skill.
    :param script_path: The relative path to the script within the skill (e.g. \'scripts/run.sh\').
    :param args: Optional arguments to pass to the script as a single string.
    """
    skill_dirs = _get_skill_dirs()

    full_path = None
    # Search through directories in order for the executable script
    for skill_dir in skill_dirs:
        potential_path = os.path.join(skill_dir, skill_name, script_path)
        if os.path.exists(potential_path):
            full_path = potential_path
            break

    if not full_path:
        return f'Error: Script {script_path} not found in skill {skill_name}.'

    try:
        cmd = [full_path]
        if args:
            cmd.extend(shlex.split(args))

        # Execute it
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120.0
        )

        output = result.stdout.strip()
        if result.returncode != 0:
            output += f'\nError (Code {result.returncode}): {result.stderr.strip()}'
        return output
    except Exception as e:
        return f'Error executing script: {e}'


def apply_skill(skill_name: str) -> str:
    """Apply an existing skill by retrieving its SKILL.md definition."""
    return read_skill_file(skill_name, 'SKILL.md')
