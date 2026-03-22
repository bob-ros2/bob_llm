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

from typing import Any, List
import os
import shlex
import subprocess
import urllib.request
import zipfile

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


def _get_skill_dir() -> str:
    """Get the skill directory from ROS parameters or use default."""
    if _NodeContext.node:
        try:
            param = _NodeContext.node.get_parameter('skill_dir')
            if param:
                return param.value
        except rclpy.exceptions.ParameterNotDeclaredException:
            pass
    return './config/skills'


# --- Tool Definitions ---

def list_skills() -> str:
    """List all available skills in the skill directory."""
    skill_dir = _get_skill_dir()
    if not os.path.exists(skill_dir):
        return f'Skill directory {skill_dir} does not exist.'
    try:
        skills = []
        for d in os.listdir(skill_dir):
            if os.path.isdir(os.path.join(skill_dir, d)):
                skills.append(d)
        if not skills:
            return 'No skills found.'
        return f'Available skills: {", ".join(skills)}'
    except Exception as e:
        return f'Error listing skills: {e}'


def read_skill_file(skill_name: str, filename: str) -> str:
    r"""
    Read a file from a specific skill. Useful for reading SKILL.md or scripts.

    :param skill_name: The name of the skill.
    :param filename: The name of the file to read (e.g. \'SKILL.md\').
    """
    skill_dir = _get_skill_dir()
    path = os.path.join(skill_dir, skill_name, filename)
    if not os.path.exists(path):
        return f'Error: File {filename} not found in skill {skill_name}.'
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f'Error reading file: {e}'


def write_skill_file(skill_name: str, filename: str, content: str) -> str:
    """
    Create or overwrite a file in a specific skill directory.

    This can be used to generate new skills.

    :param skill_name: The name of the skill.
    :param filename: The name of the file to write.
    :param content: The content to write to the file.
    """
    skill_dir = _get_skill_dir()
    path = os.path.join(skill_dir, skill_name)

    # Check for write permissions in the base directory
    if os.path.exists(skill_dir) and not os.access(skill_dir, os.W_OK):
        return (f"Error: Directory '{skill_dir}' is READ-ONLY. "
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


def download_skill(url: str, skill_name: str) -> str:
    """
    Download a skill (ZIP format) from Anthropic/Agentskills or another URL and extract it.

    :param url: The URL to the skill ZIP file.
    :param skill_name: The name under which the skill will be saved.
    """
    skill_dir = _get_skill_dir()
    path = os.path.join(skill_dir, skill_name)

    # Check for write permissions
    if os.path.exists(skill_dir) and not os.access(skill_dir, os.W_OK):
        return (f"Error: Directory '{skill_dir}' is READ-ONLY. "
                'To enable downloading skills, change permissions or '
                "set 'skill_dir' to a writable path.")

    try:
        os.makedirs(path, exist_ok=True)
        zip_path = os.path.join(path, 'downloaded.zip')
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(path)
        os.remove(zip_path)
        return f'Successfully downloaded and extracted skill {skill_name}.'
    except Exception as e:
        return f'Error downloading skill: {e}'


def execute_skill_script(skill_name: str, script_path: str, args: str = '') -> str:
    r"""
    Execute a script belonging to a skill.

    :param skill_name: The name of the skill.
    :param script_path: The relative path to the script within the skill (e.g. \'scripts/run.sh\').
    :param args: Optional arguments to pass to the script as a single string.
    """
    skill_dir = _get_skill_dir()
    full_path = os.path.join(skill_dir, skill_name, script_path)
    if not os.path.exists(full_path):
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
