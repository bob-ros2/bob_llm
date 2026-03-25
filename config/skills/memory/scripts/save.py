#!/usr/bin/env python3
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

import argparse
import json
import os


def save_memory(storage_dir, content: str, metadata: dict = None):
    path = os.path.join(storage_dir, 'memory.json')
    data = []

    if os.path.exists(path):
        with open(path, 'r') as f:
            try:
                data = json.load(f)
            except Exception:
                data = []

    data.append({'content': content, 'metadata': metadata or {}})

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    return 'Memory stored in local JSON file.'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Save information to memory.')
    parser.add_argument('content', help='The content to remember')
    parser.add_argument('--metadata', default='{}', help='Optional metadata as JSON string')
    args = parser.parse_args()

    skill_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    storage_dir = os.path.join(skill_dir, 'data')
    os.makedirs(storage_dir, exist_ok=True)

    try:
        parsed_metadata = json.loads(args.metadata)
    except json.JSONDecodeError:
        print('Error: Invalid JSON string provided for metadata.')
        parsed_metadata = {}

    result = save_memory(storage_dir, args.content, parsed_metadata)
    print(result)
