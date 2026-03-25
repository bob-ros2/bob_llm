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


def _search_json(storage_dir, query, limit):
    path = os.path.join(storage_dir, 'memory.json')
    if not os.path.exists(path):
        return 'Memory is empty.'

    with open(path, 'r') as f:
        try:
            data = json.load(f)
        except Exception:
            return 'Memory file corrupted.'

    matches = [m for m in data if query.lower() in m['content'].lower()]
    return matches[:limit] if matches else f"No matches for '{query}'"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Search memory.')
    parser.add_argument('query', help='The content to search for')
    parser.add_argument(
        '--limit',
        type=int,
        default=5,
        help='Limit the number of returned results'
    )
    args = parser.parse_args()

    skill_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    storage_dir = os.path.join(skill_dir, 'data')
    os.makedirs(storage_dir, exist_ok=True)

    res = _search_json(storage_dir, args.query, args.limit)
    if isinstance(res, list):
        print(json.dumps(res, indent=2))
    else:
        print(res)
