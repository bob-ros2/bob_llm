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
    Retrieve the current status of the robot.

    This function checks the robot's battery level and joint states.
    """
    # In a real scenario, this would query robot topics or services
    return "Robot status: Battery is at 85%. All systems are nominal. Currently idle."
