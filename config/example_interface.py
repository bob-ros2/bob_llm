
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
    Retrieves the current status of the robot.

    This function checks the robot's battery level, joint states, and current task.
    """
    # In a real scenario, this would query robot topics or services
    return "Robot status: Battery is at 85%. All systems are nominal. Currently idle."