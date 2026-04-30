import pytest
import rclpy

@pytest.fixture(scope="session")
def ros_context():
    """Fixture for ROS 2 context management."""
    if not rclpy.ok():
        rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()
