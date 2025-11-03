cmake_minimum_required(VERSION 3.8)
project(bob_llm)

find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)

# Install scripts
install(
    PROGRAMS
    "scripts/query.sh"
    DESTINATION lib/${PROJECT_NAME}
)

# Install a sample configuration
install(
    DIRECTORY config
    DESTINATION share/${PROJECT_NAME}
)

# Install Python executables
ament_python_install_package(${PROJECT_NAME}
  SETUP_CFG
    ${PROJECT_NAME}/setup.cfg
  SCRIPTS_DESTINATION
    lib/${PROJECT_NAME}
)

ament_package()