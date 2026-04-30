cmake_minimum_required(VERSION 3.8)
project(bob_llm)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)


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

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # Exclude specific linters from ament_lint_auto to avoid duplicates
  # when calling them manually below.
  list(APPEND AMENT_LINT_AUTO_EXCLUDE
    ament_cmake_copyright
    ament_cmake_flake8
    ament_cmake_pep257
  )
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_copyright REQUIRED)
  ament_copyright()

  find_package(ament_cmake_flake8 REQUIRED)
  # Explicitly lint our Python package and config directory
  ament_flake8(
    bob_llm
    config
    tests
  )

  find_package(ament_cmake_pep257 REQUIRED)
  ament_pep257(
    bob_llm
    config
    tests
  )
endif()

ament_package()
