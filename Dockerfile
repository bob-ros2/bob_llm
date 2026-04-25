# -- bob_llm Dockerfile --
ARG ROS_DISTRO=humble
FROM ros:${ROS_DISTRO}-ros-base

ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash

# -- Install System Dependencies --
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    git \
    && rm -rf /var/lib/apt/lists/*

# -- ROS 2 Workspace Setup --
WORKDIR /ros2_ws
RUN mkdir -p /ros2_ws/src

# -- Copy current local repo into workspace --
COPY . /ros2_ws/src/bob_llm

# -- Install dependencies --
RUN apt-get update && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y --rosdistro ${ROS_DISTRO} --skip-keys "python3-requests python3-yaml" && \
    if [ -f /ros2_ws/src/bob_llm/requirements.txt ]; then pip3 install --no-cache-dir -r /ros2_ws/src/bob_llm/requirements.txt; fi && \
    rm -rf /var/lib/apt/lists/*

# -- Build the Workspace --
RUN . /opt/ros/${ROS_DISTRO}/setup.sh && \
    colcon build --symlink-install --packages-select bob_llm

# -- Entrypoint Script --
RUN echo "#!/bin/bash\nsource /opt/ros/\${ROS_DISTRO}/setup.bash\nsource /ros2_ws/install/setup.bash\nexec \"\$@\"" > /ros_entrypoint.sh \
    && chmod +x /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]

CMD ["ros2", "run", "bob_llm", "llm_node"]
