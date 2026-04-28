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
    sudo \
    && rm -rf /var/lib/apt/lists/*

# -- Create a non-root user --
RUN useradd -m -s /bin/bash ros && \
    echo "ros ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# -- ROS 2 Workspace Setup --
WORKDIR /ros2_ws
RUN chown -R ros:ros /ros2_ws

# -- Copy current local repo into workspace --
COPY --chown=ros:ros . /ros2_ws/src/bob_llm

# -- Install dependencies --
USER ros
RUN rosdep update && \
    sudo apt-get update && \
    rosdep install --from-paths src --ignore-src -r -y --rosdistro ${ROS_DISTRO} --skip-keys "python3-requests python3-yaml" && \
    if [ -f /ros2_ws/src/bob_llm/requirements.txt ]; then pip3 install --no-cache-dir -r /ros2_ws/src/bob_llm/requirements.txt; fi && \
    sudo rm -rf /var/lib/apt/lists/*

# -- Build the Workspace --
RUN . /opt/ros/${ROS_DISTRO}/setup.sh && \
    colcon build --symlink-install --packages-select bob_llm

# -- Entrypoint Script --
USER root
RUN echo "#!/bin/bash\nsource /opt/ros/\${ROS_DISTRO}/setup.bash\nsource /ros2_ws/install/setup.bash\nexec \"\$@\"" > /ros_entrypoint.sh \
    && chmod +x /ros_entrypoint.sh

USER ros
ENTRYPOINT ["/ros_entrypoint.sh"]

CMD ["ros2", "run", "bob_llm", "llm_node"]
