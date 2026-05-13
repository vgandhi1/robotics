#!/bin/bash
set -e

# Source ROS 2 base setup
source /opt/ros/${ROS_DISTRO}/setup.bash

# Source workspace overlay
if [ -f /opt/ros2_ws/install/setup.bash ]; then
  source /opt/ros2_ws/install/setup.bash
fi

exec "$@"
