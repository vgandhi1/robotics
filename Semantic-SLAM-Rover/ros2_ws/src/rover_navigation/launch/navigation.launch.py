"""
Launch file: Nav2 stack + semantic navigator.

Starts:
  - Nav2 (bringup_launch via IncludeLaunchDescription)
  - semantic_navigator node

Usage:
  ros2 launch rover_navigation navigation.launch.py
  ros2 launch rover_navigation navigation.launch.py use_sim_time:=true map:=/path/to/map.yaml
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_rover_nav = get_package_share_directory("rover_navigation")
    pkg_nav2_bringup = get_package_share_directory("nav2_bringup")

    nav2_params_file = os.path.join(pkg_rover_nav, "config", "nav2_params.yaml")

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time", default_value="false"
    )
    map_arg = DeclareLaunchArgument(
        "map", default_value="",
        description="Path to map YAML for pure localization (leave empty for SLAM mode)",
    )

    # Nav2 bringup — SLAM mode (no map file needed, slam_toolbox provides /map)
    nav2_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_nav2_bringup, "launch", "navigation_launch.py")
        ),
        launch_arguments={
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "params_file": nav2_params_file,
        }.items(),
    )

    semantic_navigator_node = Node(
        package="rover_navigation",
        executable="semantic_navigator",
        name="semantic_navigator",
        output="screen",
        parameters=[{
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "map_frame": "map",
        }],
    )

    return LaunchDescription([
        use_sim_time_arg,
        map_arg,
        nav2_bringup,
        semantic_navigator_node,
    ])
