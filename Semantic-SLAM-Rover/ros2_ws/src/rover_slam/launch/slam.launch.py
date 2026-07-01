"""
Launch file: SLAM Toolbox (async mapping mode).

Starts:
  - slam_toolbox async SLAM
  - RPLiDAR ROS 2 driver (optional, enabled by 'use_lidar' arg)
  - Static TF for lidar → base_link

Usage:
  ros2 launch rover_slam slam.launch.py
  ros2 launch rover_slam slam.launch.py use_lidar:=false  # sim / bag replay
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    pkg_rover_slam = get_package_share_directory("rover_slam")

    slam_params_file = os.path.join(pkg_rover_slam, "config", "slam_toolbox_params.yaml")

    use_lidar_arg = DeclareLaunchArgument(
        "use_lidar",
        default_value="true",
        description="Launch RPLiDAR driver (false for bag replay/simulation)",
    )
    lidar_port_arg = DeclareLaunchArgument(
        "lidar_port",
        default_value="/dev/ttyUSB0",
        description="RPLiDAR serial port",
    )
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulated clock (rosbag replay)",
    )

    # RPLiDAR driver node
    rplidar_node = Node(
        package="rplidar_ros",
        executable="rplidar_node",
        name="rplidar_node",
        parameters=[{
            "serial_port": LaunchConfiguration("lidar_port"),
            "serial_baudrate": 115200,
            "frame_id": "laser",
            "angle_compensate": True,
            "scan_mode": "Standard",
        }],
        condition=IfCondition(LaunchConfiguration("use_lidar")),
    )

    # Static TF: lidar frame → base_link
    # Adjust xyz/rpy to match your physical LiDAR mounting position.
    lidar_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="lidar_to_base_tf",
        arguments=[
            "0.0", "0.0", "0.05",   # x y z (LiDAR 5 cm above base_link origin)
            "0", "0", "0",           # roll pitch yaw
            "base_link", "laser",
        ],
    )

    # Camera static TF: camera_link → base_link
    # Front-facing camera, 12 cm up, 5 cm forward
    camera_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="camera_to_base_tf",
        arguments=[
            "0.05", "0.0", "0.12",
            "0", "-0.0873", "0",    # slight downward pitch (-5°)
            "base_link", "camera_link",
        ],
    )

    # SLAM Toolbox (async mode — does not block sensor callbacks)
    slam_node = Node(
        package="slam_toolbox",
        executable="async_slam_toolbox_node",
        name="slam_toolbox",
        output="screen",
        parameters=[
            slam_params_file,
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
    )

    return LaunchDescription([
        use_lidar_arg,
        lidar_port_arg,
        use_sim_time_arg,
        rplidar_node,
        lidar_tf,
        camera_tf,
        slam_node,
    ])
