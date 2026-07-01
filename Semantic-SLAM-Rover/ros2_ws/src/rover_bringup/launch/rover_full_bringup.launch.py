"""
Full system bringup for the Semantic SLAM Rover.

Launches all subsystems in dependency order:
  1. Hardware: ESP32 bridge (odometry, cmd_vel)
  2. Camera: v4l2_camera or CSI camera node
  3. SLAM: slam_toolbox (includes RPLiDAR + TF)
  4. Perception: YOLOv8 TensorRT node
  5. Fusion: camera-LiDAR semantic fusion node
  6. Navigation: Nav2 + semantic navigator
  7. Visualization: RViz2 (optional)

Usage:
  ros2 launch rover_bringup rover_full_bringup.launch.py
  ros2 launch rover_bringup rover_full_bringup.launch.py use_rviz:=true
  ros2 launch rover_bringup rover_full_bringup.launch.py use_sim_time:=true
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    GroupAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_bringup = get_package_share_directory("rover_bringup")
    pkg_slam = get_package_share_directory("rover_slam")
    pkg_nav = get_package_share_directory("rover_navigation")

    params_file = os.path.join(pkg_bringup, "config", "rover_params.yaml")
    rviz_config = os.path.join(pkg_bringup, "rviz", "semantic_slam.rviz")

    # ── Launch arguments ────────────────────────────────────────────────────
    use_sim_time_arg = DeclareLaunchArgument("use_sim_time", default_value="false")
    use_rviz_arg = DeclareLaunchArgument("use_rviz", default_value="false")
    use_hardware_arg = DeclareLaunchArgument(
        "use_hardware", default_value="true",
        description="Launch ESP32 bridge (false for bag replay)",
    )
    use_camera_arg = DeclareLaunchArgument(
        "use_camera", default_value="true",
        description="Launch camera node (false for bag replay)",
    )
    camera_device_arg = DeclareLaunchArgument(
        "camera_device", default_value="/dev/video0"
    )
    engine_path_arg = DeclareLaunchArgument(
        "engine_path", default_value="/opt/rover/models/yolov8n.engine"
    )

    # ── 1. Hardware (ESP32 serial bridge) ────────────────────────────────────
    esp32_node = Node(
        package="rover_hardware",
        executable="esp32_bridge_node",
        name="esp32_bridge_node",
        output="screen",
        parameters=[params_file],
        condition=IfCondition(LaunchConfiguration("use_hardware")),
    )

    # ── 2. Camera (v4l2 USB or CSI) ──────────────────────────────────────────
    camera_node = Node(
        package="v4l2_camera",
        executable="v4l2_camera_node",
        name="camera",
        output="screen",
        parameters=[{
            "video_device": LaunchConfiguration("camera_device"),
            "image_size": [640, 480],
            "pixel_format": "YUYV",
            "output_encoding": "rgb8",
            "camera_frame_id": "camera_link",
        }],
        remappings=[("/image_raw", "/camera/image_raw")],
        condition=IfCondition(LaunchConfiguration("use_camera")),
    )

    # ── 3. SLAM ──────────────────────────────────────────────────────────────
    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_slam, "launch", "slam.launch.py")
        ),
        launch_arguments={
            "use_sim_time": LaunchConfiguration("use_sim_time"),
        }.items(),
    )

    # ── 4. Perception ────────────────────────────────────────────────────────
    yolo_node = Node(
        package="rover_perception",
        executable="yolo_node",
        name="yolo_node",
        output="screen",
        parameters=[
            params_file,
            {"engine_path": LaunchConfiguration("engine_path")},
        ],
    )

    # ── 5. Fusion ────────────────────────────────────────────────────────────
    fusion_node = Node(
        package="rover_fusion",
        executable="fusion_node",
        name="fusion_node",
        output="screen",
        parameters=[params_file],
    )

    # ── 6. Navigation ────────────────────────────────────────────────────────
    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_nav, "launch", "navigation.launch.py")
        ),
        launch_arguments={
            "use_sim_time": LaunchConfiguration("use_sim_time"),
        }.items(),
    )

    # ── 7. RViz ──────────────────────────────────────────────────────────────
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config],
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_rviz")),
    )

    return LaunchDescription([
        use_sim_time_arg,
        use_rviz_arg,
        use_hardware_arg,
        use_camera_arg,
        camera_device_arg,
        engine_path_arg,
        esp32_node,
        camera_node,
        slam_launch,
        yolo_node,
        fusion_node,
        navigation_launch,
        rviz_node,
    ])
