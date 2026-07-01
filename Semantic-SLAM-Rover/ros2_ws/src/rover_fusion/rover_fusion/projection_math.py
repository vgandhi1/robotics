"""
Camera-to-LiDAR projection math for semantic sensor fusion.

The core algorithm projects a 2D bounding box centroid (pixel space)
through the camera's intrinsic matrix into a 3D bearing vector, then
intersects that ray with a 2D LiDAR scan to recover the metric XYZ
coordinates of a detected object.

Coordinate conventions:
  - camera frame: X right, Y down, Z forward (optical axis)
  - robot/LiDAR frame: X forward, Y left, Z up
  - map frame: ROS REP 103 — X east (or forward), Y north (or left), Z up

The extrinsic transform T_lidar_camera rotates from camera frame to
LiDAR/robot frame. For a front-facing camera mounted above the LiDAR:
  - yaw: rotation around Z to align optical axis with LiDAR forward
  - pitch: rotation around Y for any tilt (camera pitched down)
  - roll: usually 0

Reference: "A Survey of Camera and LiDAR Sensor Fusion for Autonomous Driving"
           and ROS 2 tf2 transform lookup idiom.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class CameraIntrinsics:
    """
    Pinhole camera intrinsic matrix parameters.

    Obtained from 'ros2 run camera_calibration cameracalibrator'.
    Stored in camera_info.yaml or from the CameraInfo message.
    """
    fx: float   # focal length in pixels, x-axis
    fy: float   # focal length in pixels, y-axis
    cx: float   # principal point x (usually image_width / 2)
    cy: float   # principal point y (usually image_height / 2)
    image_width: int
    image_height: int

    @classmethod
    def from_camera_info_k(cls, K: list, width: int, height: int) -> "CameraIntrinsics":
        """Construct from the 9-element K array in sensor_msgs/CameraInfo."""
        return cls(
            fx=K[0], fy=K[4], cx=K[2], cy=K[5],
            image_width=width, image_height=height,
        )

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            [self.fx,      0, self.cx],
            [     0, self.fy, self.cy],
            [     0,      0,       1],
        ], dtype=np.float64)


@dataclass
class SensorExtrinsics:
    """
    Rigid-body transform from camera frame to LiDAR/robot frame.

    All values in SI units (meters, radians).
    translation: [x_forward, y_left, z_up] offset of camera origin
                 from LiDAR origin, expressed in robot frame.
    yaw_rad: rotation of camera optical axis around Z (robot Z = up).
    pitch_rad: downward tilt of camera (positive = looking down).
    """
    translation: Tuple[float, float, float] = (0.05, 0.0, 0.12)
    yaw_rad: float = 0.0
    pitch_rad: float = -0.0873  # ~-5 degrees default tilt

    def rotation_matrix(self) -> np.ndarray:
        """3x3 rotation matrix: camera optical frame → robot frame."""
        cy = math.cos(self.yaw_rad)
        sy = math.sin(self.yaw_rad)
        cp = math.cos(self.pitch_rad)
        sp = math.sin(self.pitch_rad)

        R_yaw = np.array([
            [cy, -sy, 0],
            [sy,  cy, 0],
            [ 0,   0, 1],
        ])
        R_pitch = np.array([
            [cp, 0, sp],
            [ 0, 1,  0],
            [-sp, 0, cp],
        ])
        # Camera optical convention: Z forward, X right, Y down
        # Robot convention: X forward, Y left, Z up
        # Align camera Z → robot X; camera X → robot -Y; camera Y → robot -Z
        R_cam_to_robot = np.array([
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0],
        ], dtype=np.float64)
        return R_yaw @ R_pitch @ R_cam_to_robot


@dataclass
class FusionResult:
    """Result of a single camera-LiDAR fusion operation."""
    x_robot: float      # meters, robot frame (forward)
    y_robot: float      # meters, robot frame (left)
    range_m: float      # distance in meters
    lidar_angle_rad: float
    lidar_index: int
    is_valid: bool
    rejection_reason: Optional[str] = None


def pixel_to_bearing(
    u: float,
    v: float,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """
    Convert pixel (u, v) to a unit bearing vector in camera optical frame.

    Camera optical frame: Z forward (optical axis), X right, Y down.

    Returns:
        unit vector [x, y, z] in camera frame
    """
    x_n = (u - intrinsics.cx) / intrinsics.fx
    y_n = (v - intrinsics.cy) / intrinsics.fy
    bearing_cam = np.array([x_n, y_n, 1.0])
    return bearing_cam / np.linalg.norm(bearing_cam)


def bearing_to_lidar_angle(
    bearing_cam: np.ndarray,
    extrinsics: SensorExtrinsics,
) -> float:
    """
    Rotate bearing vector from camera optical frame to LiDAR/robot frame
    and extract the horizontal (yaw) angle.

    The LiDAR 2D scan lives in the XY plane of the robot frame
    (z-component is ignored for a planar LiDAR).

    Returns:
        angle in radians, measured CCW from robot X-axis (forward)
    """
    R = extrinsics.rotation_matrix()
    bearing_robot = R @ bearing_cam

    # Horizontal angle in robot XY plane
    return math.atan2(bearing_robot[1], bearing_robot[0])


def lidar_angle_to_scan_index(
    angle_rad: float,
    angle_min: float,
    angle_max: float,
    angle_increment: float,
    num_ranges: int,
) -> Optional[int]:
    """
    Map a robot-frame bearing angle to the closest LiDAR scan array index.

    RPLiDAR convention: angle_min = -π, angle_max = +π,
    angle_increment = 2π / num_rays.

    Returns:
        integer index in [0, num_ranges-1], or None if out of scan range
    """
    # Normalize angle to [angle_min, angle_max]
    while angle_rad > angle_max:
        angle_rad -= 2 * math.pi
    while angle_rad < angle_min:
        angle_rad += 2 * math.pi

    if angle_rad < angle_min or angle_rad > angle_max:
        return None

    idx = int(round((angle_rad - angle_min) / angle_increment))
    return max(0, min(idx, num_ranges - 1))


def extract_range_at_index(
    ranges: list,
    index: int,
    range_min: float,
    range_max: float,
    window: int = 3,
) -> Optional[float]:
    """
    Extract a valid LiDAR range at the given index, with a small median
    window to suppress noise from individual bad returns.

    Args:
        ranges: LaserScan.ranges list
        index: target scan index
        range_min: LaserScan.range_min
        range_max: LaserScan.range_max
        window: number of rays on each side to consider for median

    Returns:
        range in meters, or None if no valid return found
    """
    samples = []
    for offset in range(-window, window + 1):
        idx = index + offset
        if 0 <= idx < len(ranges):
            r = ranges[idx]
            if not math.isnan(r) and not math.isinf(r):
                if range_min <= r <= range_max:
                    samples.append(r)
    if not samples:
        return None
    return float(np.median(samples))


def project_detection_to_3d(
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    intrinsics: CameraIntrinsics,
    extrinsics: SensorExtrinsics,
    scan_ranges: list,
    scan_angle_min: float,
    scan_angle_max: float,
    scan_angle_increment: float,
    scan_range_min: float,
    scan_range_max: float,
    min_detection_range: float = 0.30,
) -> FusionResult:
    """
    Full pipeline: bounding box → 3D position in robot frame.

    Args:
        x_min, y_min, x_max, y_max: bounding box in pixel coordinates
        intrinsics: camera calibration
        extrinsics: camera → LiDAR extrinsic transform
        scan_*: fields from sensor_msgs/LaserScan
        min_detection_range: minimum valid range (m), below which
                             near-field LiDAR noise is rejected

    Returns:
        FusionResult with robot-frame XY position and validity flag
    """
    # Centroid of bounding box
    u = (x_min + x_max) / 2.0
    v = (y_min + y_max) / 2.0

    bearing_cam = pixel_to_bearing(u, v, intrinsics)
    lidar_angle = bearing_to_lidar_angle(bearing_cam, extrinsics)

    idx = lidar_angle_to_scan_index(
        lidar_angle,
        scan_angle_min,
        scan_angle_max,
        scan_angle_increment,
        len(scan_ranges),
    )

    if idx is None:
        return FusionResult(
            x_robot=0.0, y_robot=0.0, range_m=0.0,
            lidar_angle_rad=lidar_angle, lidar_index=-1, is_valid=False,
            rejection_reason="angle out of scan range",
        )

    r = extract_range_at_index(
        scan_ranges, idx, scan_range_min, scan_range_max
    )

    if r is None:
        return FusionResult(
            x_robot=0.0, y_robot=0.0, range_m=0.0,
            lidar_angle_rad=lidar_angle, lidar_index=idx, is_valid=False,
            rejection_reason="no valid LiDAR return",
        )

    if r < min_detection_range:
        return FusionResult(
            x_robot=0.0, y_robot=0.0, range_m=r,
            lidar_angle_rad=lidar_angle, lidar_index=idx, is_valid=False,
            rejection_reason=f"range {r:.2f}m below floor {min_detection_range:.2f}m",
        )

    # Robot-frame XY from polar coordinates
    x_robot = r * math.cos(lidar_angle)
    y_robot = r * math.sin(lidar_angle)

    return FusionResult(
        x_robot=x_robot,
        y_robot=y_robot,
        range_m=r,
        lidar_angle_rad=lidar_angle,
        lidar_index=idx,
        is_valid=True,
    )


def apply_2d_transform_stamped(
    x_robot: float,
    y_robot: float,
    transform,  # geometry_msgs/TransformStamped
) -> Tuple[float, float]:
    """
    Apply a 2D TF transform (robot → map) to a point in robot frame.

    Only uses x, y, and yaw (rotation around Z) from the full 6DOF transform.
    Suitable for ground-plane localization from slam_toolbox.

    Args:
        x_robot, y_robot: point in robot frame
        transform: geometry_msgs/TransformStamped (base_link → map)

    Returns:
        x_map, y_map: point in map frame
    """
    tx = transform.transform.translation.x
    ty = transform.transform.translation.y

    q = transform.transform.rotation
    # Extract yaw from quaternion
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)

    x_map = cos_yaw * x_robot - sin_yaw * y_robot + tx
    y_map = sin_yaw * x_robot + cos_yaw * y_robot + ty
    return x_map, y_map
