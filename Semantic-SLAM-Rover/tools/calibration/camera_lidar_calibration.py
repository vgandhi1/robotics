#!/usr/bin/env python3
"""
Camera–LiDAR extrinsic calibration utility.

Measures the horizontal yaw offset between the camera optical axis and
the LiDAR forward direction by pointing both sensors at a retroreflective
target (e.g. a corner cube reflector or a white board) and comparing:
  - The camera pixel column of the target centroid → horizontal angle via intrinsics
  - The LiDAR return with minimum range (brightest retroreflective return)

Outputs a calibrated yaw_offset_deg to be pasted into rover_params.yaml.

Usage (ROS 2 bag replay or live):
  python3 camera_lidar_calibration.py \
      --camera-info /camera/camera_info \
      --scan /scan \
      --image /camera/image_raw \
      --target-color red   # or use --aruco for ArUco marker detection

Calibration procedure:
  1. Place a bright retroreflective target 1–3 m in front of the robot,
     centred in the camera frame.
  2. Run this script. It will:
     a. Subscribe to the camera and LiDAR.
     b. Detect the target in the image (color blob or ArUco marker).
     c. Find the minimum-range return in the LiDAR scan (target).
     d. Compute the angular difference and print the offset.
  3. Run for 30+ frames and use the median offset for best accuracy.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ROS 2 Python imports — available when running with 'ros2 run' or after sourcing ROS
try:
    import rclpy
    from rclpy.node import Node
    from cv_bridge import CvBridge
    from sensor_msgs.msg import CameraInfo, Image, LaserScan

    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("ROS 2 not sourced. Running in offline/demo mode.")


# ── Color detection ────────────────────────────────────────────────────────────

COLOR_RANGES = {
    "red":    ([0, 120, 70],  [10, 255, 255],  [170, 120, 70], [180, 255, 255]),
    "green":  ([36, 100, 100], [86, 255, 255],  None,           None),
    "blue":   ([100, 150, 50], [140, 255, 255], None,           None),
    "white":  ([0, 0, 200],    [180, 30, 255],  None,           None),
}


def detect_color_blob_centroid(
    bgr: np.ndarray, color: str
) -> Optional[Tuple[float, float]]:
    """
    Find the centroid of the largest blob of the given color in HSV space.
    Returns (cx, cy) in pixels or None if not found.
    """
    if color not in COLOR_RANGES:
        raise ValueError(f"Unknown color: {color}. Choices: {list(COLOR_RANGES)}")

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo1, hi1, lo2, hi2 = COLOR_RANGES[color]
    mask = cv2.inRange(hsv, np.array(lo1), np.array(hi1))
    if lo2 is not None:
        mask |= cv2.inRange(hsv, np.array(lo2), np.array(hi2))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 200:
        return None

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None
    return M["m10"] / M["m00"], M["m01"] / M["m00"]


def detect_aruco_centroid(bgr: np.ndarray) -> Optional[Tuple[float, float]]:
    """Detect the first ArUco marker (4x4_50 dict) and return its centroid."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _ = detector.detectMarkers(bgr)
    if ids is None or len(corners) == 0:
        return None
    c = corners[0][0]
    return float(c[:, 0].mean()), float(c[:, 1].mean())


# ── Calibration math ───────────────────────────────────────────────────────────

def pixel_to_camera_angle(u: float, fx: float, cx: float) -> float:
    """Horizontal angle of pixel u in camera frame (radians)."""
    return math.atan2(u - cx, fx)


def lidar_min_range_angle(
    ranges: list, angle_min: float, angle_increment: float,
    range_min: float, range_max: float,
    search_window_deg: float = 60.0,
) -> Optional[float]:
    """
    Find the angle of the minimum-range LiDAR return within ±search_window_deg
    of the forward direction (angle = 0). Returns angle in radians.
    """
    window_rad = math.radians(search_window_deg)
    best_r = float("inf")
    best_angle = None

    for i, r in enumerate(ranges):
        if math.isnan(r) or math.isinf(r):
            continue
        if not (range_min <= r <= range_max):
            continue
        angle = angle_min + i * angle_increment
        if abs(angle) > window_rad:
            continue
        if r < best_r:
            best_r = r
            best_angle = angle

    return best_angle


# ── ROS 2 calibration node ─────────────────────────────────────────────────────

if ROS_AVAILABLE:
    class CalibrationNode(Node):
        def __init__(self, target_color: str, use_aruco: bool, n_samples: int) -> None:
            super().__init__("camera_lidar_calibration")
            self.bridge = CvBridge()
            self.target_color = target_color
            self.use_aruco = use_aruco
            self.n_samples = n_samples

            self._fx: Optional[float] = None
            self._cx: Optional[float] = None
            self._latest_scan: Optional[LaserScan] = None
            self._offsets: List[float] = []

            from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
            sensor_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST, depth=1,
            )

            self.create_subscription(CameraInfo, "/camera/camera_info", self._info_cb, 1)
            self.create_subscription(LaserScan, "/scan", self._scan_cb, sensor_qos)
            self.create_subscription(Image, "/camera/image_raw", self._image_cb, sensor_qos)

            self.get_logger().info(
                "Calibration node ready. Point the target at the robot and wait…"
            )

        def _info_cb(self, msg: CameraInfo) -> None:
            self._fx = msg.k[0]
            self._cx = msg.k[2]

        def _scan_cb(self, msg: LaserScan) -> None:
            self._latest_scan = msg

        def _image_cb(self, msg: Image) -> None:
            if self._fx is None or self._latest_scan is None:
                return
            if len(self._offsets) >= self.n_samples:
                self._report()
                rclpy.shutdown()
                return

            bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            if self.use_aruco:
                centroid = detect_aruco_centroid(bgr)
            else:
                centroid = detect_color_blob_centroid(bgr, self.target_color)

            if centroid is None:
                self.get_logger().warn_throttle(2.0, "Target not detected in image.")
                return

            u, _ = centroid
            cam_angle = pixel_to_camera_angle(u, self._fx, self._cx)

            scan = self._latest_scan
            lidar_angle = lidar_min_range_angle(
                list(scan.ranges),
                scan.angle_min,
                scan.angle_increment,
                scan.range_min,
                scan.range_max,
            )

            if lidar_angle is None:
                self.get_logger().warn_throttle(2.0, "LiDAR target not detected.")
                return

            offset_deg = math.degrees(cam_angle - lidar_angle)
            self._offsets.append(offset_deg)
            self.get_logger().info(
                "Sample %d/%d: cam_angle=%.2f°  lidar_angle=%.2f°  offset=%.2f°",
                len(self._offsets), self.n_samples,
                math.degrees(cam_angle), math.degrees(lidar_angle), offset_deg,
            )

        def _report(self) -> None:
            median_offset = statistics.median(self._offsets)
            stdev = statistics.stdev(self._offsets) if len(self._offsets) > 1 else 0.0
            self.get_logger().info(
                "\n"
                "═══════════════════════════════════════════════\n"
                "  CALIBRATION RESULT (%d samples)\n"
                "  Median yaw offset: %.3f°\n"
                "  Std dev:           %.3f°\n"
                "\n"
                "  Add to rover_params.yaml:\n"
                "    fusion_node:\n"
                "      ros__parameters:\n"
                "        camera_yaw_offset_deg: %.3f\n"
                "═══════════════════════════════════════════════",
                len(self._offsets), median_offset, stdev, median_offset,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Camera–LiDAR yaw calibration")
    parser.add_argument("--target-color", default="red", choices=list(COLOR_RANGES),
                        help="Color of calibration target")
    parser.add_argument("--aruco", action="store_true", help="Use ArUco marker instead of color blob")
    parser.add_argument("--samples", type=int, default=30, help="Number of measurement samples")
    args = parser.parse_args()

    if not ROS_AVAILABLE:
        print("ROS 2 not available. Exiting.")
        sys.exit(1)

    rclpy.init()
    node = CalibrationNode(args.target_color, args.aruco, args.samples)
    rclpy.spin(node)


if __name__ == "__main__":
    main()
