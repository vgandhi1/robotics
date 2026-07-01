"""
ROS 2 node: Camera-LiDAR semantic fusion node.

Subscribes to:
  /rover/detections       (rover_msgs/Detection2DArray)
  /scan                   (sensor_msgs/LaserScan)
  /camera/camera_info     (sensor_msgs/CameraInfo)

Publishes:
  /rover/semantic/landmarks   (rover_msgs/SemanticMap)
  /rover/semantic/markers     (visualization_msgs/MarkerArray)  — RViz display

Service servers:
  /rover/get_semantic_landmarks  (rover_msgs/GetSemanticLandmarks)

The fusion node time-synchronizes detection messages with the most recent
LiDAR scan (approximate sync, ≤100 ms age tolerance). For each detection
it calls the projection pipeline in projection_math.py, then updates the
landmark tracker. A 1 Hz timer publishes the full semantic map and prunes
stale landmarks.

Parameters:
    merge_radius_m          float, default 0.5
    max_landmark_age_s      float, default 30.0
    min_confidence          float, default 0.40
    min_range_m             float, default 0.30
    map_frame               str, default "map"
    robot_frame             str, default "base_link"
    tf_timeout_s            float, default 0.10
    camera_yaw_offset_deg   float, default 0.0
    camera_pitch_offset_deg float, default -5.0
    camera_tx               float, default 0.05
    camera_ty               float, default 0.0
    camera_tz               float, default 0.12
    publish_rate_hz         float, default 1.0
    prune_rate_hz           float, default 0.2
"""

from __future__ import annotations

import math
import threading
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.duration import Duration

from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import Point
from sensor_msgs.msg import CameraInfo, LaserScan
from visualization_msgs.msg import Marker, MarkerArray

from rover_msgs.msg import Detection2DArray, SemanticLandmark, SemanticMap
from rover_msgs.srv import GetSemanticLandmarks

from .projection_math import (
    CameraIntrinsics,
    SensorExtrinsics,
    project_detection_to_3d,
    apply_2d_transform_stamped,
)
from .landmark_tracker import LandmarkTracker


class FusionNode(Node):
    """Semantic sensor fusion: camera detections × LiDAR → 3D landmark map."""

    NODE_NAME = "fusion_node"

    def __init__(self) -> None:
        super().__init__(self.NODE_NAME)
        self._declare_parameters()

        # State
        self._latest_scan: Optional[LaserScan] = None
        self._scan_lock = threading.Lock()
        self._camera_info: Optional[CameraInfo] = None
        self._intrinsics: Optional[CameraIntrinsics] = None

        params = self._read_params()
        self._extrinsics = SensorExtrinsics(
            translation=(params["camera_tx"], params["camera_ty"], params["camera_tz"]),
            yaw_rad=math.radians(params["camera_yaw_offset_deg"]),
            pitch_rad=math.radians(params["camera_pitch_offset_deg"]),
        )
        self._map_frame = params["map_frame"]
        self._robot_frame = params["robot_frame"]
        self._tf_timeout_s = params["tf_timeout_s"]
        self._min_range_m = params["min_range_m"]

        self._tracker = LandmarkTracker(
            merge_radius_m=params["merge_radius_m"],
            max_age_seconds=params["max_landmark_age_s"],
            min_confidence=params["min_confidence"],
        )

        # TF buffer
        from tf2_ros import Buffer, TransformListener
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # Subscribers
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(LaserScan, "/scan", self._scan_cb, sensor_qos)
        self.create_subscription(
            CameraInfo, "/camera/camera_info", self._camera_info_cb, 1
        )
        self.create_subscription(
            Detection2DArray, "/rover/detections", self._detections_cb, 10
        )

        # Publishers
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._pub_landmarks = self.create_publisher(
            SemanticMap, "/rover/semantic/landmarks", reliable_qos
        )
        self._pub_markers = self.create_publisher(
            MarkerArray, "/rover/semantic/markers", reliable_qos
        )

        # Services
        self.create_service(
            GetSemanticLandmarks,
            "/rover/get_semantic_landmarks",
            self._handle_get_landmarks,
        )

        # Timers
        self.create_timer(1.0 / params["publish_rate_hz"], self._publish_map)
        self.create_timer(1.0 / params["prune_rate_hz"], self._prune_landmarks)

        self.get_logger().info("FusionNode ready. TF: %s → %s", self._robot_frame, self._map_frame)

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def _declare_parameters(self) -> None:
        self.declare_parameter("merge_radius_m", 0.5)
        self.declare_parameter("max_landmark_age_s", 30.0)
        self.declare_parameter("min_confidence", 0.40)
        self.declare_parameter("min_range_m", 0.30)
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("robot_frame", "base_link")
        self.declare_parameter("tf_timeout_s", 0.10)
        self.declare_parameter("camera_yaw_offset_deg", 0.0)
        self.declare_parameter("camera_pitch_offset_deg", -5.0)
        self.declare_parameter("camera_tx", 0.05)
        self.declare_parameter("camera_ty", 0.0)
        self.declare_parameter("camera_tz", 0.12)
        self.declare_parameter("publish_rate_hz", 1.0)
        self.declare_parameter("prune_rate_hz", 0.2)

    def _read_params(self) -> dict:
        return {k: self.get_parameter(k).value for k in [
            "merge_radius_m", "max_landmark_age_s", "min_confidence",
            "min_range_m", "map_frame", "robot_frame", "tf_timeout_s",
            "camera_yaw_offset_deg", "camera_pitch_offset_deg",
            "camera_tx", "camera_ty", "camera_tz",
            "publish_rate_hz", "prune_rate_hz",
        ]}

    # ------------------------------------------------------------------
    # Subscribers
    # ------------------------------------------------------------------

    def _scan_cb(self, msg: LaserScan) -> None:
        with self._scan_lock:
            self._latest_scan = msg

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        if self._intrinsics is not None:
            return  # Already calibrated; camera_info is static
        self._camera_info = msg
        self._intrinsics = CameraIntrinsics.from_camera_info_k(
            list(msg.k), msg.width, msg.height
        )
        self.get_logger().info(
            "Camera intrinsics received: fx=%.1f fy=%.1f cx=%.1f cy=%.1f  %dx%d",
            self._intrinsics.fx, self._intrinsics.fy,
            self._intrinsics.cx, self._intrinsics.cy,
            self._intrinsics.image_width, self._intrinsics.image_height,
        )

    def _detections_cb(self, msg: Detection2DArray) -> None:
        if self._intrinsics is None:
            self.get_logger().warn_once("Waiting for camera_info to build intrinsics…")
            return

        with self._scan_lock:
            scan = self._latest_scan

        if scan is None:
            self.get_logger().warn_throttle(5.0, "No LiDAR scan received yet.")
            return

        # Check scan age
        scan_age_s = (
            self.get_clock().now() - rclpy.time.Time.from_msg(scan.header.stamp)
        ).nanoseconds / 1e9
        if scan_age_s > 0.15:
            self.get_logger().warn_throttle(
                5.0, "LiDAR scan too old (%.2f s), skipping fusion.", scan_age_s
            )
            return

        # TF lookup: base_link → map at detection timestamp
        try:
            tf_stamped = self._tf_buffer.lookup_transform(
                self._map_frame,
                self._robot_frame,
                rclpy.time.Time.from_msg(msg.header.stamp),
                timeout=Duration(seconds=self._tf_timeout_s),
            )
        except Exception as exc:
            self.get_logger().warn_throttle(
                2.0, "TF lookup failed: %s", str(exc)
            )
            return

        fused_count = 0
        for det in msg.detections:
            result = project_detection_to_3d(
                x_min=det.x_min,
                y_min=det.y_min,
                x_max=det.x_max,
                y_max=det.y_max,
                intrinsics=self._intrinsics,
                extrinsics=self._extrinsics,
                scan_ranges=list(scan.ranges),
                scan_angle_min=scan.angle_min,
                scan_angle_max=scan.angle_max,
                scan_angle_increment=scan.angle_increment,
                scan_range_min=scan.range_min,
                scan_range_max=scan.range_max,
                min_detection_range=self._min_range_m,
            )

            if not result.is_valid:
                self.get_logger().debug(
                    "Fusion rejected '%s': %s", det.class_label, result.rejection_reason
                )
                continue

            x_map, y_map = apply_2d_transform_stamped(
                result.x_robot, result.y_robot, tf_stamped
            )

            lm = self._tracker.observe(
                class_label=det.class_label,
                x_map=x_map,
                y_map=y_map,
                confidence=det.confidence,
                range_m=result.range_m,
            )
            if lm is not None:
                fused_count += 1

        if fused_count > 0:
            self.get_logger().debug(
                "Fused %d/%d detections into landmark map (total: %d)",
                fused_count, len(msg.detections), len(self._tracker),
            )

    # ------------------------------------------------------------------
    # Publisher callbacks
    # ------------------------------------------------------------------

    def _publish_map(self) -> None:
        landmarks = self._tracker.get_all()
        now = self.get_clock().now().to_msg()

        sem_map = SemanticMap()
        sem_map.header.stamp = now
        sem_map.header.frame_id = self._map_frame
        sem_map.total_landmarks = len(landmarks)

        marker_array = MarkerArray()

        for lm in landmarks:
            sl = SemanticLandmark()
            sl.header.stamp = now
            sl.header.frame_id = self._map_frame
            sl.class_label = lm.class_label
            sl.confidence = lm.confidence
            sl.position = Point(x=lm.x_map, y=lm.y_map, z=0.0)
            sl.range = lm.range_m
            sl.observation_count = lm.observation_count
            sl.landmark_id = lm.landmark_id

            # Convert float timestamps to builtin_interfaces/Time
            sl.first_seen.sec = int(lm.first_seen)
            sl.first_seen.nanosec = int((lm.first_seen % 1) * 1e9)
            sl.last_seen.sec = int(lm.last_seen)
            sl.last_seen.nanosec = int((lm.last_seen % 1) * 1e9)

            sem_map.landmarks.append(sl)

            # RViz sphere marker
            m = Marker()
            m.header = sl.header
            m.ns = "semantic_landmarks"
            m.id = abs(hash(lm.landmark_id)) % (2**31)
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position = sl.position
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.25
            m.color.a = 0.9
            color = _class_color(lm.class_label)
            m.color.r, m.color.g, m.color.b = color
            m.lifetime.sec = 5
            marker_array.markers.append(m)

            # Text label marker
            t = Marker()
            t.header = sl.header
            t.ns = "semantic_labels"
            t.id = abs(hash(lm.landmark_id + "_text")) % (2**31)
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position = Point(x=lm.x_map, y=lm.y_map, z=0.35)
            t.pose.orientation.w = 1.0
            t.scale.z = 0.18
            t.color.a = 1.0
            t.color.r = t.color.g = t.color.b = 1.0
            t.text = f"{lm.class_label}\n×{lm.observation_count}"
            t.lifetime.sec = 5
            marker_array.markers.append(t)

        self._pub_landmarks.publish(sem_map)
        self._pub_markers.publish(marker_array)

    def _prune_landmarks(self) -> None:
        removed = self._tracker.prune_stale()
        if removed > 0:
            self.get_logger().info("Pruned %d stale landmarks. Remaining: %d", removed, len(self._tracker))

    # ------------------------------------------------------------------
    # Service handlers
    # ------------------------------------------------------------------

    def _handle_get_landmarks(
        self,
        request: GetSemanticLandmarks.Request,
        response: GetSemanticLandmarks.Response,
    ) -> GetSemanticLandmarks.Response:
        max_age = request.max_age_seconds if request.max_age_seconds > 0 else None
        now = self.get_clock().now().to_msg()

        if request.class_label:
            entries = self._tracker.get_by_class(request.class_label, max_age)
        else:
            entries = self._tracker.get_all()

        response.found = len(entries) > 0
        for lm in entries:
            sl = SemanticLandmark()
            sl.header.stamp = now
            sl.header.frame_id = self._map_frame
            sl.class_label = lm.class_label
            sl.confidence = lm.confidence
            sl.position = Point(x=lm.x_map, y=lm.y_map, z=0.0)
            sl.range = lm.range_m
            sl.observation_count = lm.observation_count
            sl.landmark_id = lm.landmark_id
            response.landmarks.append(sl)

        return response


def _class_color(class_label: str):
    """Deterministic RGB color from class name hash."""
    h = abs(hash(class_label))
    r = ((h & 0xFF0000) >> 16) / 255.0
    g = ((h & 0x00FF00) >> 8) / 255.0
    b = (h & 0x0000FF) / 255.0
    # Boost brightness so color is visible on dark RViz background
    scale = 0.7 / max(r, g, b, 0.01)
    return min(r * scale, 1.0), min(g * scale, 1.0), min(b * scale, 1.0)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
