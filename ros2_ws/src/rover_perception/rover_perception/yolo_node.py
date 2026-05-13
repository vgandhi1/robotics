"""
ROS 2 node: YOLO perception node.

Subscribes to /camera/image_raw (sensor_msgs/Image).
Runs YOLOv8 inference via TensorRT (Jetson) or Ultralytics PyTorch (dev machine).
Publishes rover_msgs/Detection2DArray to /rover/detections.
Publishes an annotated sensor_msgs/Image to /rover/detections/debug_image.

Parameters (ros2 param set):
    engine_path         path to .engine file (TRT) or .pt file (PyTorch)
    use_tensorrt        bool, default True on Jetson
    conf_threshold      float, default 0.45
    nms_threshold       float, default 0.50
    input_width         int, default 640
    input_height        int, default 640
    camera_topic        str, default /camera/image_raw
    publish_debug_image bool, default True
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from rover_msgs.msg import Detection2D, Detection2DArray

from .tensorrt_engine import TensorRTEngine, UltralyticsEngine, TRT_AVAILABLE

logger = logging.getLogger(__name__)


class YoloNode(Node):
    """
    Perception node: subscribes to camera frames, runs YOLO, publishes detections.
    """

    NODE_NAME = "yolo_node"

    def __init__(self) -> None:
        super().__init__(self.NODE_NAME)

        self._declare_parameters()
        params = self._get_parameters()

        self.bridge = CvBridge()
        self._engine = self._build_engine(params)
        self._frame_count = 0

        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._sub_image = self.create_subscription(
            Image,
            params["camera_topic"],
            self._image_callback,
            camera_qos,
        )

        det_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._pub_detections = self.create_publisher(
            Detection2DArray, "/rover/detections", det_qos
        )

        if params["publish_debug_image"]:
            self._pub_debug = self.create_publisher(
                Image, "/rover/detections/debug_image", 1
            )
        else:
            self._pub_debug = None

        self.get_logger().info(
            "YoloNode ready. Backend: %s  Topic: %s",
            "TensorRT" if isinstance(self._engine, TensorRTEngine) else "Ultralytics",
            params["camera_topic"],
        )

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    def _declare_parameters(self) -> None:
        self.declare_parameter("engine_path", "yolov8n.engine")
        self.declare_parameter("use_tensorrt", TRT_AVAILABLE)
        self.declare_parameter("conf_threshold", 0.45)
        self.declare_parameter("nms_threshold", 0.50)
        self.declare_parameter("input_width", 640)
        self.declare_parameter("input_height", 640)
        self.declare_parameter("camera_topic", "/camera/image_raw")
        self.declare_parameter("publish_debug_image", True)

    def _get_parameters(self) -> dict:
        return {
            "engine_path": self.get_parameter("engine_path").value,
            "use_tensorrt": self.get_parameter("use_tensorrt").value,
            "conf_threshold": self.get_parameter("conf_threshold").value,
            "nms_threshold": self.get_parameter("nms_threshold").value,
            "input_width": self.get_parameter("input_width").value,
            "input_height": self.get_parameter("input_height").value,
            "camera_topic": self.get_parameter("camera_topic").value,
            "publish_debug_image": self.get_parameter("publish_debug_image").value,
        }

    def _build_engine(self, params: dict):
        input_shape = (params["input_height"], params["input_width"])
        if params["use_tensorrt"] and TRT_AVAILABLE:
            try:
                return TensorRTEngine(
                    params["engine_path"],
                    input_shape=input_shape,
                    conf_threshold=params["conf_threshold"],
                    nms_threshold=params["nms_threshold"],
                )
            except Exception as exc:
                self.get_logger().error(
                    "TensorRT engine load failed (%s), falling back to Ultralytics.", str(exc)
                )
        return UltralyticsEngine(
            model_path=params["engine_path"],
            input_shape=input_shape,
            conf_threshold=params["conf_threshold"],
            nms_threshold=params["nms_threshold"],
        )

    # ------------------------------------------------------------------
    # Callback
    # ------------------------------------------------------------------

    def _image_callback(self, msg: Image) -> None:
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error("cv_bridge conversion failed: %s", str(exc))
            return

        detections, latency_ms = self._engine.infer(bgr)
        self._frame_count += 1

        if self._frame_count % 30 == 0:
            self.get_logger().info(
                "Frame %d: %d detections in %.1f ms",
                self._frame_count,
                len(detections),
                latency_ms,
            )

        det_array = Detection2DArray()
        det_array.header = msg.header
        det_array.inference_latency_ms = float(latency_ms)

        for d in detections:
            det_msg = Detection2D()
            det_msg.header = msg.header
            det_msg.class_label = d.class_label
            det_msg.confidence = d.confidence
            det_msg.x_min = d.x_min
            det_msg.y_min = d.y_min
            det_msg.x_max = d.x_max
            det_msg.y_max = d.y_max
            det_msg.image_width = d.image_width
            det_msg.image_height = d.image_height
            det_msg.track_id = -1
            det_array.detections.append(det_msg)

        self._pub_detections.publish(det_array)

        if self._pub_debug is not None:
            debug_img = self._draw_detections(bgr.copy(), detections)
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
            debug_msg.header = msg.header
            self._pub_debug.publish(debug_msg)

    @staticmethod
    def _draw_detections(img: np.ndarray, detections) -> np.ndarray:
        for d in detections:
            cv2.rectangle(img, (d.x_min, d.y_min), (d.x_max, d.y_max), (0, 255, 0), 2)
            label = f"{d.class_label} {d.confidence:.2f}"
            cv2.putText(
                img, label,
                (d.x_min, max(d.y_min - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )
        return img


def main(args=None) -> None:
    rclpy.init(args=args)
    node = YoloNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
