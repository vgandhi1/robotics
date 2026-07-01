"""
ROS 2 node: ESP32 serial bridge.

Translates between ROS 2 topics and the ESP32's simple ASCII serial protocol.

Protocol (newline-terminated ASCII):
  Jetson → ESP32:  "CMD,<v_linear_m_s>,<v_angular_rad_s>\\n"
  ESP32  → Jetson: "ODO,<left_ticks>,<right_ticks>,<dt_ms>\\n"
                   "IMU,<ax>,<ay>,<az>,<gx>,<gy>,<gz>\\n"   (if MPU-6050 fitted)
                   "ERR,<message>\\n"

ROS topics:
  Subscribes: /cmd_vel  (geometry_msgs/Twist)
  Publishes:  /odom     (nav_msgs/Odometry)
              /imu/data (sensor_msgs/Imu)   — if IMU lines received

On serial disconnect, the node publishes zero velocity, logs a warning, and
retries the connection every 5 seconds. This prevents runaway motion if the
serial link drops mid-drive.

Parameters:
    port                str,    default /dev/ttyUSB1
    baudrate            int,    default 115200
    wheel_base_m        float,  default 0.22
    wheel_radius_m      float,  default 0.033
    encoder_ticks_rev   int,    default 1120    (28 pulse × 40 gear ratio)
    reconnect_delay_s   float,  default 5.0
    odom_frame          str,    default odom
    base_frame          str,    default base_link
    cmd_timeout_s       float,  default 0.5  (stop if no cmd received)
"""

from __future__ import annotations

import math
import threading
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from tf2_ros import TransformBroadcaster

try:
    import serial

    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


class Esp32BridgeNode(Node):
    """Serial bridge between the ROS 2 network and the ESP32 motor controller."""

    NODE_NAME = "esp32_bridge_node"

    def __init__(self) -> None:
        super().__init__(self.NODE_NAME)
        self._declare_parameters()

        self._port = self.get_parameter("port").value
        self._baudrate = self.get_parameter("baudrate").value
        self._wheel_base = self.get_parameter("wheel_base_m").value
        self._wheel_radius = self.get_parameter("wheel_radius_m").value
        self._ticks_per_rev = self.get_parameter("encoder_ticks_rev").value
        self._reconnect_delay = self.get_parameter("reconnect_delay_s").value
        self._odom_frame = self.get_parameter("odom_frame").value
        self._base_frame = self.get_parameter("base_frame").value
        self._cmd_timeout = self.get_parameter("cmd_timeout_s").value

        # Odometry state
        self._x = 0.0
        self._y = 0.0
        self._theta = 0.0
        self._last_cmd_time = time.time()

        self._serial: Optional[serial.Serial] = None
        self._serial_lock = threading.Lock()
        self._connected = False

        # QoS
        nav_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._pub_odom = self.create_publisher(Odometry, "/odom", nav_qos)
        self._pub_imu = self.create_publisher(Imu, "/imu/data", 10)
        self._tf_broadcaster = TransformBroadcaster(self)

        self._sub_cmd_vel = self.create_subscription(
            Twist, "/cmd_vel", self._cmd_vel_cb,
            QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1),
        )

        # Serial read thread
        self._read_thread = threading.Thread(target=self._serial_read_loop, daemon=True)
        self._read_thread.start()

        # Watchdog: stop robot if no cmd_vel received within timeout
        self.create_timer(0.1, self._watchdog_cb)

        self.get_logger().info("Esp32BridgeNode ready. Port: %s @ %d baud", self._port, self._baudrate)

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def _declare_parameters(self) -> None:
        self.declare_parameter("port", "/dev/ttyUSB1")
        self.declare_parameter("baudrate", 115200)
        self.declare_parameter("wheel_base_m", 0.22)
        self.declare_parameter("wheel_radius_m", 0.033)
        self.declare_parameter("encoder_ticks_rev", 1120)
        self.declare_parameter("reconnect_delay_s", 5.0)
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("cmd_timeout_s", 0.5)

    # ------------------------------------------------------------------
    # cmd_vel callback
    # ------------------------------------------------------------------

    def _cmd_vel_cb(self, msg: Twist) -> None:
        self._last_cmd_time = time.time()
        self._send_velocity(msg.linear.x, msg.angular.z)

    def _send_velocity(self, v_lin: float, v_ang: float) -> None:
        line = f"CMD,{v_lin:.4f},{v_ang:.4f}\n"
        with self._serial_lock:
            if self._serial and self._serial.is_open:
                try:
                    self._serial.write(line.encode())
                except Exception as exc:
                    self.get_logger().warn("Serial write error: %s", str(exc))
                    self._connected = False

    # ------------------------------------------------------------------
    # Watchdog
    # ------------------------------------------------------------------

    def _watchdog_cb(self) -> None:
        if (time.time() - self._last_cmd_time) > self._cmd_timeout:
            self._send_velocity(0.0, 0.0)

    # ------------------------------------------------------------------
    # Serial read loop (dedicated thread)
    # ------------------------------------------------------------------

    def _serial_read_loop(self) -> None:
        while rclpy.ok():
            if not self._connected:
                self._try_connect()
                if not self._connected:
                    time.sleep(self._reconnect_delay)
                    continue

            try:
                with self._serial_lock:
                    raw = self._serial.readline()
                line = raw.decode("ascii", errors="ignore").strip()
                if line:
                    self._parse_serial_line(line)
            except Exception as exc:
                self.get_logger().warn("Serial read error: %s — reconnecting…", str(exc))
                self._connected = False

    def _try_connect(self) -> None:
        if not SERIAL_AVAILABLE:
            self.get_logger().warn_once("pyserial not installed. Running in mock mode.")
            self._connected = True
            return
        try:
            with self._serial_lock:
                if self._serial and self._serial.is_open:
                    self._serial.close()
                self._serial = serial.Serial(
                    self._port, self._baudrate, timeout=1.0
                )
                time.sleep(0.1)  # ESP32 reset delay after DTR toggle
            self._connected = True
            self.get_logger().info("Connected to ESP32 on %s", self._port)
        except Exception as exc:
            self.get_logger().warn("Cannot connect to %s: %s", self._port, str(exc))

    # ------------------------------------------------------------------
    # Protocol parser
    # ------------------------------------------------------------------

    def _parse_serial_line(self, line: str) -> None:
        parts = line.split(",")
        if not parts:
            return

        msg_type = parts[0]

        if msg_type == "ODO" and len(parts) >= 4:
            try:
                left_ticks = int(parts[1])
                right_ticks = int(parts[2])
                dt_ms = int(parts[3])
                self._handle_odometry(left_ticks, right_ticks, dt_ms)
            except ValueError:
                self.get_logger().debug("Bad ODO line: %s", line)

        elif msg_type == "IMU" and len(parts) >= 7:
            try:
                ax, ay, az = float(parts[1]), float(parts[2]), float(parts[3])
                gx, gy, gz = float(parts[4]), float(parts[5]), float(parts[6])
                self._handle_imu(ax, ay, az, gx, gy, gz)
            except ValueError:
                self.get_logger().debug("Bad IMU line: %s", line)

        elif msg_type == "ERR":
            self.get_logger().warn("ESP32 error: %s", ",".join(parts[1:]))

    # ------------------------------------------------------------------
    # Odometry calculation
    # ------------------------------------------------------------------

    def _handle_odometry(self, left_ticks: int, right_ticks: int, dt_ms: int) -> None:
        if dt_ms <= 0:
            return

        dt_s = dt_ms / 1000.0
        meters_per_tick = (2.0 * math.pi * self._wheel_radius) / self._ticks_per_rev

        d_left = left_ticks * meters_per_tick
        d_right = right_ticks * meters_per_tick

        d_center = (d_right + d_left) / 2.0
        d_theta = (d_right - d_left) / self._wheel_base

        # Integrate pose
        delta_x = d_center * math.cos(self._theta + d_theta / 2.0)
        delta_y = d_center * math.sin(self._theta + d_theta / 2.0)
        self._x += delta_x
        self._y += delta_y
        self._theta = (self._theta + d_theta + math.pi) % (2 * math.pi) - math.pi

        v_lin = d_center / dt_s
        v_ang = d_theta / dt_s

        now = self.get_clock().now()
        now_msg = now.to_msg()

        # TF odom → base_link
        tf_msg = TransformStamped()
        tf_msg.header.stamp = now_msg
        tf_msg.header.frame_id = self._odom_frame
        tf_msg.child_frame_id = self._base_frame
        tf_msg.transform.translation.x = self._x
        tf_msg.transform.translation.y = self._y
        tf_msg.transform.translation.z = 0.0
        q = _yaw_to_quat(self._theta)
        tf_msg.transform.rotation.x = q[0]
        tf_msg.transform.rotation.y = q[1]
        tf_msg.transform.rotation.z = q[2]
        tf_msg.transform.rotation.w = q[3]
        self._tf_broadcaster.sendTransform(tf_msg)

        # Odometry message
        odom = Odometry()
        odom.header.stamp = now_msg
        odom.header.frame_id = self._odom_frame
        odom.child_frame_id = self._base_frame
        odom.pose.pose.position.x = self._x
        odom.pose.pose.position.y = self._y
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]
        odom.twist.twist.linear.x = v_lin
        odom.twist.twist.angular.z = v_ang

        # Covariance: diagonal, tuned empirically
        pose_cov = [0.0] * 36
        pose_cov[0] = 0.001   # x variance
        pose_cov[7] = 0.001   # y variance
        pose_cov[35] = 0.001  # yaw variance
        odom.pose.covariance = pose_cov

        twist_cov = [0.0] * 36
        twist_cov[0] = 0.001
        twist_cov[35] = 0.001
        odom.twist.covariance = twist_cov

        self._pub_odom.publish(odom)

    def _handle_imu(
        self, ax: float, ay: float, az: float,
        gx: float, gy: float, gz: float
    ) -> None:
        imu_msg = Imu()
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.header.frame_id = self._base_frame
        imu_msg.linear_acceleration.x = ax
        imu_msg.linear_acceleration.y = ay
        imu_msg.linear_acceleration.z = az
        imu_msg.angular_velocity.x = gx
        imu_msg.angular_velocity.y = gy
        imu_msg.angular_velocity.z = gz
        # No orientation estimate from raw IMU (fusion handled by SLAM)
        imu_msg.orientation_covariance[0] = -1.0  # flag: orientation not provided
        self._pub_imu.publish(imu_msg)


def _yaw_to_quat(yaw: float):
    """Convert a 2D yaw angle to a quaternion (x, y, z, w)."""
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Esp32BridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._send_velocity(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
