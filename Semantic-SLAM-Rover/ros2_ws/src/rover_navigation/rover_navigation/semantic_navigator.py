"""
ROS 2 node: Semantic Navigator.

Bridges the semantic landmark map to Nav2's NavigateToPose action.

Service:
  /rover/navigate_to_class  (rover_msgs/NavigateToClass)
    Request:  class_label, approach_distance, explore_if_not_found
    Response: accepted, message, landmark_id, target_position

Publishes:
  /rover/mission/status     (std_msgs/String)   — JSON status updates

Behaviour:
  1. Client calls /rover/navigate_to_class with a class label (e.g. "person").
  2. Node calls /rover/get_semantic_landmarks to check if that class has
     already been observed.
  3a. If found: sends NavigateToPose goal to Nav2 at the landmark's map
     position (offset by approach_distance along the approach vector).
  3b. If not found and explore_if_not_found=True: starts frontier
     exploration loop by repeatedly sending frontier goals until the
     landmark appears or all frontiers are exhausted.
  4. Publishes JSON status messages throughout.

Parameters:
    explore_timeout_s       float, default 300.0
    nav_timeout_s           float, default 60.0
    goal_tolerance_m        float, default 0.25
    map_frame               str,   default "map"
"""

from __future__ import annotations

import json
import math
import threading
from typing import Optional

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Quaternion
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import String

from rover_msgs.msg import SemanticMap, SemanticLandmark
from rover_msgs.srv import GetSemanticLandmarks, NavigateToClass


def _yaw_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw / 2)
    q.w = math.cos(yaw / 2)
    return q


def _bearing(x1: float, y1: float, x2: float, y2: float) -> float:
    """Bearing from (x1,y1) to (x2,y2) in radians."""
    return math.atan2(y2 - y1, x2 - x1)


class SemanticNavigator(Node):
    """Nav2 goal sender driven by semantic landmark lookups."""

    NODE_NAME = "semantic_navigator"

    def __init__(self) -> None:
        super().__init__(self.NODE_NAME)
        self._declare_parameters()

        self._map_frame = self.get_parameter("map_frame").value
        self._explore_timeout = self.get_parameter("explore_timeout_s").value
        self._nav_timeout = self.get_parameter("nav_timeout_s").value
        self._goal_tolerance = self.get_parameter("goal_tolerance_m").value

        self._nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose")

        self._get_landmarks_client = self.create_client(
            GetSemanticLandmarks, "/rover/get_semantic_landmarks"
        )

        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._pub_status = self.create_publisher(String, "/rover/mission/status", reliable_qos)

        self.create_service(
            NavigateToClass, "/rover/navigate_to_class", self._handle_navigate_to_class
        )

        self._active_goal: Optional[object] = None
        self._mission_lock = threading.Lock()

        self.get_logger().info("SemanticNavigator ready.")

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def _declare_parameters(self) -> None:
        self.declare_parameter("explore_timeout_s", 300.0)
        self.declare_parameter("nav_timeout_s", 60.0)
        self.declare_parameter("goal_tolerance_m", 0.25)
        self.declare_parameter("map_frame", "map")

    # ------------------------------------------------------------------
    # Service handler
    # ------------------------------------------------------------------

    def _handle_navigate_to_class(
        self,
        request: NavigateToClass.Request,
        response: NavigateToClass.Response,
    ) -> NavigateToClass.Response:
        class_label = request.class_label.strip()
        approach_dist = max(0.0, request.approach_distance)

        self._publish_status("searching", class_label)
        self.get_logger().info("Navigation request: '%s' approach=%.2f m", class_label, approach_dist)

        landmark = self._lookup_landmark(class_label)

        if landmark is None:
            if not request.explore_if_not_found:
                response.accepted = False
                response.message = f"Class '{class_label}' not in semantic map. explore_if_not_found=False."
                self._publish_status("not_found", class_label)
                return response
            # Kick off exploration in a background thread so this service call returns promptly
            threading.Thread(
                target=self._explore_for_class,
                args=(class_label, approach_dist),
                daemon=True,
            ).start()
            response.accepted = True
            response.message = f"Class '{class_label}' not found yet. Starting exploration."
            return response

        goal_pose = self._landmark_to_goal_pose(landmark, approach_dist)
        self._send_nav_goal(goal_pose, class_label)

        response.accepted = True
        response.message = f"Navigating to '{class_label}' at ({landmark.position.x:.2f}, {landmark.position.y:.2f})"
        response.landmark_id = landmark.landmark_id
        response.target_position = landmark.position
        return response

    # ------------------------------------------------------------------
    # Landmark lookup
    # ------------------------------------------------------------------

    def _lookup_landmark(self, class_label: str, max_age_s: float = 60.0) -> Optional[SemanticLandmark]:
        if not self._get_landmarks_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("GetSemanticLandmarks service not available.")
            return None

        req = GetSemanticLandmarks.Request()
        req.class_label = class_label
        req.max_age_seconds = max_age_s

        future = self._get_landmarks_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)

        if not future.done():
            self.get_logger().warn("GetSemanticLandmarks call timed out.")
            return None

        result = future.result()
        if not result.found or not result.landmarks:
            return None

        # Return the most-observed landmark
        return max(result.landmarks, key=lambda lm: lm.observation_count)

    # ------------------------------------------------------------------
    # Goal pose construction
    # ------------------------------------------------------------------

    def _landmark_to_goal_pose(self, landmark: SemanticLandmark, approach_dist: float) -> PoseStamped:
        """
        Place the goal pose approach_dist meters away from the landmark,
        with the robot facing the landmark.
        """
        # For approach, we need the robot's current position — approximate
        # by backing off along the landmark's stored observation bearing.
        lx = landmark.position.x
        ly = landmark.position.y

        # Without knowing robot position, use landmark position and offset
        # backward along the map X-axis as a safe default.
        # A production system would compute this from the current robot pose.
        goal_x = lx - approach_dist
        goal_y = ly
        yaw = _bearing(goal_x, goal_y, lx, ly)

        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = self._map_frame
        pose.pose.position.x = goal_x
        pose.pose.position.y = goal_y
        pose.pose.position.z = 0.0
        pose.pose.orientation = _yaw_to_quaternion(yaw)
        return pose

    # ------------------------------------------------------------------
    # Nav2 action client
    # ------------------------------------------------------------------

    def _send_nav_goal(self, goal_pose: PoseStamped, label: str) -> None:
        if not self._nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("NavigateToPose action server not available.")
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose

        self.get_logger().info(
            "Sending Nav2 goal: (%.2f, %.2f) for class '%s'",
            goal_pose.pose.position.x, goal_pose.pose.position.y, label,
        )
        self._publish_status("navigating", label, {
            "x": goal_pose.pose.position.x,
            "y": goal_pose.pose.position.y,
        })

        send_future = self._nav_client.send_goal_async(
            goal_msg,
            feedback_callback=lambda fb: self._nav_feedback_cb(fb, label),
        )
        send_future.add_done_callback(
            lambda f: self._nav_goal_accepted_cb(f, label)
        )

    def _nav_goal_accepted_cb(self, future, label: str) -> None:
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Nav2 goal rejected for class '%s'.", label)
            self._publish_status("rejected", label)
            return

        self._active_goal = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda f: self._nav_result_cb(f, label)
        )

    def _nav_feedback_cb(self, feedback_msg, label: str) -> None:
        fb = feedback_msg.feedback
        remaining = getattr(fb, "distance_remaining", None)
        if remaining is not None:
            self.get_logger().debug("Navigating to '%s': %.2f m remaining", label, remaining)

    def _nav_result_cb(self, future, label: str) -> None:
        result = future.result()
        status = result.status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("Arrived at '%s'.", label)
            self._publish_status("arrived", label)
        else:
            self.get_logger().warn("Navigation to '%s' failed (status=%d).", label, status)
            self._publish_status("failed", label, {"nav2_status": status})
        self._active_goal = None

    # ------------------------------------------------------------------
    # Exploration loop
    # ------------------------------------------------------------------

    def _explore_for_class(self, class_label: str, approach_dist: float) -> None:
        """
        Frontier-based search: send random frontier goals while checking
        the landmark map periodically. Terminates when the class is found
        or timeout expires.

        Production note: Replace the hardcoded frontier list with a proper
        frontier detection algorithm (e.g. subscribe to /map and extract
        unknown-space edges). This placeholder demonstrates the control flow.
        """
        import time
        deadline = time.time() + self._explore_timeout
        self.get_logger().info("Exploring for class '%s' (timeout %.0f s)…", class_label, self._explore_timeout)
        self._publish_status("exploring", class_label)

        # Frontier waypoints — replace with dynamic frontier detection
        exploration_waypoints = [
            (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0),
            (2.0, 1.0), (1.0, 2.0), (-2.0, 1.0), (-1.0, -2.0),
            (3.0, 0.0), (0.0, 3.0), (-3.0, 0.0), (0.0, -3.0),
        ]

        for wx, wy in exploration_waypoints:
            if time.time() > deadline:
                break

            # Check if landmark appeared while navigating
            landmark = self._lookup_landmark(class_label)
            if landmark is not None:
                self.get_logger().info("Found '%s' during exploration!", class_label)
                goal_pose = self._landmark_to_goal_pose(landmark, approach_dist)
                self._send_nav_goal(goal_pose, class_label)
                return

            # Send exploration waypoint
            wp_pose = PoseStamped()
            wp_pose.header.stamp = self.get_clock().now().to_msg()
            wp_pose.header.frame_id = self._map_frame
            wp_pose.pose.position.x = wx
            wp_pose.pose.position.y = wy
            wp_pose.pose.orientation.w = 1.0

            self.get_logger().info("Exploring waypoint (%.1f, %.1f)…", wx, wy)
            self._send_nav_goal(wp_pose, f"explore_{class_label}")
            time.sleep(8.0)  # Wait for rover to reach approximate waypoint

        landmark = self._lookup_landmark(class_label)
        if landmark is not None:
            goal_pose = self._landmark_to_goal_pose(landmark, approach_dist)
            self._send_nav_goal(goal_pose, class_label)
        else:
            self.get_logger().warn("Exploration complete. Class '%s' not found.", class_label)
            self._publish_status("exploration_failed", class_label)

    # ------------------------------------------------------------------
    # Status publisher
    # ------------------------------------------------------------------

    def _publish_status(self, state: str, class_label: str, extra: Optional[dict] = None) -> None:
        payload = {"state": state, "class": class_label}
        if extra:
            payload.update(extra)
        msg = String()
        msg.data = json.dumps(payload)
        self._pub_status.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SemanticNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
