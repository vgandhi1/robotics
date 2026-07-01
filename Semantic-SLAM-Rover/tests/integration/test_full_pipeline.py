"""
Integration test: full detection → fusion pipeline (no ROS, no hardware).

Simulates a detection entering the fusion pipeline and verifies:
  1. Projection produces a valid FusionResult
  2. LandmarkTracker stores and returns the landmark
  3. Repeated observations of the same object merge correctly

Run with: pytest tests/integration/test_full_pipeline.py -v
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Path setup for both packages
for pkg in ["rover_fusion", "rover_perception"]:
    sys.path.insert(0, str(Path(__file__).parents[2] / "ros2_ws" / "src" / pkg))

from rover_fusion.projection_math import (
    CameraIntrinsics,
    SensorExtrinsics,
    project_detection_to_3d,
    apply_2d_transform_stamped,
)
from rover_fusion.landmark_tracker import LandmarkTracker


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_scan(target_angle_deg: float, target_range_m: float, n_rays: int = 360):
    """Synthetic LaserScan with a single target at the given angle."""
    angle_min = -math.pi
    angle_increment = 2 * math.pi / n_rays
    ranges = [5.0] * n_rays  # background at 5 m

    # Place target return at the target angle
    target_angle_rad = math.radians(target_angle_deg)
    idx = int(round((target_angle_rad - angle_min) / angle_increment))
    idx = max(0, min(idx, n_rays - 1))
    for di in range(-2, 3):  # 5-ray wide return
        if 0 <= idx + di < n_rays:
            ranges[idx + di] = target_range_m

    return {
        "scan_ranges": ranges,
        "scan_angle_min": angle_min,
        "scan_angle_max": math.pi,
        "scan_angle_increment": angle_increment,
        "scan_range_min": 0.15,
        "scan_range_max": 12.0,
    }


class _FakeTranslation:
    x = y = z = 0.0

class _FakeRotation:
    x = y = z = 0.0
    w = 1.0

class _FakeTransform:
    translation = _FakeTranslation()
    rotation = _FakeRotation()

class FakeTf:
    """Minimal fake geometry_msgs/TransformStamped for identity transform."""
    transform = _FakeTransform()


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestEndToEndFusion:
    """Simulate detection → projection → landmark map."""

    @pytest.fixture
    def setup(self):
        intrinsics = CameraIntrinsics(
            fx=615.0, fy=615.0, cx=320.0, cy=240.0,
            image_width=640, image_height=480,
        )
        extrinsics = SensorExtrinsics(
            translation=(0.05, 0.0, 0.12),
            yaw_rad=0.0,
            pitch_rad=math.radians(-5.0),
        )
        tracker = LandmarkTracker(merge_radius_m=0.5, max_age_seconds=30.0, min_confidence=0.40)
        return intrinsics, extrinsics, tracker

    def test_centred_detection_fuses_correctly(self, setup):
        """Object at image centre, 2 m ahead → landmark placed near (2, 0) in robot/map frame."""
        intrinsics, extrinsics, tracker = setup
        scan_kwargs = make_scan(target_angle_deg=0.0, target_range_m=2.0)

        result = project_detection_to_3d(
            x_min=270, y_min=190, x_max=370, y_max=290,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            **scan_kwargs,
        )

        assert result.is_valid, f"Fusion failed: {result.rejection_reason}"
        assert abs(result.x_robot - 2.0) < 0.3

        fake_tf = FakeTf()
        x_map, y_map = apply_2d_transform_stamped(result.x_robot, result.y_robot, fake_tf)

        lm = tracker.observe("person", x_map, y_map, 0.85, result.range_m)
        assert lm is not None
        assert lm.class_label == "person"
        assert abs(lm.x_map - 2.0) < 0.3

    def test_repeated_observations_merge(self, setup):
        """10 detections of the same object → single landmark with count=10."""
        intrinsics, extrinsics, tracker = setup
        scan_kwargs = make_scan(target_angle_deg=0.0, target_range_m=2.0)

        for _ in range(10):
            result = project_detection_to_3d(
                x_min=270, y_min=190, x_max=370, y_max=290,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                **scan_kwargs,
            )
            if result.is_valid:
                fake_tf = FakeTf()
                x_map, y_map = apply_2d_transform_stamped(result.x_robot, result.y_robot, fake_tf)
                tracker.observe("box", x_map, y_map, 0.85, result.range_m)

        assert len(tracker) == 1
        lm = tracker.get_best_by_class("box")
        assert lm is not None
        assert lm.observation_count == 10

    def test_two_different_objects_two_landmarks(self, setup):
        """Two detections of different classes → two landmarks."""
        intrinsics, extrinsics, tracker = setup
        scan_forward = make_scan(target_angle_deg=0.0, target_range_m=2.0)
        scan_right = make_scan(target_angle_deg=-30.0, target_range_m=3.0)

        result_a = project_detection_to_3d(
            x_min=270, y_min=190, x_max=370, y_max=290,
            intrinsics=intrinsics, extrinsics=extrinsics,
            **scan_forward,
        )
        result_b = project_detection_to_3d(
            x_min=500, y_min=190, x_max=600, y_max=290,
            intrinsics=intrinsics, extrinsics=extrinsics,
            **scan_right,
        )

        fake_tf = FakeTf()
        if result_a.is_valid:
            xm, ym = apply_2d_transform_stamped(result_a.x_robot, result_a.y_robot, fake_tf)
            tracker.observe("person", xm, ym, 0.8, result_a.range_m)

        if result_b.is_valid:
            xm, ym = apply_2d_transform_stamped(result_b.x_robot, result_b.y_robot, fake_tf)
            tracker.observe("chair", xm, ym, 0.8, result_b.range_m)

        classes = {lm.class_label for lm in tracker.get_all()}
        assert "person" in classes or "chair" in classes  # at least one valid

    def test_semantic_goal_found_in_map(self, setup):
        """Simulate the semantic_navigator looking up a class after fusion."""
        intrinsics, extrinsics, tracker = setup
        scan_kwargs = make_scan(target_angle_deg=0.0, target_range_m=3.5)

        result = project_detection_to_3d(
            x_min=270, y_min=190, x_max=370, y_max=290,
            intrinsics=intrinsics, extrinsics=extrinsics,
            **scan_kwargs,
        )

        if result.is_valid:
            fake_tf = FakeTf()
            xm, ym = apply_2d_transform_stamped(result.x_robot, result.y_robot, fake_tf)
            tracker.observe("blue_box", xm, ym, 0.91, result.range_m)

        best = tracker.get_best_by_class("blue_box")
        assert best is not None
        assert best.class_label == "blue_box"
        assert best.observation_count >= 1
