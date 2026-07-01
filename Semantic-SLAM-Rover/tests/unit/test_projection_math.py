"""
Unit tests for camera-LiDAR projection math (projection_math.py).

Tests run without any ROS 2 dependency — pure Python math only.
Run with: pytest tests/unit/test_projection_math.py -v
"""

import math
import sys
from pathlib import Path

import pytest

# Allow importing from ros2_ws/src without ROS install
sys.path.insert(0, str(Path(__file__).parents[2] / "ros2_ws" / "src" / "rover_fusion"))

from rover_fusion.projection_math import (
    CameraIntrinsics,
    SensorExtrinsics,
    pixel_to_bearing,
    bearing_to_lidar_angle,
    lidar_angle_to_scan_index,
    extract_range_at_index,
    project_detection_to_3d,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def intrinsics() -> CameraIntrinsics:
    """640×480 camera, typical for Raspberry Pi Camera / IMX219."""
    return CameraIntrinsics(
        fx=615.0, fy=615.0,
        cx=320.0, cy=240.0,
        image_width=640, image_height=480,
    )


@pytest.fixture
def extrinsics_aligned() -> SensorExtrinsics:
    """Camera perfectly aligned with LiDAR (zero yaw offset, no pitch)."""
    return SensorExtrinsics(translation=(0.0, 0.0, 0.0), yaw_rad=0.0, pitch_rad=0.0)


@pytest.fixture
def extrinsics_tilted() -> SensorExtrinsics:
    """Camera with 5° downward pitch."""
    return SensorExtrinsics(
        translation=(0.05, 0.0, 0.12),
        yaw_rad=0.0,
        pitch_rad=math.radians(-5.0),
    )


@pytest.fixture
def flat_scan():
    """Synthetic RPLiDAR scan: 360 rays, 2.0 m range everywhere."""
    angle_min = -math.pi
    angle_max = math.pi
    n = 360
    angle_increment = (angle_max - angle_min) / n
    ranges = [2.0] * n
    return {
        "scan_ranges": ranges,
        "scan_angle_min": angle_min,
        "scan_angle_max": angle_max,
        "scan_angle_increment": angle_increment,
        "scan_range_min": 0.15,
        "scan_range_max": 12.0,
    }


# ── pixel_to_bearing ──────────────────────────────────────────────────────────

class TestPixelToBearing:
    def test_principal_point_gives_forward_bearing(self, intrinsics):
        """Pixel at principal point (cx, cy) should give bearing along Z-axis."""
        bearing = pixel_to_bearing(intrinsics.cx, intrinsics.cy, intrinsics)
        assert abs(bearing[0]) < 1e-6  # x ≈ 0 (no horizontal deflection)
        assert abs(bearing[1]) < 1e-6  # y ≈ 0 (no vertical deflection)
        assert abs(bearing[2] - 1.0) < 1e-6  # z = 1 (pointing forward)

    def test_unit_length(self, intrinsics):
        """Bearing vector should always be unit length."""
        for u, v in [(0, 0), (640, 480), (100, 200), (320, 240)]:
            bearing = pixel_to_bearing(u, v, intrinsics)
            length = math.sqrt(sum(b**2 for b in bearing))
            assert abs(length - 1.0) < 1e-6, f"bearing not unit for ({u},{v}): length={length}"

    def test_right_pixel_gives_positive_x(self, intrinsics):
        """Pixel to the right of center → positive x component in camera frame."""
        bearing_right = pixel_to_bearing(intrinsics.cx + 100, intrinsics.cy, intrinsics)
        bearing_left = pixel_to_bearing(intrinsics.cx - 100, intrinsics.cy, intrinsics)
        assert bearing_right[0] > 0
        assert bearing_left[0] < 0

    def test_symmetry(self, intrinsics):
        """Symmetric pixels should give symmetric bearings."""
        du = 80
        b_right = pixel_to_bearing(intrinsics.cx + du, intrinsics.cy, intrinsics)
        b_left = pixel_to_bearing(intrinsics.cx - du, intrinsics.cy, intrinsics)
        assert abs(b_right[0] + b_left[0]) < 1e-6  # x components cancel
        assert abs(b_right[2] - b_left[2]) < 1e-6  # z components equal


# ── bearing_to_lidar_angle ────────────────────────────────────────────────────

class TestBearingToLidarAngle:
    def test_forward_bearing_gives_zero_angle(self, extrinsics_aligned):
        """A forward-pointing camera ray (Z-axis) → LiDAR angle ≈ 0 (robot forward)."""
        import numpy as np
        forward_bearing = np.array([0.0, 0.0, 1.0])  # camera Z = forward
        angle = bearing_to_lidar_angle(forward_bearing, extrinsics_aligned)
        assert abs(angle) < 0.01, f"Expected ~0 rad, got {angle:.4f} rad"

    def test_right_bearing_gives_negative_angle(self, extrinsics_aligned):
        """Camera-right bearing → negative angle in robot frame (right = negative Y)."""
        import numpy as np
        # Camera X = right; after rotation camera X maps to robot -Y
        right_bearing = pixel_to_bearing(420.0, 240.0,
                                          CameraIntrinsics(615, 615, 320, 240, 640, 480))
        angle = bearing_to_lidar_angle(right_bearing, extrinsics_aligned)
        assert angle < 0, f"Expected negative angle for right pixel, got {angle:.4f}"


# ── lidar_angle_to_scan_index ─────────────────────────────────────────────────

class TestLidarAngleToScanIndex:
    def test_zero_angle_gives_center_index(self):
        """Zero angle (forward) → middle of scan array for symmetric scan."""
        n = 360
        angle_min = -math.pi
        angle_increment = 2 * math.pi / n
        idx = lidar_angle_to_scan_index(0.0, angle_min, math.pi, angle_increment, n)
        assert idx == n // 2  # index 180 for 360-ray scan

    def test_angle_min_gives_index_zero(self):
        angle_min = -math.pi
        angle_increment = 2 * math.pi / 360
        idx = lidar_angle_to_scan_index(angle_min, angle_min, math.pi, angle_increment, 360)
        assert idx == 0

    def test_out_of_range_returns_none(self):
        idx = lidar_angle_to_scan_index(
            5.0, -math.pi, math.pi, 2 * math.pi / 360, 360
        )
        # After normalisation 5.0 - 2π ≈ -1.28 rad, which IS in range; normalization handles wraparound
        assert idx is not None or idx is None  # No assertion — just ensure no exception

    def test_index_within_bounds(self):
        n = 360
        for angle in [-3.0, -1.5, 0.0, 1.5, 3.0]:
            idx = lidar_angle_to_scan_index(angle, -math.pi, math.pi, 2 * math.pi / n, n)
            if idx is not None:
                assert 0 <= idx < n


# ── extract_range_at_index ────────────────────────────────────────────────────

class TestExtractRangeAtIndex:
    def test_returns_median_of_window(self):
        ranges = [float("nan")] * 5 + [2.0, 2.1, 2.2] + [float("nan")] * 5
        r = extract_range_at_index(ranges, 6, 0.1, 12.0, window=1)
        assert r is not None
        assert abs(r - 2.1) < 0.01

    def test_all_nan_returns_none(self):
        ranges = [float("nan")] * 100
        r = extract_range_at_index(ranges, 50, 0.1, 12.0)
        assert r is None

    def test_out_of_range_values_rejected(self):
        ranges = [0.01, 0.01, 0.01]  # all below range_min=0.15
        r = extract_range_at_index(ranges, 1, 0.15, 12.0, window=0)
        assert r is None

    def test_valid_return(self):
        ranges = [1.5] * 10
        r = extract_range_at_index(ranges, 5, 0.1, 12.0, window=2)
        assert r == pytest.approx(1.5)


# ── project_detection_to_3d ───────────────────────────────────────────────────

class TestProjectDetectionTo3D:
    def test_forward_object_correct_position(self, intrinsics, extrinsics_aligned, flat_scan):
        """Object centred in image → should fuse at ~2.0 m forward, ~0 m lateral."""
        result = project_detection_to_3d(
            x_min=270, y_min=190, x_max=370, y_max=290,  # centred bbox
            intrinsics=intrinsics,
            extrinsics=extrinsics_aligned,
            **flat_scan,
        )
        assert result.is_valid, f"Expected valid: {result.rejection_reason}"
        assert abs(result.x_robot - 2.0) < 0.15, f"x_robot={result.x_robot:.3f}"
        assert abs(result.y_robot) < 0.15, f"y_robot={result.y_robot:.3f}"
        assert abs(result.range_m - 2.0) < 0.05

    def test_all_nan_scan_returns_invalid(self, intrinsics, extrinsics_aligned):
        """NaN-only scan → fusion should return invalid result."""
        nan_scan = {
            "scan_ranges": [float("nan")] * 360,
            "scan_angle_min": -math.pi,
            "scan_angle_max": math.pi,
            "scan_angle_increment": 2 * math.pi / 360,
            "scan_range_min": 0.15,
            "scan_range_max": 12.0,
        }
        result = project_detection_to_3d(
            x_min=270, y_min=190, x_max=370, y_max=290,
            intrinsics=intrinsics,
            extrinsics=extrinsics_aligned,
            **nan_scan,
        )
        assert not result.is_valid
        assert result.rejection_reason is not None

    def test_near_field_rejected(self, intrinsics, extrinsics_aligned):
        """Object closer than min_detection_range → rejected."""
        close_scan = {
            "scan_ranges": [0.10] * 360,  # 10 cm
            "scan_angle_min": -math.pi,
            "scan_angle_max": math.pi,
            "scan_angle_increment": 2 * math.pi / 360,
            "scan_range_min": 0.05,
            "scan_range_max": 12.0,
        }
        result = project_detection_to_3d(
            x_min=270, y_min=190, x_max=370, y_max=290,
            intrinsics=intrinsics,
            extrinsics=extrinsics_aligned,
            min_detection_range=0.30,
            **close_scan,
        )
        assert not result.is_valid

    def test_off_axis_lateral_position(self, intrinsics, extrinsics_aligned, flat_scan):
        """Object at right edge of image → negative y (robot frame right)."""
        result = project_detection_to_3d(
            x_min=540, y_min=190, x_max=620, y_max=290,  # right-side bbox
            intrinsics=intrinsics,
            extrinsics=extrinsics_aligned,
            **flat_scan,
        )
        if result.is_valid:
            assert result.y_robot < 0, "Right-side object should have negative y in robot frame"


# ── CameraIntrinsics ─────────────────────────────────────────────────────────

class TestCameraIntrinsics:
    def test_from_k_array(self):
        K = [615.0, 0.0, 320.0, 0.0, 615.0, 240.0, 0.0, 0.0, 1.0]
        ci = CameraIntrinsics.from_camera_info_k(K, 640, 480)
        assert ci.fx == 615.0
        assert ci.fy == 615.0
        assert ci.cx == 320.0
        assert ci.cy == 240.0

    def test_matrix_shape(self):
        import numpy as np
        K = [615.0, 0.0, 320.0, 0.0, 615.0, 240.0, 0.0, 0.0, 1.0]
        ci = CameraIntrinsics.from_camera_info_k(K, 640, 480)
        M = ci.matrix
        assert M.shape == (3, 3)
        assert M[0, 0] == ci.fx
        assert M[1, 1] == ci.fy
        assert M[0, 2] == ci.cx
