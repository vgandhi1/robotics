"""
Unit tests for TensorRT engine pre/post-processing (no GPU required).

Tests only the CPU-side pre/post processing logic of the inference engines,
since TensorRT/PyCUDA are only available on Jetson hardware.

Run with: pytest tests/unit/test_tensorrt_engine.py -v
"""

import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "ros2_ws" / "src" / "rover_perception"))

from rover_perception.tensorrt_engine import (
    Detection,
    COCO_CLASSES,
    UltralyticsEngine,
)


class TestDetection:
    def test_fields_accessible(self):
        d = Detection(
            class_id=0, class_label="person", confidence=0.8,
            x_min=10, y_min=20, x_max=110, y_max=220,
            image_width=640, image_height=480,
        )
        assert d.class_label == "person"
        assert d.confidence == pytest.approx(0.8)
        assert d.x_max - d.x_min == 100

    def test_coco_classes_count(self):
        assert len(COCO_CLASSES) == 80

    def test_coco_first_last(self):
        assert COCO_CLASSES[0] == "person"
        assert COCO_CLASSES[-1] == "toothbrush"


class TestPreprocessing:
    """Tests for letterbox + normalization logic (no engine needed)."""

    def _make_engine_stub(self):
        """Create a UltralyticsEngine stub without loading a model."""
        class StubEngine:
            def __init__(self):
                self.input_h = 640
                self.input_w = 640
                self.conf_threshold = 0.45
                self.nms_threshold = 0.5
                self.class_names = COCO_CLASSES
            # Borrow the _preprocess method from TensorRTEngine for testing
            from rover_perception.tensorrt_engine import TensorRTEngine
            _preprocess = TensorRTEngine._preprocess
        return StubEngine()

    def test_square_image_no_padding(self):
        """640×640 input → no letterbox padding needed."""
        from rover_perception.tensorrt_engine import TensorRTEngine

        class Stub:
            input_h, input_w = 640, 640
        stub = Stub()
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        chw, scale, pad_x, pad_y, orig_w, orig_h = TensorRTEngine._preprocess(stub, img)
        assert scale == pytest.approx(1.0)
        assert pad_x == 0
        assert pad_y == 0
        assert chw.shape == (3, 640, 640)

    def test_rectangular_image_letterboxed(self):
        """320×240 input → scaled up to 640×480, padded to 640×640."""
        from rover_perception.tensorrt_engine import TensorRTEngine

        class Stub:
            input_h, input_w = 640, 640
        stub = Stub()
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        chw, scale, pad_x, pad_y, orig_w, orig_h = TensorRTEngine._preprocess(stub, img)
        assert scale == pytest.approx(2.0)
        assert pad_y == 80  # (640 - 480) / 2
        assert pad_x == 0
        assert chw.shape == (3, 640, 640)

    def test_output_normalized(self):
        """Pixels should be in [0, 1] after preprocessing."""
        from rover_perception.tensorrt_engine import TensorRTEngine

        class Stub:
            input_h, input_w = 640, 640
        stub = Stub()
        img = np.full((480, 640, 3), 255, dtype=np.uint8)
        chw, *_ = TensorRTEngine._preprocess(stub, img)
        assert chw.max() <= 1.0 + 1e-6
        assert chw.min() >= 0.0 - 1e-6


class TestUltralyticsEngineMock:
    """Test UltralyticsEngine with a mocked model (no actual GPU inference)."""

    def test_import_without_ultralytics(self, monkeypatch):
        """If ultralytics is not installed, ImportError should be raised."""
        monkeypatch.setitem(sys.modules, "ultralytics", None)
        import importlib
        import rover_perception.tensorrt_engine as trt_mod
        importlib.reload(trt_mod)

        with pytest.raises((ImportError, Exception)):
            trt_mod.UltralyticsEngine(model_path="fake.pt")

    def test_inference_returns_list(self, monkeypatch):
        """Mock ultralytics YOLO to return empty results and verify return type."""
        _raw = np.array([10, 20, 110, 220], dtype=np.float32)

        class _FakeTensor:
            """Minimal torch.Tensor stand-in: supports .cpu().numpy() chain."""
            def __init__(self, arr):
                self._arr = arr
            def cpu(self):
                return self
            def numpy(self):
                return self._arr

        class MockBox:
            cls  = type("T", (), {"item": lambda s: 0})()
            conf = type("T", (), {"item": lambda s: 0.9})()
            xyxy = [_FakeTensor(_raw)]

        class MockResult:
            boxes = [MockBox()]

        class MockYOLO:
            def __init__(self, path): pass
            def __call__(self, img, **kw): return [MockResult()]

        import rover_perception.tensorrt_engine as trt_mod
        monkeypatch.setattr(trt_mod, "TRT_AVAILABLE", False)

        mock_mod = type(sys)("ultralytics")
        mock_mod.YOLO = MockYOLO
        monkeypatch.setitem(sys.modules, "ultralytics", mock_mod)

        engine = trt_mod.UltralyticsEngine.__new__(trt_mod.UltralyticsEngine)
        engine.class_names = COCO_CLASSES
        engine.conf_threshold = 0.45
        engine.nms_threshold = 0.5
        engine.model = MockYOLO("fake")

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        dets, lat = engine.infer(img)
        assert isinstance(dets, list)
        assert isinstance(lat, float)
        assert lat > 0
