"""
TensorRT inference engine wrapper for YOLOv8.

Handles engine loading, GPU buffer management, pre/post-processing,
and FP16 NMS decoding. Designed to run on NVIDIA Jetson (JetPack 6).

Usage:
    engine = TensorRTEngine('yolov8n.engine', input_shape=(640, 640))
    detections = engine.infer(bgr_image)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# TensorRT and PyCUDA are only available on Jetson; guarded import.
try:
    import tensorrt as trt
    import pycuda.autoinit  # noqa: F401 — initializes CUDA context
    import pycuda.driver as cuda

    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    logger.warning(
        "TensorRT/PyCUDA not available. "
        "TensorRTEngine will be unusable. Use UltralyticsEngine for development."
    )


@dataclass
class Detection:
    class_id: int
    class_label: str
    confidence: float
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    image_width: int
    image_height: int


# COCO-80 class names (default YOLOv8 training set)
COCO_CLASSES: List[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


class TensorRTEngine:
    """
    Wraps a TensorRT serialized engine (.engine file) for YOLOv8 inference.

    The engine must have been built from a YOLOv8 ONNX export with
    --fp16 flag. Input binding: (1, 3, H, W) FP16. Output binding:
    (1, 84, 8400) FP16 in YOLOv8 ultralytics format.
    """

    def __init__(
        self,
        engine_path: str,
        input_shape: Tuple[int, int] = (640, 640),
        conf_threshold: float = 0.45,
        nms_threshold: float = 0.5,
        class_names: Optional[List[str]] = None,
    ) -> None:
        if not TRT_AVAILABLE:
            raise RuntimeError(
                "TensorRT is not installed. Cannot instantiate TensorRTEngine."
            )
        self.input_h, self.input_w = input_shape
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.class_names = class_names or COCO_CLASSES

        engine_path = Path(engine_path)
        if not engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

        self._load_engine(engine_path)
        self._allocate_buffers()
        logger.info(
            "TensorRTEngine loaded: %s  input=%dx%d  fp16=%s",
            engine_path.name,
            self.input_w,
            self.input_h,
            self.engine.get_tensor_dtype(self.input_name) == trt.DataType.HALF,
        )

    def _load_engine(self, engine_path: Path) -> None:
        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Identify input/output tensor names
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)

    def _allocate_buffers(self) -> None:
        """Allocate pinned host memory and device memory for I/O bindings."""
        in_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
        out_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output_name))

        in_shape = (1, 3, self.input_h, self.input_w)
        out_shape = self.context.get_tensor_shape(self.output_name)

        self.h_input = cuda.pagelocked_empty(int(np.prod(in_shape)), dtype=in_dtype)
        self.h_output = cuda.pagelocked_empty(int(np.prod(out_shape)), dtype=out_dtype)
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        self.stream = cuda.Stream()
        self._out_shape = tuple(out_shape)

    def _preprocess(self, bgr_image: np.ndarray) -> Tuple[np.ndarray, float, float, int, int]:
        """
        Resize with letterbox padding → normalize to [0,1] → CHW float32/fp16.
        Returns preprocessed blob and scale/offset info for coordinate recovery.
        """
        orig_h, orig_w = bgr_image.shape[:2]
        scale = min(self.input_w / orig_w, self.input_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        pad_x = (self.input_w - new_w) // 2
        pad_y = (self.input_h - new_h) // 2

        resized = cv2.resize(bgr_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)
        canvas[pad_y: pad_y + new_h, pad_x: pad_x + new_w] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = np.ascontiguousarray(rgb.transpose(2, 0, 1))
        return chw, scale, pad_x, pad_y, orig_w, orig_h

    def _postprocess(
        self,
        raw_output: np.ndarray,
        scale: float,
        pad_x: int,
        pad_y: int,
        orig_w: int,
        orig_h: int,
    ) -> List[Detection]:
        """
        Decode YOLOv8 output tensor (1, 84, 8400) → filtered Detection list.

        YOLOv8 output layout per anchor:
          [cx, cy, w, h, cls0_score, ..., cls79_score]
        """
        preds = raw_output.reshape(self._out_shape[1], self._out_shape[2])  # (84, 8400)
        preds = preds.T  # (8400, 84)

        boxes_xywh = preds[:, :4]
        class_scores = preds[:, 4:]

        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_ids)), class_ids]

        mask = confidences >= self.conf_threshold
        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes_xywh) == 0:
            return []

        # cx,cy,w,h → x1,y1,x2,y2 (letterboxed space)
        x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

        # Remove letterbox padding and undo scale
        x1 = np.clip((x1 - pad_x) / scale, 0, orig_w)
        y1 = np.clip((y1 - pad_y) / scale, 0, orig_h)
        x2 = np.clip((x2 - pad_x) / scale, 0, orig_w)
        y2 = np.clip((y2 - pad_y) / scale, 0, orig_h)

        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy.tolist(),
            confidences.tolist(),
            self.conf_threshold,
            self.nms_threshold,
        )

        detections: List[Detection] = []
        for idx in (indices.flatten() if len(indices) > 0 else []):
            cid = int(class_ids[idx])
            label = self.class_names[cid] if cid < len(self.class_names) else str(cid)
            detections.append(
                Detection(
                    class_id=cid,
                    class_label=label,
                    confidence=float(confidences[idx]),
                    x_min=int(x1[idx]),
                    y_min=int(y1[idx]),
                    x_max=int(x2[idx]),
                    y_max=int(y2[idx]),
                    image_width=orig_w,
                    image_height=orig_h,
                )
            )
        return detections

    def infer(self, bgr_image: np.ndarray) -> Tuple[List[Detection], float]:
        """
        Run a full inference pass.

        Returns:
            detections: list of Detection objects
            latency_ms: wall-clock inference time in milliseconds
        """
        chw, scale, pad_x, pad_y, orig_w, orig_h = self._preprocess(bgr_image)

        np.copyto(self.h_input, chw.astype(self.h_input.dtype).ravel())

        t0 = time.perf_counter()
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        detections = self._postprocess(
            self.h_output.copy(), scale, pad_x, pad_y, orig_w, orig_h
        )
        return detections, latency_ms

    def __del__(self) -> None:
        try:
            del self.d_input
            del self.d_output
        except Exception:
            pass


class UltralyticsEngine:
    """
    PyTorch-backed fallback for development on x86 machines without TensorRT.
    API-compatible with TensorRTEngine.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        input_shape: Tuple[int, int] = (640, 640),
        conf_threshold: float = 0.45,
        nms_threshold: float = 0.5,
        class_names: Optional[List[str]] = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics package required: pip install ultralytics"
            ) from exc

        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.class_names = class_names or COCO_CLASSES
        logger.info("UltralyticsEngine loaded: %s", model_path)

    def infer(self, bgr_image: np.ndarray) -> Tuple[List[Detection], float]:
        t0 = time.perf_counter()
        results = self.model(bgr_image, conf=self.conf_threshold, iou=self.nms_threshold, verbose=False)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        detections: List[Detection] = []
        h, w = bgr_image.shape[:2]
        for r in results:
            for box in r.boxes:
                cid = int(box.cls.item())
                label = self.class_names[cid] if cid < len(self.class_names) else str(cid)
                xyxy = box.xyxy[0].cpu().numpy()
                detections.append(
                    Detection(
                        class_id=cid,
                        class_label=label,
                        confidence=float(box.conf.item()),
                        x_min=int(xyxy[0]),
                        y_min=int(xyxy[1]),
                        x_max=int(xyxy[2]),
                        y_max=int(xyxy[3]),
                        image_width=w,
                        image_height=h,
                    )
                )
        return detections, latency_ms
