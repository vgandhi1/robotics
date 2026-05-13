#!/usr/bin/env python3
"""
YOLOv8 → TensorRT FP16 engine export script.

Run on the target Jetson Orin Nano with JetPack 6 installed.
This script:
  1. Downloads/loads a YOLOv8 model from ultralytics
  2. Exports to ONNX (simplified graph)
  3. Builds a TensorRT FP16 engine using trtexec
  4. Validates the engine by running a dummy inference and reporting latency

Requirements (Jetson, JetPack 6):
    pip install ultralytics onnx onnxsim

TensorRT and trtexec come with JetPack; no additional install needed.

Usage:
    python3 export_tensorrt.py --model yolov8n --output /opt/rover/models
    python3 export_tensorrt.py --model yolov8s --input-size 640 --workspace 4096
    python3 export_tensorrt.py --model /path/to/custom.pt --output ./engines
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def export_onnx(model_name: str, input_size: int, output_dir: Path) -> Path:
    """Export a YOLOv8 model from .pt to simplified ONNX."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed: pip install ultralytics")
        sys.exit(1)

    logger.info("Loading model: %s", model_name)
    model = YOLO(model_name)

    onnx_name = Path(model_name).stem + f"_{input_size}.onnx"
    onnx_path = output_dir / onnx_name

    logger.info("Exporting to ONNX (imgsz=%d)…", input_size)
    model.export(
        format="onnx",
        imgsz=input_size,
        simplify=True,
        opset=17,
        dynamic=False,
    )

    # ultralytics exports to the same directory as the .pt file
    default_onnx = Path(model_name).with_suffix(".onnx")
    if default_onnx.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        default_onnx.rename(onnx_path)
    else:
        # Try to find the exported file
        candidates = list(Path(".").glob(f"{Path(model_name).stem}*.onnx"))
        if candidates:
            candidates[0].rename(onnx_path)
        else:
            logger.error("ONNX export file not found.")
            sys.exit(1)

    logger.info("ONNX saved to: %s  (%.1f MB)", onnx_path, onnx_path.stat().st_size / 1e6)
    return onnx_path


def build_tensorrt_engine(
    onnx_path: Path,
    output_dir: Path,
    workspace_mb: int,
    fp16: bool,
    int8: bool,
) -> Path:
    """
    Build a TensorRT engine from an ONNX file using trtexec.

    trtexec is bundled with JetPack at /usr/src/tensorrt/bin/trtexec.
    """
    engine_path = output_dir / onnx_path.stem.replace(f"_{onnx_path.stem.split('_')[-1]}", "") \
        .rstrip("_") + ".engine"
    engine_path = output_dir / (Path(onnx_path.stem).name + ".engine")

    trtexec = _find_trtexec()
    if trtexec is None:
        logger.error(
            "trtexec not found. Install TensorRT or JetPack, "
            "or add /usr/src/tensorrt/bin to PATH."
        )
        sys.exit(1)

    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--workspace={workspace_mb}",
        "--memPoolSize=workspace:{}MiB".format(workspace_mb),
    ]
    if fp16:
        cmd.append("--fp16")
    if int8:
        cmd.append("--int8")

    logger.info("Building TensorRT engine (fp16=%s int8=%s)…", fp16, int8)
    logger.info("Command: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("trtexec failed:\n%s", result.stderr[-2000:])
        sys.exit(1)

    # Parse latency from trtexec output
    for line in result.stdout.splitlines():
        if "mean" in line.lower() and "ms" in line.lower():
            logger.info("TRT benchmark: %s", line.strip())

    logger.info("Engine saved to: %s  (%.1f MB)", engine_path, engine_path.stat().st_size / 1e6)
    return engine_path


def _find_trtexec() -> str | None:
    """Search known paths for trtexec binary."""
    candidates = [
        "trtexec",
        "/usr/src/tensorrt/bin/trtexec",
        "/usr/local/bin/trtexec",
        "/opt/tensorrt/bin/trtexec",
    ]
    for c in candidates:
        result = subprocess.run(["which", c], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        if Path(c).exists():
            return c
    return None


def validate_engine(engine_path: Path, input_size: int, n_warmup: int = 5, n_runs: int = 20) -> None:
    """
    Run dummy inference through the TensorRT engine and report latency.
    Requires tensorrt and pycuda Python bindings.
    """
    try:
        import tensorrt as trt
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda
    except ImportError:
        logger.warning("tensorrt/pycuda not importable. Skipping Python validation.")
        return

    trt_logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(trt_logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    in_name = engine.get_tensor_name(0)
    out_name = engine.get_tensor_name(1)

    in_shape = (1, 3, input_size, input_size)
    out_shape = tuple(context.get_tensor_shape(out_name))

    h_in = cuda.pagelocked_empty(int(np.prod(in_shape)), dtype=np.float16)
    h_out = cuda.pagelocked_empty(int(np.prod(out_shape)), dtype=np.float16)
    d_in = cuda.mem_alloc(h_in.nbytes)
    d_out = cuda.mem_alloc(h_out.nbytes)
    stream = cuda.Stream()

    np.copyto(h_in, np.random.rand(*in_shape).astype(np.float16).ravel())

    # Warm up
    for _ in range(n_warmup):
        cuda.memcpy_htod_async(d_in, h_in, stream)
        context.set_tensor_address(in_name, int(d_in))
        context.set_tensor_address(out_name, int(d_out))
        context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_out, d_out, stream)
        stream.synchronize()

    # Benchmark
    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        cuda.memcpy_htod_async(d_in, h_in, stream)
        context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_out, d_out, stream)
        stream.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)

    lat = np.array(latencies)
    logger.info(
        "Validation (%d runs): mean=%.1f ms  p50=%.1f ms  p95=%.1f ms  FPS=%.0f",
        n_runs, lat.mean(), np.percentile(lat, 50), np.percentile(lat, 95), 1000 / lat.mean(),
    )

    if lat.mean() > 30.0:
        logger.warning(
            "Mean latency %.1f ms > 30 ms target. Consider input_size=320 or int8 quantization.",
            lat.mean(),
        )
    else:
        logger.info("PASS: Latency target met (%.1f ms < 30 ms).", lat.mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLOv8 to TensorRT FP16 engine")
    parser.add_argument("--model", default="yolov8n", help="Model name (yolov8n/s/m) or path to .pt")
    parser.add_argument("--output", default="/opt/rover/models", help="Output directory for engine files")
    parser.add_argument("--input-size", type=int, default=640, help="Inference input size (square)")
    parser.add_argument("--workspace", type=int, default=2048, help="TensorRT workspace in MB")
    parser.add_argument("--fp16", action="store_true", default=True, help="Enable FP16 (default: on)")
    parser.add_argument("--int8", action="store_true", default=False, help="Enable INT8 (needs calibration data)")
    parser.add_argument("--skip-onnx", action="store_true", help="Skip ONNX export (use existing .onnx)")
    parser.add_argument("--validate", action="store_true", default=True, help="Run Python validation after build")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_onnx:
        onnx_path = export_onnx(args.model, args.input_size, output_dir)
    else:
        stem = Path(args.model).stem
        onnx_path = output_dir / f"{stem}_{args.input_size}.onnx"
        if not onnx_path.exists():
            logger.error("ONNX file not found: %s", onnx_path)
            sys.exit(1)

    engine_path = build_tensorrt_engine(onnx_path, output_dir, args.workspace, args.fp16, args.int8)

    if args.validate:
        validate_engine(engine_path, args.input_size)

    logger.info("Done. Engine: %s", engine_path)


if __name__ == "__main__":
    main()
