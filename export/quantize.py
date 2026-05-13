"""
Model quantization pipeline: ONNX FP32 → INT8 ONNX → TFLite FlatBuffer.

The quantization reduces the model from ~150 KB to ~32 KB, making it
suitable for deployment on the ESP32's 4 MB flash while staying within the
~320 KB SRAM limit for inference buffers.

Quantization approach:
  - ONNX INT8: Dynamic quantization (no calibration data needed; activations
    are quantized at runtime). This is appropriate for MLP inference loops.
  - TFLite INT8: Full integer quantization using representative dataset for
    activation range calibration (produces deterministic INT8 model).

Usage:
    python export/quantize.py --onnx export/model.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def quantize_onnx_int8(
    onnx_fp32_path: str,
    output_path: Optional[str] = None,
    verify: bool = True,
) -> str:
    """
    Apply dynamic INT8 quantization to an ONNX FP32 model.

    Dynamic quantization: weights are statically quantized to INT8 at export
    time; activations are dynamically quantized at inference time. No
    calibration data is required.

    Args:
        onnx_fp32_path: Path to the FP32 ONNX model.
        output_path:    Path for the INT8 ONNX output (default: <name>_int8.onnx).
        verify:         Verify output equivalence after quantization.

    Returns:
        Path to the INT8 ONNX model.
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError as e:
        raise ImportError(
            "onnxruntime-tools required: pip install onnxruntime-tools"
        ) from e

    fp32_path = Path(onnx_fp32_path)
    if output_path is None:
        output_path = str(fp32_path.parent / (fp32_path.stem + "_int8.onnx"))

    print(f"Quantizing {fp32_path.name} → {Path(output_path).name} (INT8 dynamic)")

    # Run onnxruntime preprocessing to fix shape annotations and opset compatibility
    # before passing to the quantizer (required for models exported by PyTorch 2.5+).
    try:
        from onnxruntime.quantization.preprocess import quant_pre_process
        preprocessed_path = str(fp32_path.parent / (fp32_path.stem + "_preprocessed.onnx"))
        quant_pre_process(str(fp32_path), preprocessed_path, skip_optimization=False)
        quantizer_input = preprocessed_path
    except Exception:
        # Fall back to direct quantization if preprocessing fails
        quantizer_input = str(fp32_path)

    quantize_dynamic(
        model_input=quantizer_input,
        model_output=output_path,
        weight_type=QuantType.QInt8,
    )

    fp32_size = fp32_path.stat().st_size
    int8_size = Path(output_path).stat().st_size
    print(f"  FP32 size : {fp32_size / 1024:.1f} KB")
    print(f"  INT8 size : {int8_size / 1024:.1f} KB  "
          f"({100 * int8_size / fp32_size:.0f}% of original)")

    if verify:
        _verify_equivalence(str(fp32_path), output_path)

    return output_path


def _verify_equivalence(
    fp32_path: str, int8_path: str, n_tests: int = 50, tol: float = 0.05
) -> None:
    """
    Compare FP32 and INT8 model outputs on random inputs.

    Tolerance is set to 0.05 (5% of the action range [-1, 1]) which is
    acceptable for control given that the policy is robust to perturbation.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("  [WARN] onnxruntime not available; skipping equivalence check")
        return

    sess_fp32 = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
    input_name = sess_fp32.get_inputs()[0].name
    obs_dim = sess_fp32.get_inputs()[0].shape[1]

    rng = np.random.default_rng(42)
    test_inputs = rng.uniform(-1.0, 1.0, size=(n_tests, obs_dim)).astype(np.float32)

    diffs = []
    for x in test_inputs:
        x_batch = x[None]
        fp32_out = sess_fp32.run(None, {input_name: x_batch})[0]
        int8_out = sess_int8.run(None, {input_name: x_batch})[0]
        diffs.append(float(np.abs(fp32_out - int8_out).max()))

    mean_diff = float(np.mean(diffs))
    max_diff = float(np.max(diffs))
    print(f"  Quantization error — mean={mean_diff:.4f}, max={max_diff:.4f}")
    if max_diff > tol:
        print(f"  [WARN] Max diff {max_diff:.4f} exceeds tolerance {tol}. "
              f"Consider re-training with quantization-aware training (QAT).")
    else:
        print(f"  INT8 equivalence verified (max diff < {tol})")


def convert_to_tflite(
    onnx_path: str,
    output_path: Optional[str] = None,
    obs_dim: int = 4,
    n_calibration_samples: int = 200,
) -> str:
    """
    Convert an ONNX model to TensorFlow Lite (FlatBuffer) with full INT8
    quantization using a representative calibration dataset.

    The calibration dataset is generated synthetically from the observation
    space distribution (uniform [-1, 1] for each normalized state component).

    Args:
        onnx_path:               Path to the ONNX model (FP32 or INT8 ONNX).
        output_path:             Path for the .tflite output.
        obs_dim:                 Observation dimension (default: 4).
        n_calibration_samples:   Samples for activation range calibration.

    Returns:
        Path to the .tflite model.
    """
    try:
        import tensorflow as tf
        import tf2onnx  # noqa: F401
    except ImportError:
        print(
            "[WARN] TensorFlow not available. Skipping TFLite conversion.\n"
            "  Install with: pip install tensorflow tf2onnx"
        )
        return ""

    onnx_path_obj = Path(onnx_path)
    if output_path is None:
        output_path = str(onnx_path_obj.parent / (onnx_path_obj.stem.replace("_int8", "") + ".tflite"))

    print(f"\nConverting to TFLite: {onnx_path_obj.name} → {Path(output_path).name}")

    # Load ONNX model and convert to TF SavedModel via onnx-tf or tf2onnx
    saved_model_dir = str(onnx_path_obj.parent / "_tmp_saved_model")
    _onnx_to_saved_model(str(onnx_path), saved_model_dir)

    # TFLite conversion with INT8 quantization
    rng = np.random.default_rng(0)
    calibration_data = rng.uniform(-1.0, 1.0, size=(n_calibration_samples, obs_dim)).astype(np.float32)

    def representative_dataset():
        for sample in calibration_data:
            yield [sample[None]]

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    Path(output_path).write_bytes(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"  TFLite model size: {size_kb:.1f} KB")
    print(f"  Saved to: {output_path}")
    return output_path


def _onnx_to_saved_model(onnx_path: str, saved_model_dir: str) -> None:
    """Convert ONNX → TF SavedModel using onnx-tensorflow."""
    try:
        import onnx
        import onnx_tf
        model = onnx.load(onnx_path)
        tf_rep = onnx_tf.backend.prepare(model)
        tf_rep.export_graph(saved_model_dir)
    except ImportError:
        raise ImportError(
            "onnx-tf required for TFLite conversion: pip install onnx-tf"
        )


def generate_c_header(tflite_path: str, output_path: Optional[str] = None) -> str:
    """
    Convert a TFLite model file to a C header (uint8_t array) that can be
    directly included in the ESP32 Arduino sketch.

    This replicates `xxd -i model.tflite > rl_policy_data.h`.

    Args:
        tflite_path:  Path to .tflite model file.
        output_path:  Path for the .h header file.

    Returns:
        Path to the generated header file.
    """
    tflite_path_obj = Path(tflite_path)
    if output_path is None:
        output_path = str(
            Path(__file__).resolve().parents[1]
            / "firmware/inference_loop/rl_policy_data.h"
        )

    data = tflite_path_obj.read_bytes()
    var_name = tflite_path_obj.stem.replace("-", "_").replace(".", "_")

    hex_lines = []
    for i in range(0, len(data), 12):
        chunk = data[i : i + 12]
        hex_lines.append("  " + ", ".join(f"0x{b:02x}" for b in chunk))

    header_content = (
        f"// Auto-generated from {tflite_path_obj.name}\n"
        f"// DO NOT EDIT MANUALLY\n\n"
        f"#pragma once\n\n"
        f"const unsigned char {var_name}[] = {{\n"
        + ",\n".join(hex_lines)
        + f"\n}};\n\n"
        f"const unsigned int {var_name}_len = {len(data)};\n"
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(header_content)
    print(f"\nC header written: {output_path}  ({len(data) / 1024:.1f} KB)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Quantize and convert RL policy model")
    parser.add_argument("--onnx", required=True, help="Path to FP32 ONNX model")
    parser.add_argument(
        "--output-int8", default=None, help="Output path for INT8 ONNX"
    )
    parser.add_argument(
        "--output-tflite", default=None, help="Output path for TFLite model"
    )
    parser.add_argument(
        "--obs-dim", type=int, default=4, help="Observation vector dimension"
    )
    parser.add_argument(
        "--generate-header", action="store_true",
        help="Generate C header file for ESP32 firmware"
    )
    parser.add_argument(
        "--no-verify", action="store_true", help="Skip equivalence verification"
    )
    args = parser.parse_args()

    # Step 1: INT8 ONNX
    int8_path = quantize_onnx_int8(
        args.onnx,
        output_path=args.output_int8,
        verify=not args.no_verify,
    )

    # Step 2: TFLite (optional, requires TF)
    tflite_path = convert_to_tflite(
        args.onnx,
        output_path=args.output_tflite,
        obs_dim=args.obs_dim,
    )

    # Step 3: C header
    if args.generate_header and tflite_path:
        generate_c_header(tflite_path)


if __name__ == "__main__":
    main()
