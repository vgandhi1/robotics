"""
Unit tests for the ONNX export and quantization pipeline.

Tests:
  - ONNX export from a dummy PyTorch MLP (simulates the actor network)
  - ONNX model loads and produces outputs with correct shapes
  - INT8 quantization runs without error
  - FP32 and INT8 outputs are numerically close
  - C header generation produces valid C syntax
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ─── Helpers ────────────────────────────────────────────────────────────────────

def _make_dummy_actor(obs_dim: int = 4, hidden: int = 64, action_dim: int = 1) -> nn.Module:
    """Create a simple MLP that mimics the PPO actor for export testing."""
    class DummyActor(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh(),
                nn.Linear(hidden, action_dim), nn.Tanh(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    model = DummyActor()
    model.eval()
    return model


def _export_dummy_onnx(
    actor: nn.Module, obs_dim: int, path: str, static_batch: bool = False
) -> None:
    dummy = torch.zeros(1, obs_dim, dtype=torch.float32)
    # For quantization tests use static batch (no dynamic axes) to avoid shape
    # inference issues in onnxruntime's quantizer with dynamic-batch models.
    dynamic_axes = (
        None if static_batch
        else {"state": {0: "batch"}, "action": {0: "batch"}}
    )
    torch.onnx.export(
        actor, dummy, path,
        input_names=["state"],
        output_names=["action"],
        dynamic_axes=dynamic_axes,
        export_params=True,
    )


# ─── Tests ──────────────────────────────────────────────────────────────────────

class TestONNXExport:
    @pytest.fixture
    def onnx_file(self, tmp_path):
        actor = _make_dummy_actor()
        onnx_path = str(tmp_path / "model.onnx")
        _export_dummy_onnx(actor, obs_dim=4, path=onnx_path)
        return onnx_path, actor

    def test_onnx_file_created(self, onnx_file):
        onnx_path, _ = onnx_file
        assert Path(onnx_path).exists()
        assert Path(onnx_path).stat().st_size > 0

    def test_onnx_model_valid(self, onnx_file):
        onnx_path, _ = onnx_file
        onnx = pytest.importorskip("onnx")
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)

    def test_onnx_inference_shape(self, onnx_file):
        onnx_path, _ = onnx_file
        ort = pytest.importorskip("onnxruntime")
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name

        x = np.zeros((1, 4), dtype=np.float32)
        output = sess.run(None, {input_name: x})[0]
        assert output.shape == (1, 1)

    def test_onnx_matches_pytorch(self, onnx_file):
        """ONNX output should numerically match PyTorch forward pass."""
        onnx_path, actor = onnx_file
        ort = pytest.importorskip("onnxruntime")
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name

        rng = np.random.default_rng(0)
        for _ in range(20):
            x = rng.uniform(-1.0, 1.0, size=(1, 4)).astype(np.float32)
            pt_out = actor(torch.from_numpy(x)).detach().numpy()
            ort_out = sess.run(None, {input_name: x})[0]
            np.testing.assert_allclose(pt_out, ort_out, atol=1e-5)

    def test_onnx_batch_inference(self, onnx_file):
        onnx_path, _ = onnx_file
        ort = pytest.importorskip("onnxruntime")
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name

        x = np.random.randn(8, 4).astype(np.float32)
        output = sess.run(None, {input_name: x})[0]
        assert output.shape == (8, 1)


class TestQuantization:
    @pytest.fixture
    def int8_paths(self, tmp_path):
        from onnxruntime.quantization import quantize_dynamic, QuantType

        actor = _make_dummy_actor()
        fp32_path = str(tmp_path / "model.onnx")
        int8_path = str(tmp_path / "model_int8.onnx")
        # Use static batch=1 for quantization (matches ESP32 single-sample inference)
        _export_dummy_onnx(actor, obs_dim=4, path=fp32_path, static_batch=True)

        # Preprocess model to fix shape annotations before quantization
        try:
            from onnxruntime.quantization.preprocess import quant_pre_process
            preprocessed = str(Path(fp32_path).parent / "model_pre.onnx")
            quant_pre_process(fp32_path, preprocessed, skip_optimization=False)
            quant_input = preprocessed
        except Exception:
            quant_input = fp32_path

        quantize_dynamic(
            model_input=quant_input,
            model_output=int8_path,
            weight_type=QuantType.QInt8,
        )
        return fp32_path, int8_path

    def test_int8_file_created(self, int8_paths):
        pytest.importorskip("onnxruntime.quantization")
        _, int8_path = int8_paths
        assert Path(int8_path).exists()
        assert Path(int8_path).stat().st_size > 0

    def test_int8_model_is_valid_onnx(self, int8_paths):
        """INT8 model should be a valid, loadable ONNX file."""
        pytest.importorskip("onnxruntime.quantization")
        onnx = pytest.importorskip("onnx")
        _, int8_path = int8_paths
        model = onnx.load(int8_path)
        # Basic check: INT8 model has quantization nodes (MatMulInteger or QLinearMatMul)
        node_op_types = {n.op_type for n in model.graph.node}
        has_quant_ops = bool(
            node_op_types & {"MatMulInteger", "QLinearMatMul", "DynamicQuantizeLinear"}
        )
        assert has_quant_ops, (
            f"INT8 model should contain quantization ops. Got: {node_op_types}"
        )

    def test_int8_close_to_fp32(self, int8_paths):
        """INT8 outputs should be within 0.05 of FP32 outputs."""
        ort = pytest.importorskip("onnxruntime")
        pytest.importorskip("onnxruntime.quantization")
        fp32_path, int8_path = int8_paths

        sess_fp32 = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
        sess_int8 = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
        name = sess_fp32.get_inputs()[0].name

        rng = np.random.default_rng(0)
        for _ in range(50):
            x = rng.uniform(-1.0, 1.0, size=(1, 4)).astype(np.float32)
            fp32_out = sess_fp32.run(None, {name: x})[0]
            int8_out = sess_int8.run(None, {name: x})[0]
            diff = float(np.abs(fp32_out - int8_out).max())
            assert diff < 0.05, (
                f"INT8 output differs from FP32 by {diff:.4f} (threshold 0.05)"
            )


class TestCHeaderGeneration:
    def test_header_generated(self, tmp_path):
        from export.quantize import generate_c_header

        # Create a fake tflite file
        fake_tflite = tmp_path / "model.tflite"
        fake_tflite.write_bytes(bytes(range(256)))

        header_path = str(tmp_path / "rl_policy_data.h")
        result = generate_c_header(str(fake_tflite), header_path)

        assert Path(result).exists()
        content = Path(result).read_text()

        assert "#pragma once" in content
        assert "unsigned char" in content
        assert "unsigned int" in content
        assert "256" in content  # length value

    def test_header_valid_c_syntax_hint(self, tmp_path):
        from export.quantize import generate_c_header

        fake_tflite = tmp_path / "model.tflite"
        fake_tflite.write_bytes(b"\x1c\x00\x00\x00" * 8)
        header_path = str(tmp_path / "test.h")
        generate_c_header(str(fake_tflite), header_path)
        content = Path(header_path).read_text()

        # Should contain proper C hex values
        assert "0x1c" in content
        assert "{" in content
        assert "};" in content
