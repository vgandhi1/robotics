"""
Export a trained SB3 PPO policy to ONNX format.

The exported model takes a single input tensor (normalized state vector) and
outputs an action tensor. The ONNX graph is verified with onnxruntime to
ensure numerical equivalence with the original PyTorch model.

Pipeline:
    SB3 .zip → PyTorch actor MLP → ONNX FP32 → verified → saved

Usage:
    python export/export_onnx.py --model logs/best_model.zip --output export/model.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _extract_actor(model_path: str) -> tuple[nn.Module, int]:
    """
    Load an SB3 PPO model and extract the deterministic actor sub-graph.

    SB3's ActorCriticPolicy contains both actor and critic networks.
    For edge deployment we only need the actor (policy) path.

    Returns:
        (actor_module, obs_dim)
    """
    from stable_baselines3 import PPO

    model = PPO.load(model_path, device="cpu")
    policy = model.policy.cpu()
    obs_dim = policy.observation_space.shape[0]

    # Build a thin wrapper that mirrors the deterministic inference path:
    # obs → mlp_extractor → action_net → tanh (continuous action squashing)
    class ActorWrapper(nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.mlp_extractor = policy.mlp_extractor
            self.action_net = policy.action_net
            # SB3 uses a diagonal Gaussian; for deterministic export we
            # take the mean and apply tanh squashing.
            self.log_std = policy.log_std  # not used in forward, just kept

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            features = self.mlp_extractor.forward_actor(obs)
            mean_action = self.action_net(features)
            # Tanh squashing to keep output in [-1, 1]
            return torch.tanh(mean_action)

    actor = ActorWrapper(policy)
    actor.eval()
    return actor, obs_dim


def export_to_onnx(
    model_path: str,
    output_path: str = "export/model.onnx",
    opset_version: int = 17,
    verify: bool = True,
) -> str:
    """
    Export a trained SB3 PPO actor to ONNX.

    Args:
        model_path:    Path to .zip SB3 model.
        output_path:   Destination .onnx file.
        opset_version: ONNX opset version (17 = latest stable).
        verify:        Run onnxruntime verification after export.

    Returns:
        Path to the exported ONNX file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from: {model_path}")
    actor, obs_dim = _extract_actor(model_path)

    dummy_input = torch.zeros(1, obs_dim, dtype=torch.float32)

    print(f"Exporting ONNX (opset {opset_version}) → {output_path}")
    # Use legacy tracing-based export (avoids onnxscript dependency in PyTorch 2.x)
    with torch.no_grad():
        torch.onnx.export(
            actor,
            dummy_input,
            str(output_path),
            opset_version=opset_version,
            input_names=["state"],
            output_names=["action"],
            dynamic_axes={"state": {0: "batch_size"}, "action": {0: "batch_size"}},
            export_params=True,
            do_constant_folding=True,
        )

    if verify:
        _verify_onnx(actor, str(output_path), obs_dim)

    print(f"ONNX export complete: {output_path}")
    return str(output_path)


def _verify_onnx(actor: nn.Module, onnx_path: str, obs_dim: int) -> None:
    """
    Verify the ONNX model against the original PyTorch module using
    random test vectors. Raises AssertionError if outputs diverge.
    """
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("  [WARN] onnx / onnxruntime not installed; skipping verification")
        return

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    rng = np.random.default_rng(0)
    test_inputs = rng.uniform(-1.0, 1.0, size=(20, obs_dim)).astype(np.float32)

    max_diff = 0.0
    for x in test_inputs:
        x_tensor = torch.from_numpy(x[None])
        with torch.no_grad():
            pt_out = actor(x_tensor).numpy()
        ort_out = sess.run(None, {input_name: x[None]})[0]
        diff = float(np.abs(pt_out - ort_out).max())
        max_diff = max(max_diff, diff)

    assert max_diff < 1e-5, (
        f"ONNX verification failed: max output diff = {max_diff:.2e} (threshold 1e-5)"
    )
    print(f"  ONNX verification passed (max diff = {max_diff:.2e})")


def print_model_info(onnx_path: str) -> None:
    """Print ONNX model metadata: inputs, outputs, parameter count."""
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError:
        return

    model = onnx.load(onnx_path)
    print(f"\nONNX Model Info: {onnx_path}")
    print(f"  Opset version  : {model.opset_import[0].version}")
    for inp in model.graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"  Input  '{inp.name}': shape={shape}")
    for out in model.graph.output:
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"  Output '{out.name}': shape={shape}")

    total_params = sum(
        np.prod(t.dims)
        for t in model.graph.initializer
        if t.dims
    )
    print(f"  Total parameters: {total_params:,}")
    size_bytes = Path(onnx_path).stat().st_size
    print(f"  File size       : {size_bytes / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(description="Export SB3 PPO policy to ONNX")
    parser.add_argument("--model", required=True, help="Path to SB3 .zip model")
    parser.add_argument(
        "--output", default="export/model.onnx", help="Output ONNX path"
    )
    parser.add_argument(
        "--opset", type=int, default=17, help="ONNX opset version"
    )
    parser.add_argument(
        "--no-verify", action="store_true", help="Skip onnxruntime verification"
    )
    args = parser.parse_args()

    onnx_path = export_to_onnx(
        model_path=args.model,
        output_path=args.output,
        opset_version=args.opset,
        verify=not args.no_verify,
    )
    print_model_info(onnx_path)


if __name__ == "__main__":
    main()
