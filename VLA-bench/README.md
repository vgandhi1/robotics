# vla-bench

[![W&B Report](https://img.shields.io/badge/W%26B-Report-FFBE00?style=flat-square&logo=weightsandbiases)](https://wandb.ai/vgandhi1/vla-bench)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![FSDP](https://img.shields.io/badge/multi--GPU-FSDP-76B900?style=flat-square)](https://pytorch.org/docs/stable/fsdp.html)
[![FlashAttention](https://img.shields.io/badge/attention-FlashAttn--2-blue?style=flat-square)](https://github.com/Dao-AILab/flash-attention)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

> Systematic profiling and optimization of the real-hardware fine-tuning stage in a sim-to-real VLA pipeline.
> Three optimization layers applied in sequence — WebDataset, FlashAttention-2, FSDP — with measured gains at each step.
> Full W&B comparison report linked above.

## Results Summary

| Metric | Naive Baseline | Phase 3 Optimized | Gain |
|---|---|---|---|
| Images/sec (2 GPU) | ~14 | ~40+ | **3.3×** |
| Peak VRAM total | ~38 GB | ~28 GB | **-26%** |
| GPU utilization | ~40% | ~87% | **+47pp** |
| Multi-GPU scaling efficiency | ~65% | ~88% | **+23pp** |
| Cost per training run (500K frames) | ~$24.75 | ~$8.75 | **-65%** |

## What Each Optimization Contributes

| Layer | Optimization | Primary Gain |
|---|---|---|
| Phase 2 | WebDataset streaming | GPU utilization 40% → 87% (CPU bottleneck removed) |
| Phase 3a | FlashAttention-2 | Peak VRAM -26% (fused attention, no N² materialization) |
| Phase 3b | FSDP full shard | Enables 2× batch size; 88% multi-GPU scaling efficiency |
| Phase 3c | Activation checkpointing | Additional -15% VRAM; trades 20% compute overhead |

## Reproduce

```bash
git clone https://github.com/vgandhi1/vla-bench.git && cd vla-bench
pip install -r requirements.txt
pip install flash-attn --no-build-isolation

# Generate synthetic dataset
python data/synthetic_episodes.py
python data/prepare_dataset.py

# Phase 1 — baseline (1 GPU)
bash scripts/run_baseline.sh

# Phase 3 — optimized (2 GPU)
bash scripts/run_optimized.sh
```

## Profiler Traces

| Before (Phase 1) | After (Phase 2) |
|---|---|
| ![baseline](docs/profiler_traces/01_baseline_trace.png) | ![optimized](docs/profiler_traces/02_webdataset_trace.png) |

White gaps = GPU idle waiting for CPU image decode.
Dense kernels = GPU fed continuously by WebDataset async workers.

## Cost Analysis

| Configuration | Throughput | Time for 500K frames | RunPod cost ($2.50/hr, 2×RTX 3090) |
|---|---|---|---|
| Naive baseline | ~14 imgs/s | ~9.9 hrs | **$24.75** |
| Phase 3 optimized | ~40 imgs/s | ~3.5 hrs | **$8.75** |
| **Savings per run** | | **6.4 hrs** | **$16.00** |

> In a sim-to-real pipeline, this time reduction compresses the real-floor failure → policy patch → redeployment cycle from ~2 days to ~6 hours, enabling same-day iteration on new failure modes.

### Research Sprint ROI

| Scenario | Runs/Sprint | Cost/Sprint (Naive) | Cost/Sprint (Optimized) | Savings/Sprint |
|---|---|---|---|---|
| Hyperparameter sweep | 10 runs | $247.50 | $87.50 | **$160.00** |
| Architecture ablation | 20 runs | $495.00 | $175.00 | **$320.00** |
| Monthly research budget | ~50 runs | $1,237.50 | $437.50 | **$800.00** |

## Stack

PyTorch 2.2 · HuggingFace Transformers · FlashAttention-2 · FSDP ·
WebDataset · Weights & Biases · TensorBoard · RunPod (2× RTX 3090)

## Related

Part of a broader factory AI portfolio. See also:
- [edge-telemetry-plane (DETCP)](https://github.com/vgandhi1/edge-telemetry-plane) — fault-tolerant edge infrastructure
- [apex-recovery](https://github.com/vgandhi1/apex-recovery) — operator cockpit for VLA recovery data collection
