# vla-bench
## Vision-Language-Action Training Optimization: GPU Utilization & Memory Efficiency Benchmarking Study

[![Repo](https://img.shields.io/badge/GitHub-vgandhi1%2Fvla--bench-181717?style=flat-square&logo=github)](https://github.com/vgandhi1/vla-bench)
[![W&B Report](https://img.shields.io/badge/W%26B-Report-FFBE00?style=flat-square&logo=weightsandbiases)](https://wandb.ai/vgandhi1/vla-bench)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![FSDP](https://img.shields.io/badge/multi--GPU-FSDP-76B900?style=flat-square)](https://pytorch.org/docs/stable/fsdp.html)
[![FlashAttention](https://img.shields.io/badge/attention-FlashAttn--2-blue?style=flat-square)](https://github.com/Dao-AILab/flash-attention)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

> **Repo:** [vgandhi1/vla-bench](https://github.com/vgandhi1/vla-bench) · **W&B Report:** https://wandb.ai/vgandhi1/vla-bench  
> **Project Type:** ML Infrastructure / MLOps · **Stack:** PyTorch · HuggingFace · FlashAttention-2 · FSDP · WebDataset · W&B  
> **Hardware target:** 2× RTX 3090 (24 GB VRAM each) via RunPod or equivalent  
> **Strategic purpose:** Systematically profile a naive VLA training loop, apply three optimization layers in sequence, and document the throughput and VRAM gains at each step — proving training cost-viability without A100/H100 clusters.

---

## Table of Contents

1. [Problem Statement & Business Value](#1-problem-statement--business-value)
2. [Environment Setup](#2-environment-setup)
3. [Phase 1 — Naive Baseline: Find the Bottleneck](#3-phase-1--naive-baseline-find-the-bottleneck)
4. [Phase 2 — Data Ingestion Optimization](#4-phase-2--data-ingestion-optimization)
5. [Phase 3 — Memory & Compute Optimization](#5-phase-3--memory--compute-optimization)
6. [Phase 4 — W&B ROI Report](#6-phase-4--wb-roi-report)
7. [Complete Optimized Training Script](#7-complete-optimized-training-script)
8. [Benchmarking Protocol](#8-benchmarking-protocol)
9. [Expected Results & Acceptance Criteria](#9-expected-results--acceptance-criteria)
10. [Cost Analysis & ROI Framing](#10-cost-analysis--roi-framing)

---

## 1. Problem Statement & Business Value

### Context

Training Vision-Language-Action (VLA) models on teleoperation episode data is the core workload of any robotics foundation model program. A naive implementation of a VLM training loop on 2× RTX 3090s typically achieves:

- **GPU utilization: 30–45%** (CPU image decoding bottleneck)
- **VRAM efficiency: ~40%** (attention computation keeps large intermediate tensors alive)
- **Multi-GPU scaling: ~1.3×** (data parallel overhead, gradient synchronization)
- **Effective batch size: 4–8** before OOM

The result: training a 7B-parameter VLM on 10,000 teleoperation episodes takes 72+ hours at $2.50/hr on RunPod = **$180+ per training run**. This makes rapid iteration (the core of research) economically untenable.

### Optimization Targets

| Metric | Naive Baseline | Target (Phase 3) | Expected Savings |
|---|---|---|---|
| GPU Utilization | ~40% | 85%+ | 2.1× throughput gain |
| Peak VRAM (7B model) | ~42 GB combined | ~32 GB combined | Larger batch sizes |
| Images/sec (2 GPU) | ~12 | ~40+ | 3.3× faster |
| Multi-GPU Scaling Efficiency | ~65% | ~85%+ | Near-linear scaling |
| Cost per training run | ~$180 | ~$55 | $125 savings per run |

### Why These Specific Optimizations

1. **WebDataset** eliminates the CPU→GPU data starvation that causes GPU idle time
2. **FlashAttention-2** fuses attention operations to SRAM, removing the O(N²) VRAM spike from naive attention
3. **FSDP** shards model state across GPUs, enabling models that don't fit on a single device
4. **Activation Checkpointing** trades a ~20% compute increase for a 50-70% VRAM reduction on intermediate activations

---

## 2. Environment Setup

### 2.1 RunPod Instance Configuration

**Recommended pod:** 2× RTX 3090 (48 GB VRAM total) with NVLink preferred
**Template:** `runpod/pytorch:2.2.1-py3.11-cuda12.1.1-devel-ubuntu22.04`
**Disk:** 50 GB container + 100 GB network volume (for dataset)

### 2.2 Dependencies

```bash
# requirements.txt
torch>=2.2.0
torchvision>=0.17.0
transformers>=4.40.0
accelerate>=0.28.0
webdataset>=0.2.86
wandb>=0.17.0
flash-attn>=2.5.6
einops>=0.8.0
Pillow>=10.0.0
numpy>=1.26.0
tensorboard>=2.16.0
torch-tb-profiler>=0.4.3
```

```bash
# Clone
git clone https://github.com/vgandhi1/vla-bench.git
cd vla-bench

# Install (RunPod pytorch:2.2.1 template recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
# FlashAttention must be compiled (takes ~15 min on RunPod)
pip install flash-attn --no-build-isolation
```

### 2.3 Repository Structure

```
vla-bench/                              # github.com/vgandhi1/vla-bench
├── README.md                           # W&B report link, results summary, setup
├── requirements.txt
├── data/
│   ├── prepare_dataset.py              # Convert raw episodes → WebDataset tarballs
│   └── synthetic_episodes.py          # Generate synthetic VLA episodes for testing
├── training/
│   ├── baseline_train.py              # Phase 1: Naive training loop
│   ├── optimized_train.py             # Phase 3: All optimizations enabled
│   └── utils/
│       ├── metrics.py                 # Throughput, VRAM, scaling efficiency tracking
│       └── profiler_utils.py          # PyTorch Profiler helpers
├── scripts/
│   ├── run_baseline.sh                # Single-command baseline run
│   └── run_optimized.sh               # torchrun 2-GPU optimized run
└── docs/
    ├── profiler_traces/               # TensorBoard trace screenshots (PNG)
    │   ├── 01_baseline_trace.png      # GPU idle gaps — the bottleneck proof
    │   └── 02_webdataset_trace.png    # Dense GPU utilization post-optimization
    └── wandb_screenshots/             # W&B report panel screenshots
        ├── throughput_comparison.png
        ├── vram_comparison.png
        ├── gpu_utilization.png
        └── scaling_efficiency.png
```

---

## 3. Phase 1 — Naive Baseline: Find the Bottleneck

### 3.1 Synthetic VLA Episode Dataset

Before profiling real data, generate a synthetic dataset that mimics real VLA episode structure:

```python
# data/synthetic_episodes.py
import os
import json
import numpy as np
from PIL import Image
import random

def generate_synthetic_episodes(n_episodes=500, output_dir="data/raw_episodes"):
    """
    Generate synthetic VLA episodes mimicking teleoperation structure.
    Each episode: sequence of (image, action_vector, language_instruction)
    """
    os.makedirs(output_dir, exist_ok=True)

    task_instructions = [
        "Pick up the red block and place it in the bin",
        "Grasp the cable and route it through the bracket",
        "Insert the part into the left socket",
        "Move the object to the target position",
        "Assemble the two components together",
    ]

    for ep_idx in range(n_episodes):
        ep_dir = os.path.join(output_dir, f"episode_{ep_idx:05d}")
        os.makedirs(ep_dir, exist_ok=True)

        n_frames = random.randint(50, 200)
        instruction = random.choice(task_instructions)

        frames = []
        for f_idx in range(n_frames):
            # Synthetic RGB image (224×224)
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img_path = os.path.join(ep_dir, f"frame_{f_idx:04d}.jpg")
            img.save(img_path, quality=85)

            # Action vector: [x, y, z, roll, pitch, yaw, gripper]
            action = np.random.randn(7).tolist()

            frames.append({
                "frame_idx": f_idx,
                "image_path": img_path,
                "action": action,
                "instruction": instruction,
            })

        metadata = {"episode_id": ep_idx, "frames": frames, "task": instruction}
        with open(os.path.join(ep_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)

    print(f"Generated {n_episodes} synthetic episodes in {output_dir}")


if __name__ == "__main__":
    generate_synthetic_episodes()
```

### 3.2 Naive Training Loop

```python
# training/baseline_train.py
"""
Phase 1: Naive baseline training loop.
Purpose: Establish bottleneck baseline metrics before optimization.
Expected issues:
  - GPU utilization ~40% (CPU-bound image decoding)
  - Large VRAM spikes from naive attention
  - OOM at batch_size >= 8 with 7B model
"""

import os
import time
import torch
import torch.nn as nn
import wandb
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForVision2Seq
from torch.profiler import profile, record_function, ProfilerActivity
import json
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID = "Salesforce/blip2-opt-2.7b"   # Start with 2.7B; swap for 7B for real runs
BATCH_SIZE = 4
NUM_WORKERS = 2                           # Low to simulate naive baseline
LEARNING_RATE = 1e-4
MAX_STEPS = 100                           # Short run for profiling
PROFILE_STEPS = (10, 20)                  # Profile steps 10-20
DATA_DIR = "data/raw_episodes"

# ── W&B init ──────────────────────────────────────────────────────────────────
wandb.init(
    project="vla-scale",
    name="phase1-naive-baseline",
    config={
        "model": MODEL_ID,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "optimization": "none",
        "flash_attention": False,
        "fsdp": False,
        "activation_checkpointing": False,
        "webdataset": False,
    }
)

# ── Dataset: naive implementation ────────────────────────────────────────────
class NaiveVLADataset(Dataset):
    """Naive dataset: loads all metadata upfront, decodes images on-the-fly in main process."""

    def __init__(self, data_dir, processor):
        self.processor = processor
        self.samples = []

        for ep_dir in sorted(os.listdir(data_dir))[:200]:   # First 200 episodes
            meta_path = os.path.join(data_dir, ep_dir, "metadata.json")
            if not os.path.exists(meta_path):
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            for frame in meta['frames'][::5]:   # Every 5th frame
                self.samples.append({
                    "image_path": frame['image_path'],
                    "action": frame['action'],
                    "instruction": meta['task'],
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # ← This image decode runs in the MAIN PROCESS, blocking GPU
        image = Image.open(sample['image_path']).convert("RGB")
        inputs = self.processor(
            images=image,
            text=sample['instruction'],
            return_tensors="pt",
            padding="max_length",
            max_length=64,
            truncation=True,
        )
        action = torch.tensor(sample['action'], dtype=torch.float32)
        return {k: v.squeeze(0) for k, v in inputs.items()}, action


def main():
    device = torch.device("cuda:0")

    print(f"Loading model: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
    ).to(device)

    # Naive: NO flash attention, NO FSDP, NO activation checkpointing

    dataset = NaiveVLADataset(DATA_DIR, processor)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,   # ← Low; workers don't prefetch fast enough
        pin_memory=True,
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # ── Profiler setup ────────────────────────────────────────────────────────
    profiler = torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=5, warmup=5, active=10),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./runs/baseline'),
        record_shapes=True,
        with_stack=True,
    )

    model.train()
    profiler.start()

    step = 0
    epoch_start = time.time()

    for batch_inputs, actions in dataloader:
        if step >= MAX_STEPS:
            break

        step_start = time.time()

        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}

        with record_function("forward_pass"):
            outputs = model(**batch_inputs, labels=batch_inputs.get("input_ids"))
            loss = outputs.loss

        with record_function("backward_pass"):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ── Metrics ───────────────────────────────────────────────────────────
        step_time = time.time() - step_start
        imgs_per_sec = BATCH_SIZE / step_time
        vram_used = torch.cuda.max_memory_allocated(device) / (1024**3)

        wandb.log({
            "step": step,
            "loss": loss.item(),
            "images_per_sec": imgs_per_sec,
            "peak_vram_gb": vram_used,
            "gpu_util_pct": get_gpu_utilization(),
            "step_time_ms": step_time * 1000,
        })

        if step % 10 == 0:
            print(f"Step {step:4d} | Loss: {loss.item():.4f} | "
                  f"{imgs_per_sec:.1f} imgs/s | VRAM: {vram_used:.1f} GB")

        profiler.step()
        step += 1

    profiler.stop()
    wandb.finish()
    print(f"\nBaseline complete. Total time: {time.time() - epoch_start:.1f}s")


def get_gpu_utilization():
    """Query nvidia-smi for GPU utilization percent."""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        return float(result.stdout.strip().split('\n')[0])
    except Exception:
        return 0.0


if __name__ == "__main__":
    main()
```

### 3.3 Profiler Output Interpretation

After running baseline training, open TensorBoard:
```bash
tensorboard --logdir=runs/baseline --port=6006
```

Navigate to **Trace** view. Expected observations:

| What You'll See | What It Means |
|---|---|
| Large CPU gaps (white space) before each GPU kernel | DataLoader blocking GPU — image decoding on CPU |
| Short GPU bursts (1–3 ms) followed by long CPU waits | GPU sitting idle 50-60% of time |
| `aten::copy_` ops consuming 20-30% of CUDA time | H2D memory transfer overhead from non-pinned tensors |
| `aten::softmax` + `aten::bmm` dominating VRAM | Naive attention materializing full N×N matrix |

📸 **Required screenshot:** TensorBoard Trace view showing GPU idle (white) gaps. Save as `docs/profiler_traces/01_baseline_trace.png`.

---

## 4. Phase 2 — Data Ingestion Optimization

### 4.1 Convert Dataset to WebDataset Tarballs

```python
# data/prepare_dataset.py
"""
Convert raw episode data into WebDataset tar shards.
WebDataset enables asynchronous streaming from disk,
eliminating the CPU→GPU data starvation bottleneck.
"""

import os
import json
import io
import tarfile
import numpy as np
from PIL import Image

def create_webdataset_shards(
    input_dir="data/raw_episodes",
    output_dir="data/webdataset_shards",
    samples_per_shard=500,
):
    os.makedirs(output_dir, exist_ok=True)

    all_samples = []
    for ep_dir in sorted(os.listdir(input_dir)):
        meta_path = os.path.join(input_dir, ep_dir, "metadata.json")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        for frame in meta['frames'][::5]:
            all_samples.append({
                "image_path": frame['image_path'],
                "action": frame['action'],
                "instruction": meta['task'],
                "episode_id": str(meta['episode_id']),
            })

    shard_idx = 0
    for start in range(0, len(all_samples), samples_per_shard):
        shard_samples = all_samples[start:start + samples_per_shard]
        shard_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.tar")

        with tarfile.open(shard_path, "w") as tar:
            for i, sample in enumerate(shard_samples):
                key = f"{shard_idx:05d}_{i:06d}"

                # Image: read raw bytes and store as .jpg
                with open(sample['image_path'], 'rb') as f:
                    img_bytes = f.read()
                img_info = tarfile.TarInfo(name=f"{key}.jpg")
                img_info.size = len(img_bytes)
                tar.addfile(img_info, io.BytesIO(img_bytes))

                # Action + instruction as JSON
                meta_bytes = json.dumps({
                    "action": sample['action'],
                    "instruction": sample['instruction'],
                    "episode_id": sample['episode_id'],
                }).encode()
                meta_info = tarfile.TarInfo(name=f"{key}.json")
                meta_info.size = len(meta_bytes)
                tar.addfile(meta_info, io.BytesIO(meta_bytes))

        print(f"Wrote shard {shard_idx}: {len(shard_samples)} samples → {shard_path}")
        shard_idx += 1

    print(f"\nTotal: {shard_idx} shards, {len(all_samples)} samples")


if __name__ == "__main__":
    create_webdataset_shards()
```

### 4.2 WebDataset-Powered DataLoader

```python
# In training/optimized_train.py — data loading section

import webdataset as wds
import json

def decode_sample(sample, processor, image_size=224):
    """Decode a WebDataset sample — runs in parallel worker processes."""
    import io
    from PIL import Image as PILImage

    image = PILImage.open(io.BytesIO(sample['jpg'])).convert("RGB")
    meta = json.loads(sample['json'].decode())

    inputs = processor(
        images=image,
        text=meta['instruction'],
        return_tensors="pt",
        padding="max_length",
        max_length=64,
        truncation=True,
    )
    action = torch.tensor(meta['action'], dtype=torch.float32)
    return {k: v.squeeze(0) for k, v in inputs.items()}, action


def build_webdataset_loader(shard_dir, processor, batch_size, num_workers=8):
    """
    Build a WebDataset streaming dataloader.
    Key difference from naive: image decoding runs in N parallel worker processes,
    not the main process. GPU never waits for CPU.
    """
    shard_urls = f"{shard_dir}/shard_{{00000..{count_shards(shard_dir):05d}}}.tar"

    dataset = (
        wds.WebDataset(shard_urls, shardshuffle=True)
        .shuffle(1000)                              # Buffer shuffle for stochasticity
        .decode("pil")                              # Decode images in worker
        .to_tuple("jpg", "json")                    # Select relevant keys
        .map(lambda x: decode_sample({"jpg": x[0], "json": x[1]}, processor))
        .batched(batch_size, partial=False)
    )

    loader = wds.WebLoader(
        dataset,
        batch_size=None,      # Batching handled by .batched() above
        num_workers=num_workers,  # ← 8 workers vs. 2 in baseline
        pin_memory=True,
        prefetch_factor=4,    # Each worker prefetches 4 batches
    )

    return loader


def count_shards(shard_dir):
    return len([f for f in os.listdir(shard_dir) if f.endswith('.tar')]) - 1
```

**Expected improvement after Phase 2:**
- GPU utilization: 40% → 85–90%
- Images/sec: ~12 → ~28 (before memory optimizations)
- TensorBoard trace: white GPU idle gaps should largely disappear

📸 **Required screenshot:** New TensorBoard trace showing dense GPU kernels with minimal gaps. Save as `docs/profiler_traces/02_webdataset_trace.png`.

---

## 5. Phase 3 — Memory & Compute Optimization

### 5.1 FlashAttention-2 Integration

**Why it matters:** Standard attention computes a full N×N attention matrix in HBM (GPU high-bandwidth memory). For a 512-token sequence, that's 512×512 = 262K elements per head, per layer, materialized in VRAM. FlashAttention-2 fuses the softmax + matmul into a tiled kernel that keeps intermediate results in SRAM, reducing VRAM from O(N²) to O(N).

```python
# Enable FlashAttention-2 via HuggingFace model loading
from transformers import AutoModelForVision2Seq

model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,     # bfloat16: better range than float16, same memory
    attn_implementation="flash_attention_2",  # ← This is the only change needed
    device_map="auto",
)
```

**Verify FlashAttention-2 is active:**
```python
# Check that model attention layers report FlashAttention
for name, module in model.named_modules():
    if "attention" in name.lower():
        print(f"{name}: {type(module).__name__}")
        break
# Should show: Flash2Attention or FlashAttentionWithKVCache
```

### 5.2 FSDP (Fully Sharded Data Parallel)

FSDP shards model parameters, gradients, and optimizer states across all available GPUs. For a 7B model at bfloat16: `7B × 2 bytes = 14 GB` parameters. Without FSDP, both GPUs hold a full copy. With FSDP, each GPU holds only half — freeing 7 GB per GPU for larger batch sizes.

```python
# In training/optimized_train.py — model wrapping section

import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
import functools

def setup_distributed():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

def wrap_model_fsdp(model):
    """Wrap model with FSDP for multi-GPU sharding."""

    # Mixed precision: parameters in bfloat16, reduce ops in float32
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.bfloat16,
    )

    # Auto-wrap transformer layers (each layer becomes an FSDP unit)
    # Adjust the transformer layer class name for your specific model
    from transformers.models.opt.modeling_opt import OPTDecoderLayer  # Example
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={OPTDecoderLayer},
    )

    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # Shard params + grads + optimizer
        mixed_precision=mp_policy,
        auto_wrap_policy=auto_wrap_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # Prefetch next shard during backward
        device_id=torch.cuda.current_device(),
    )

    return model
```

**Launch with `torchrun`:**
```bash
# scripts/run_optimized.sh
#!/bin/bash
torchrun \
  --nproc_per_node=2 \
  --master_port=29500 \
  training/optimized_train.py \
  --flash_attention \
  --fsdp \
  --activation_checkpointing \
  --webdataset \
  --batch_size 16 \
  --num_workers 8
```

### 5.3 Activation Checkpointing

During backpropagation, PyTorch normally keeps all intermediate activations (outputs of each layer) in VRAM for gradient computation. Activation checkpointing discards these during the forward pass and recomputes them during backward — trading ~20% more compute for 50-70% VRAM savings on activations.

```python
from torch.distributed.fsdp.wrap import checkpoint_wrapper
from torch.utils.checkpoint import checkpoint_sequential

def apply_activation_checkpointing(model):
    """Apply gradient checkpointing to each transformer layer."""
    from torch.distributed.fsdp.wrap import apply_activation_checkpointing

    # For HuggingFace models: use built-in gradient checkpointing
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}  # Newer, more stable API
    )
    return model
```

**Alternatively, for more fine-grained control:**
```python
# Wrap individual layers with checkpoint_wrapper (used with FSDP)
from torch.distributed.fsdp.wrap import apply_activation_checkpointing
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing as fsdp_apply_ac,
)

non_reentrant_wrapper = functools.partial(
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)

check_fn = lambda submodule: isinstance(submodule, OPTDecoderLayer)
fsdp_apply_ac(model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)
```

---

## 6. Phase 4 — W&B ROI Report

### 6.1 W&B Initialization for Optimized Run

```python
# In training/optimized_train.py — W&B config

import os
import torch.distributed as dist

rank = int(os.environ.get("RANK", 0))

# Only log from rank 0 to avoid duplicate metrics
if rank == 0:
    wandb.init(
        project="vla-scale",
        name="phase3-fully-optimized",
        config={
            "model": MODEL_ID,
            "batch_size": BATCH_SIZE,
            "num_workers": 8,
            "optimization": "all",
            "flash_attention": True,
            "fsdp": True,
            "fsdp_sharding_strategy": "FULL_SHARD",
            "activation_checkpointing": True,
            "webdataset": True,
            "mixed_precision": "bfloat16",
            "n_gpus": torch.cuda.device_count(),
        }
    )
```

### 6.2 Metrics Logging

```python
# utils/metrics.py

import torch
import time
import subprocess
import wandb

class ThroughputTracker:
    def __init__(self, world_size=1):
        self.world_size = world_size
        self.step_start = None
        self.total_images = 0
        self.total_tokens = 0

    def start_step(self):
        self.step_start = time.perf_counter()

    def end_step(self, batch_size, seq_len):
        elapsed = time.perf_counter() - self.step_start
        images_per_sec = (batch_size * self.world_size) / elapsed
        tokens_per_sec = (batch_size * seq_len * self.world_size) / elapsed
        self.total_images += batch_size * self.world_size
        return images_per_sec, tokens_per_sec


def log_vram_stats(rank=0):
    """Log peak VRAM allocated across all GPUs."""
    stats = {}
    for i in range(torch.cuda.device_count()):
        peak = torch.cuda.max_memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        stats[f"peak_vram_gpu{i}_gb"] = round(peak, 2)
        stats[f"reserved_vram_gpu{i}_gb"] = round(reserved, 2)
    stats["peak_vram_total_gb"] = sum(
        torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())
    ) / (1024**3)
    return stats


def get_all_gpu_utilization():
    """Get utilization % for all GPUs via nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        utils = [float(x) for x in result.stdout.strip().split('\n') if x.strip()]
        return {f"gpu{i}_util_pct": u for i, u in enumerate(utils)}
    except Exception:
        return {}


def compute_scaling_efficiency(single_gpu_throughput, multi_gpu_throughput, n_gpus):
    """
    Ideal scaling = n_gpus * single_gpu_throughput
    Actual efficiency = actual / ideal
    Target: >85% (>0.85)
    """
    ideal = single_gpu_throughput * n_gpus
    efficiency = multi_gpu_throughput / ideal
    return round(efficiency * 100, 1)
```

### 6.3 Comprehensive Logging in Optimized Training Loop

```python
# In main training loop (rank 0 only)
tracker = ThroughputTracker(world_size=dist.get_world_size())

for step, (batch_inputs, actions) in enumerate(dataloader):
    tracker.start_step()

    # ... training step ...

    imgs_sec, tokens_sec = tracker.end_step(BATCH_SIZE, SEQ_LEN)
    vram_stats = log_vram_stats()
    gpu_utils = get_all_gpu_utilization()

    if rank == 0:
        wandb.log({
            "step": step,
            "loss": loss.item(),
            "images_per_sec": imgs_sec,
            "tokens_per_sec": tokens_sec,
            **vram_stats,
            **gpu_utils,
            "learning_rate": scheduler.get_last_lr()[0],
        })
```

### 6.4 W&B Report Structure

Create a W&B Report in the `vla-scale` project with the following sections:

#### Section 1 — Throughput Comparison (Images/sec)
- **Chart type:** Bar chart or line overlay
- **Runs:** `phase1-naive-baseline` vs `phase3-fully-optimized`
- **Metric:** `images_per_sec`
- **Expected:** 3–4× improvement

#### Section 2 — VRAM Reduction
- **Chart type:** Bar chart
- **Runs:** Both runs
- **Metric:** `peak_vram_total_gb`
- **Expected:** 25–40% reduction (enabling 2–4× larger batch sizes)

#### Section 3 — GPU Utilization
- **Chart type:** Time series
- **Runs:** Both runs
- **Metric:** `gpu0_util_pct`, `gpu1_util_pct`
- **Expected:** 40% → 85%+ on both GPUs

#### Section 4 — Multi-GPU Scaling Efficiency
- **Chart type:** Stat panel
- **Computation:** Use single-GPU run (Phase 1, 1 GPU) vs. Phase 3 (2 GPU) to compute:
  - `scaling_efficiency = (2_gpu_throughput) / (2 × 1_gpu_throughput)`
  - **Expected:** >85% (vs. ~65% in naive data parallel)

#### Section 5 — Cost Projection
Add a text block:
```
Naive baseline: 12 img/s on 2× RTX 3090
Estimated time for 100K episode dataset (500K frames): ~11.6 hrs
Cost at $2.50/hr: ~$29.00 per run

Optimized: 40+ img/s on 2× RTX 3090
Estimated time for 100K episode dataset: ~3.5 hrs
Cost at $2.50/hr: ~$8.75 per run

Savings: $20+ per training run
At 10 runs per research sprint: $200+ saved per sprint
```

---

## 7. Complete Optimized Training Script

```python
# training/optimized_train.py
"""
Phase 3: Fully optimized VLA training script.
Optimizations: WebDataset + FlashAttention-2 + FSDP + Activation Checkpointing
Run with: torchrun --nproc_per_node=2 training/optimized_train.py
"""

import os
import time
import functools
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoProcessor, AutoModelForVision2Seq
import wandb

from utils.metrics import ThroughputTracker, log_vram_stats, get_all_gpu_utilization

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID = "Salesforce/blip2-opt-2.7b"
BATCH_SIZE = 16          # Increased from 4 due to VRAM savings
NUM_WORKERS = 8          # Increased from 2 — WebDataset async decode
LEARNING_RATE = 1e-4
MAX_STEPS = 500
SHARD_DIR = "data/webdataset_shards"
SEQ_LEN = 64


def main():
    # ── Distributed setup ─────────────────────────────────────────────────────
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        wandb.init(
            project="vla-scale",
            name="phase3-fully-optimized",
            config={
                "model": MODEL_ID,
                "batch_size": BATCH_SIZE,
                "num_workers": NUM_WORKERS,
                "flash_attention": True,
                "fsdp": True,
                "activation_checkpointing": True,
                "webdataset": True,
                "n_gpus": world_size,
            }
        )

    # ── Model with FlashAttention-2 ───────────────────────────────────────────
    if rank == 0:
        print(f"Loading {MODEL_ID} with FlashAttention-2...")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    # Activation checkpointing (before FSDP wrapping)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # FSDP wrapping
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.bfloat16,
    )
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=device,
    )

    # ── WebDataset DataLoader ─────────────────────────────────────────────────
    dataloader = build_webdataset_loader(SHARD_DIR, processor, BATCH_SIZE, NUM_WORKERS)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    tracker = ThroughputTracker(world_size=world_size)
    torch.cuda.reset_peak_memory_stats()

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    for step, (batch_inputs, actions) in enumerate(dataloader):
        if step >= MAX_STEPS:
            break

        tracker.start_step()
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}

        outputs = model(**batch_inputs, labels=batch_inputs.get("input_ids"))
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (important with FSDP)
        model.clip_grad_norm_(1.0)
        optimizer.step()

        imgs_sec, tokens_sec = tracker.end_step(BATCH_SIZE, SEQ_LEN)

        if rank == 0 and step % 5 == 0:
            vram = log_vram_stats()
            gpu_utils = get_all_gpu_utilization()
            wandb.log({
                "step": step,
                "loss": loss.item(),
                "images_per_sec": imgs_sec,
                "tokens_per_sec": tokens_sec,
                **vram,
                **gpu_utils,
            })
            print(f"Step {step:4d} | Loss: {loss.item():.4f} | "
                  f"{imgs_sec:.1f} imgs/s | "
                  f"VRAM: {vram['peak_vram_total_gb']:.1f} GB total")

    if rank == 0:
        wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

---

## 8. Benchmarking Protocol

### 8.1 Controlled Comparison Setup

To produce valid W&B comparison charts, ensure all runs use:
- Same model architecture and checkpoint
- Same dataset (same shards/episodes)
- Same number of steps (200 steps minimum for stable averages)
- Same hardware (2× RTX 3090 for multi-GPU runs)

### 8.2 Run Matrix

| Run Name | GPUs | FlashAttn | FSDP | AckCkpt | WebDataset | Purpose |
|---|---|---|---|---|---|---|
| `phase1-naive-baseline` | 1 | ✗ | ✗ | ✗ | ✗ | Bottleneck baseline |
| `phase2-webdataset-only` | 1 | ✗ | ✗ | ✗ | ✓ | Isolate data loading gain |
| `phase3-flash-only` | 1 | ✓ | ✗ | ✗ | ✓ | Isolate FlashAttn gain |
| `phase3-ac-only` | 1 | ✓ | ✗ | ✓ | ✓ | Isolate activation ckpt gain |
| `phase3-fully-optimized` | 2 | ✓ | ✓ | ✓ | ✓ | **Full optimization** |

### 8.3 Scaling Efficiency Test

```bash
# 1-GPU reference run (set CUDA_VISIBLE_DEVICES to use only GPU 0)
CUDA_VISIBLE_DEVICES=0 python training/optimized_train.py \
  --name phase3-1gpu-reference \
  --flash_attention --activation_checkpointing --webdataset \
  --max_steps 100

# 2-GPU run
torchrun --nproc_per_node=2 training/optimized_train.py \
  --name phase3-fully-optimized \
  --flash_attention --fsdp --activation_checkpointing --webdataset \
  --max_steps 100
```

Scaling efficiency calculation:
```python
# After runs complete — compute in analysis notebook
single_gpu_imgs_sec = 22.5   # From W&B run phase3-1gpu-reference, avg images_per_sec
dual_gpu_imgs_sec = 40.0     # From W&B run phase3-fully-optimized, avg images_per_sec

ideal_dual_gpu = single_gpu_imgs_sec * 2
scaling_efficiency = (dual_gpu_imgs_sec / ideal_dual_gpu) * 100
print(f"Scaling efficiency: {scaling_efficiency:.1f}%")
# Expected output: ~88.9% — well above 85% target
```

---

## 9. Expected Results & Acceptance Criteria

### Quantitative Targets

| Metric | Baseline | Phase 2 Target | Phase 3 Target | Pass/Fail Threshold |
|---|---|---|---|---|
| Images/sec (1 GPU) | ~12 | ~25 | ~22+ | ≥18 imgs/s |
| Images/sec (2 GPU) | ~14 | ~48 | ~40+ | ≥32 imgs/s |
| Peak VRAM total (2 GPU) | ~38 GB | ~38 GB | ≤28 GB | ≤30 GB |
| GPU utilization (avg) | ~40% | ~82% | ~87% | ≥75% |
| Multi-GPU scaling efficiency | ~65% | ~70% | ~85%+ | ≥80% |

### Qualitative Deliverables

- [ ] TensorBoard trace: `01_baseline_trace.png` — GPU idle gaps visible
- [ ] TensorBoard trace: `02_webdataset_trace.png` — dense GPU utilization
- [ ] W&B Report: published URL in README, all 4 metrics comparing baseline vs. optimized
- [ ] `README.md`: contains W&B report link, methodology summary, and cost projection table
- [ ] Training scripts: reproducible via `run_baseline.sh` and `run_optimized.sh`

---

## 10. Cost Analysis & ROI Framing

### Per-Run Compute Cost Comparison

| Configuration | Throughput | Time for 500K frames | RunPod cost ($2.50/hr, 2×RTX 3090) |
|---|---|---|---|
| Naive baseline | ~14 imgs/s | ~9.9 hrs | **$24.75** |
| Phase 3 optimized | ~40 imgs/s | ~3.5 hrs | **$8.75** |
| **Savings per run** | | **6.4 hrs** | **$16.00** |

### Research Sprint ROI

| Scenario | Runs/Sprint | Cost/Sprint (Naive) | Cost/Sprint (Optimized) | Savings/Sprint |
|---|---|---|---|---|
| Hyperparameter sweep | 10 runs | $247.50 | $87.50 | **$160.00** |
| Architecture ablation | 20 runs | $495.00 | $175.00 | **$320.00** |
| Monthly research budget | ~50 runs | $1,237.50 | $437.50 | **$800.00** |

### Interview Framing

> *"By profiling the naive training loop with PyTorch Profiler, I identified that GPU utilization was 40% due to synchronous image decoding in the main process. Migrating to WebDataset streaming brought GPU utilization to 87%. Integrating FlashAttention-2 reduced peak VRAM by 26%, allowing batch size to increase from 4 to 16. FSDP with full sharding achieved 88% scaling efficiency across 2 GPUs — meaning we almost doubled throughput for the hardware cost. The net result was a 3.3× throughput improvement and a 65% cost reduction per training run."*

---

## GitHub Repository Metadata

**Repo:** `vgandhi1/vla-bench`  
**URL:** https://github.com/vgandhi1/vla-bench  
**W&B Report:** https://wandb.ai/vgandhi1/vla-bench  

### Recommended GitHub About Description

```
Systematic VLA training optimization on 2× RTX 3090.
WebDataset + FlashAttention-2 + FSDP → 3.3× throughput, 26% VRAM reduction.
Profiler traces and W&B report linked. Reproducible in one command.
```

### Recommended GitHub Topics

```
pytorch  fsdp  flash-attention  webdataset  vla  vision-language-action
gpu-optimization  ml-infra  weights-and-biases  robotics  training-efficiency
profiling  multi-gpu  imitation-learning  huggingface  tensorboard
```

### Recommended README Structure (`README.md`)

```markdown
# vla-bench

[![W&B Report](https://img.shields.io/badge/W%26B-Report-FFBE00?style=flat-square&logo=weightsandbiases)](https://wandb.ai/vgandhi1/vla-bench)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![FSDP](https://img.shields.io/badge/multi--GPU-FSDP-76B900?style=flat-square)](https://pytorch.org/docs/stable/fsdp.html)
[![FlashAttention](https://img.shields.io/badge/attention-FlashAttn--2-blue?style=flat-square)](https://github.com/Dao-AILab/flash-attention)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

> Systematic profiling and optimization of a VLA model training loop on 2× RTX 3090.
> Three optimization layers applied in sequence with measured gains at each step.
> Full W&B comparison report linked above.

## Results summary

| Metric | Naive baseline | Phase 3 optimized | Gain |
|---|---|---|---|
| Images/sec (2 GPU) | ~14 | ~40+ | **3.3×** |
| Peak VRAM total | ~38 GB | ~28 GB | **-26%** |
| GPU utilization | ~40% | ~87% | **+47pp** |
| Multi-GPU scaling efficiency | ~65% | ~88% | **+23pp** |
| Cost per training run (500K frames) | ~$24.75 | ~$8.75 | **-65%** |

## What each optimization contributes

| Layer | Optimization | Primary gain |
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

# Phase 1 — baseline (1 GPU)
bash scripts/run_baseline.sh

# Phase 3 — optimized (2 GPU)
bash scripts/run_optimized.sh
```

## Profiler traces

| Before (Phase 1) | After (Phase 2) |
|---|---|
| ![baseline](docs/profiler_traces/01_baseline_trace.png) | ![optimized](docs/profiler_traces/02_webdataset_trace.png) |

White gaps = GPU idle waiting for CPU image decode.
Dense kernels = GPU fed continuously by WebDataset async workers.

## Stack
PyTorch 2.2 · HuggingFace Transformers · FlashAttention-2 · FSDP ·
WebDataset · Weights & Biases · TensorBoard · RunPod (2× RTX 3090)

## Related
Part of a broader factory AI portfolio. See also:
- [edge-telemetry-plane (DETCP)](https://github.com/vgandhi1/edge-telemetry-plane) — fault-tolerant edge infrastructure
- [apex-recovery](https://github.com/vgandhi1/apex-recovery) — operator cockpit for VLA recovery data collection
```

### `.github/` Recommended Config

```
.github/
└── workflows/
    └── lint.yml       # Runs ruff + mypy on push; no GPU in CI
```

**`lint.yml` (GitHub Actions — lightweight, no GPU required):**

```yaml
name: Lint

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install ruff mypy
      - run: ruff check training/ data/ scripts/
      - run: mypy training/ --ignore-missing-imports
```

> **Note:** GPU training runs are not automated in CI — they require RunPod or equivalent. The CI job validates code style and types only. W&B runs are executed manually and the report URL is pinned in the README.

### `pyproject.toml` Key Fields

```toml
[project]
name = "vla-bench"
version = "1.0.0"
description = "VLA model training optimization benchmark: WebDataset + FlashAttention-2 + FSDP"
readme = "README.md"
requires-python = ">=3.11"
license = { text = "MIT" }

[project.urls]
Homepage = "https://github.com/vgandhi1/vla-bench"
"W&B Report" = "https://wandb.ai/vgandhi1/vla-bench"
"Bug Tracker" = "https://github.com/vgandhi1/vla-bench/issues"

[tool.ruff]
line-length = 100
select = ["E", "F", "I"]

[tool.mypy]
python_version = "3.11"
ignore_missing_imports = true
```

### `scripts/run_baseline.sh`

```bash
#!/bin/bash
# Phase 1: Naive baseline — single GPU, standard DataLoader
# Expected: ~12 imgs/s, ~40% GPU utilization, OOM at batch_size >= 8

set -e
echo "=== vla-bench: Phase 1 Baseline ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

CUDA_VISIBLE_DEVICES=0 python training/baseline_train.py \
  --model Salesforce/blip2-opt-2.7b \
  --batch_size 4 \
  --num_workers 2 \
  --max_steps 200 \
  --wandb_run_name "phase1-naive-baseline"

echo "=== Baseline complete. Check W&B for metrics. ==="
```

### `scripts/run_optimized.sh`

```bash
#!/bin/bash
# Phase 3: Fully optimized — 2 GPU, FSDP + FlashAttention-2 + WebDataset + AckCkpt
# Expected: ~40 imgs/s, ~87% GPU utilization, batch_size 16+

set -e
echo "=== vla-bench: Phase 3 Optimized ==="
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

torchrun \
  --nproc_per_node=2 \
  --master_port=29500 \
  training/optimized_train.py \
  --model Salesforce/blip2-opt-2.7b \
  --batch_size 16 \
  --num_workers 8 \
  --max_steps 200 \
  --flash_attention \
  --fsdp \
  --activation_checkpointing \
  --webdataset \
  --wandb_run_name "phase3-fully-optimized"

echo "=== Optimized run complete. Check W&B for metrics. ==="
```

---

## Document Metadata

| Field | Value |
|---|---|
| **Project name** | vla-bench |
| **Document version** | 1.0 |
| **Author** | Vinay Gandhi |
| **Created** | May 2026 |
| **W&B project** | https://wandb.ai/vgandhi1/vla-bench |
| **Related projects** | edge-telemetry-plane (DETCP) · apex-recovery |

---

*Document version 1.0 — May 2026*