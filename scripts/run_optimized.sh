#!/bin/bash
# Phase 3: Fully optimized — 2 GPU, FSDP + FlashAttention-2 + WebDataset + ActCkpt
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
