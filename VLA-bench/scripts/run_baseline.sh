#!/bin/bash
# Phase 1: Naive baseline — single GPU, standard DataLoader
# Expected: ~12 imgs/s, ~40% GPU utilization; OOM at batch_size >= 8 at 7B scale (2.7B proxy used here)

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
