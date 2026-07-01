"""Throughput, VRAM, and scaling efficiency tracking utilities."""

import time
import subprocess
import torch
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


def log_vram_stats():
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


def get_gpu_utilization():
    """Query nvidia-smi for single GPU utilization percent."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        return float(result.stdout.strip().split('\n')[0])
    except Exception:
        return 0.0


def compute_scaling_efficiency(single_gpu_throughput, multi_gpu_throughput, n_gpus):
    """
    Ideal scaling = n_gpus * single_gpu_throughput.
    Target: >85% (>0.85).
    """
    ideal = single_gpu_throughput * n_gpus
    efficiency = multi_gpu_throughput / ideal
    return round(efficiency * 100, 1)
