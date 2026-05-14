"""PyTorch Profiler helpers for VLA training benchmarking."""

import torch
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler


def make_profiler(log_dir: str, wait: int = 5, warmup: int = 5, active: int = 10):
    """
    Build a PyTorch Profiler configured for TensorBoard trace export.

    Usage:
        prof = make_profiler("runs/baseline")
        prof.start()
        for step, batch in enumerate(loader):
            ...train step...
            prof.step()
        prof.stop()
    """
    return profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=wait, warmup=warmup, active=active),
        on_trace_ready=tensorboard_trace_handler(log_dir),
        record_shapes=True,
        with_stack=True,
    )


def print_top_ops(prof, top_k: int = 20):
    """Print the top-k CUDA operations by self CUDA time."""
    print(
        prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=top_k,
        )
    )
