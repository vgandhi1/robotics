"""
Phase 1: Naive baseline training loop.

Purpose: Establish bottleneck baseline metrics before optimization.
Expected issues:
  - GPU utilization ~40% (CPU-bound image decoding)
  - Large VRAM spikes from naive attention
  - OOM at batch_size >= 8 at 7B scale (scripts use 2.7B proxy; swap MODEL_ID for the full bottleneck profile)
"""

import os
import time
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torch.profiler import record_function, ProfilerActivity
from transformers import AutoProcessor, AutoModelForVision2Seq
import wandb
import json
from PIL import Image

from utils.metrics import get_gpu_utilization
from utils.profiler_utils import make_profiler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Salesforce/blip2-opt-2.7b")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--data_dir", type=str, default="data/raw_episodes")
    p.add_argument("--wandb_run_name", type=str, default="phase1-naive-baseline")
    return p.parse_args()


class NaiveVLADataset(Dataset):
    """Naive dataset: loads all metadata upfront, decodes images on-the-fly in main process."""

    def __init__(self, data_dir, processor):
        self.processor = processor
        self.samples = []

        for ep_dir in sorted(os.listdir(data_dir))[:200]:
            meta_path = os.path.join(data_dir, ep_dir, "metadata.json")
            if not os.path.exists(meta_path):
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            for frame in meta['frames'][::5]:
                self.samples.append({
                    "image_path": frame['image_path'],
                    "action": frame['action'],
                    "instruction": meta['task'],
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Image decode runs in the MAIN PROCESS, blocking GPU
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
    args = parse_args()
    device = torch.device("cuda:0")

    wandb.init(
        project="vla-scale",
        name=args.wandb_run_name,
        config={
            "model": args.model,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "optimization": "none",
            "flash_attention": False,
            "fsdp": False,
            "activation_checkpointing": False,
            "webdataset": False,
        }
    )

    print(f"Loading model: {args.model}")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
    ).to(device)

    dataset = NaiveVLADataset(args.data_dir, processor)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    profiler = make_profiler("runs/baseline")
    model.train()
    profiler.start()

    step = 0
    epoch_start = time.time()

    for batch_inputs, actions in dataloader:
        if step >= args.max_steps:
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

        step_time = time.time() - step_start
        imgs_per_sec = args.batch_size / step_time
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


if __name__ == "__main__":
    main()
