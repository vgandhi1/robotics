"""
Phase 3: Fully optimized VLA training script.

Optimizations: WebDataset + FlashAttention-2 + FSDP + Activation Checkpointing
Run with: torchrun --nproc_per_node=2 training/optimized_train.py [args]
"""

import os
import io
import time
import json
import functools
import argparse
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
import webdataset as wds
import wandb
from PIL import Image as PILImage

from utils.metrics import ThroughputTracker, log_vram_stats, get_all_gpu_utilization


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Salesforce/blip2-opt-2.7b")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--shard_dir", type=str, default="data/webdataset_shards")
    p.add_argument("--wandb_run_name", type=str, default="phase3-fully-optimized")
    p.add_argument("--flash_attention", action="store_true")
    p.add_argument("--fsdp", action="store_true")
    p.add_argument("--activation_checkpointing", action="store_true")
    p.add_argument("--webdataset", action="store_true")
    return p.parse_args()


SEQ_LEN = 64


def decode_sample(sample, processor):
    """Decode a WebDataset sample — runs in parallel worker processes."""
    image = PILImage.open(io.BytesIO(sample['jpg'])).convert("RGB")
    meta = json.loads(sample['json'].decode())

    inputs = processor(
        images=image,
        text=meta['instruction'],
        return_tensors="pt",
        padding="max_length",
        max_length=SEQ_LEN,
        truncation=True,
    )
    action = torch.tensor(meta['action'], dtype=torch.float32)
    return {k: v.squeeze(0) for k, v in inputs.items()}, action


def count_shards(shard_dir):
    return len([f for f in os.listdir(shard_dir) if f.endswith('.tar')]) - 1


def build_webdataset_loader(shard_dir, processor, batch_size, num_workers=8):
    """
    Build a WebDataset streaming dataloader.
    Image decoding runs in N parallel worker processes — GPU never waits for CPU.
    """
    n_shards = count_shards(shard_dir)
    shard_urls = f"{shard_dir}/shard_{{00000..{n_shards:05d}}}.tar"

    dataset = (
        wds.WebDataset(shard_urls, shardshuffle=True)
        .shuffle(1000)
        .decode("pil")
        .to_tuple("jpg", "json")
        .map(lambda x: decode_sample({"jpg": x[0], "json": x[1]}, processor))
        .batched(batch_size, partial=False)
    )

    return wds.WebLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4,
    )


def wrap_model_fsdp(model):
    """Wrap model with FSDP for multi-GPU parameter sharding."""
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.bfloat16,
    )

    try:
        from transformers.models.opt.modeling_opt import OPTDecoderLayer
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={OPTDecoderLayer},
        )
    except ImportError:
        auto_wrap_policy = None

    kwargs = dict(
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=torch.cuda.current_device(),
    )
    if auto_wrap_policy is not None:
        kwargs["auto_wrap_policy"] = auto_wrap_policy

    return FSDP(model, **kwargs)


def main():
    args = parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        wandb.init(
            project="vla-scale",
            name=args.wandb_run_name,
            config={
                "model": args.model,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "flash_attention": args.flash_attention,
                "fsdp": args.fsdp,
                "activation_checkpointing": args.activation_checkpointing,
                "webdataset": args.webdataset,
                "n_gpus": world_size,
                "mixed_precision": "bfloat16",
            }
        )

    if rank == 0:
        print(f"Loading {args.model}...")

    processor = AutoProcessor.from_pretrained(args.model)

    load_kwargs = {"torch_dtype": torch.bfloat16}
    if args.flash_attention:
        load_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForVision2Seq.from_pretrained(args.model, **load_kwargs)

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    if args.fsdp:
        model = wrap_model_fsdp(model)
    else:
        model = model.to(device)

    dataloader = build_webdataset_loader(
        args.shard_dir, processor, args.batch_size, args.num_workers
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    tracker = ThroughputTracker(world_size=world_size)
    torch.cuda.reset_peak_memory_stats()

    model.train()
    for step, (batch_inputs, actions) in enumerate(dataloader):
        if step >= args.max_steps:
            break

        tracker.start_step()
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}

        outputs = model(**batch_inputs, labels=batch_inputs.get("input_ids"))
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()

        if args.fsdp:
            model.clip_grad_norm_(1.0)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        imgs_sec, tokens_sec = tracker.end_step(args.batch_size, SEQ_LEN)

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
