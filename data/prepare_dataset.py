"""
Convert raw episode data into WebDataset tar shards.

WebDataset enables asynchronous streaming from disk,
eliminating the CPU→GPU data starvation bottleneck.
"""

import os
import json
import io
import tarfile
import argparse


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

                with open(sample['image_path'], 'rb') as f:
                    img_bytes = f.read()
                img_info = tarfile.TarInfo(name=f"{key}.jpg")
                img_info.size = len(img_bytes)
                tar.addfile(img_info, io.BytesIO(img_bytes))

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/raw_episodes")
    parser.add_argument("--output_dir", type=str, default="data/webdataset_shards")
    parser.add_argument("--samples_per_shard", type=int, default=500)
    args = parser.parse_args()
    create_webdataset_shards(args.input_dir, args.output_dir, args.samples_per_shard)
