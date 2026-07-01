"""Generate synthetic VLA episodes mimicking teleoperation structure."""

import os
import json
import numpy as np
from PIL import Image
import random
import argparse


def generate_synthetic_episodes(n_episodes=500, output_dir="data/raw_episodes"):
    """
    Each episode: sequence of (image, action_vector, language_instruction).
    Images are 224×224 RGB JPEGs; actions are 7-DOF vectors.
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
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img_path = os.path.join(ep_dir, f"frame_{f_idx:04d}.jpg")
            img.save(img_path, quality=85)

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

        if (ep_idx + 1) % 50 == 0:
            print(f"Generated {ep_idx + 1}/{n_episodes} episodes...")

    print(f"Generated {n_episodes} synthetic episodes in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_episodes", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="data/raw_episodes")
    args = parser.parse_args()
    generate_synthetic_episodes(args.n_episodes, args.output_dir)
