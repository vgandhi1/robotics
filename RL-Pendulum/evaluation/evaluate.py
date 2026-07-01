"""
Policy evaluation and visualization for RL-Pendulum.

Generates the following outputs:
  - Console summary: success rate, mean reward, mean episode length.
  - Plots:
      1. Balance time distribution (histogram).
      2. Phase portrait: pitch angle vs. angular velocity.
      3. Single episode trajectory: state components over time.
      4. Reward curve over evaluation episodes.

Usage:
    python evaluation/evaluate.py --model logs/best_model.zip --episodes 50
    python evaluation/evaluate.py --model export/model.onnx --onnx --episodes 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load_sb3_policy(model_path: str):
    """Returns a callable policy(obs) → action from an SB3 model."""
    from stable_baselines3 import PPO
    model = PPO.load(model_path, device="cpu")
    model.policy.set_training_mode(False)

    def policy_fn(obs: np.ndarray) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=True)
        return action

    return policy_fn


def _load_onnx_policy(onnx_path: str):
    """Returns a callable policy(obs) → action from an ONNX model."""
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    def policy_fn(obs: np.ndarray) -> np.ndarray:
        obs_batch = obs[None].astype(np.float32)
        action = sess.run(None, {input_name: obs_batch})[0]
        return action[0]

    return policy_fn


def run_episode(env, policy_fn) -> dict:
    """
    Run a single episode and return trajectory data.

    Returns:
        dict with keys: rewards, pitches, pitch_rates, lw_speeds, rw_speeds,
                        actions, episode_length, total_reward, success
    """
    obs, _ = env.reset()
    done = False
    data = {
        "rewards": [],
        "pitches": [],
        "pitch_rates": [],
        "lw_speeds": [],
        "rw_speeds": [],
        "actions": [],
    }

    while not done:
        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        data["rewards"].append(float(reward))
        data["pitches"].append(info.get("pitch_rad", float(obs[0])))
        data["pitch_rates"].append(float(obs[1]))
        data["lw_speeds"].append(float(obs[2]))
        data["rw_speeds"].append(float(obs[3]))
        data["actions"].append(float(action[0]) if hasattr(action, "__len__") else float(action))

    ep_len = len(data["rewards"])
    data["episode_length"] = ep_len
    data["total_reward"] = float(np.sum(data["rewards"]))
    data["success"] = ep_len >= 999  # 10s at 100 Hz

    return data


def evaluate_policy_full(
    model_path: str,
    n_episodes: int = 50,
    use_dr: bool = False,
    onnx_mode: bool = False,
    plot: bool = True,
    output_dir: str = "logs/eval",
) -> dict:
    """
    Evaluate a trained policy across multiple episodes and generate plots.

    Args:
        model_path:  Path to SB3 .zip or ONNX .onnx model.
        n_episodes:  Number of evaluation episodes.
        use_dr:      Use domain randomization during evaluation (stress test).
        onnx_mode:   Load as ONNX model (instead of SB3).
        plot:        Generate and save plots.
        output_dir:  Directory to save plot images.

    Returns:
        dict with summary statistics.
    """
    from envs.pendulum_env import PendulumBalanceEnv
    from envs.domain_randomization import DomainRandomizationWrapper, DRConfig

    env = PendulumBalanceEnv()
    if use_dr:
        env = DomainRandomizationWrapper(env, DRConfig())

    policy_fn = _load_onnx_policy(model_path) if onnx_mode else _load_sb3_policy(model_path)

    print(f"\nEvaluating {'ONNX' if onnx_mode else 'SB3'} model: {model_path}")
    print(f"  Episodes: {n_episodes}  |  DR: {'ON' if use_dr else 'OFF'}")
    print("-" * 50)

    all_episodes = []
    for ep in range(n_episodes):
        data = run_episode(env, policy_fn)
        all_episodes.append(data)
        status = "✓" if data["success"] else "✗"
        print(
            f"  [{ep+1:3d}/{n_episodes}] {status}  "
            f"length={data['episode_length']:4d}  "
            f"reward={data['total_reward']:7.1f}"
        )

    env.close()

    # ── Summary statistics ─────────────────────────────────────────────────────
    success_rate = float(np.mean([ep["success"] for ep in all_episodes]))
    mean_reward = float(np.mean([ep["total_reward"] for ep in all_episodes]))
    std_reward = float(np.std([ep["total_reward"] for ep in all_episodes]))
    mean_length = float(np.mean([ep["episode_length"] for ep in all_episodes]))
    median_length = float(np.median([ep["episode_length"] for ep in all_episodes]))

    summary = {
        "success_rate": success_rate,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_episode_length": mean_length,
        "median_episode_length": median_length,
        "n_episodes": n_episodes,
    }

    print("\n" + "=" * 50)
    print(f"  Success rate     : {success_rate:.1%}")
    print(f"  Mean reward      : {mean_reward:.1f} ± {std_reward:.1f}")
    print(f"  Mean ep. length  : {mean_length:.0f} steps")
    print(f"  Median ep. length: {median_length:.0f} steps")
    print("=" * 50)

    if plot:
        _generate_plots(all_episodes, output_dir, model_path)

    return summary


def _generate_plots(
    all_episodes: list[dict],
    output_dir: str,
    model_label: str,
) -> None:
    """Generate evaluation plots and save to output_dir."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="darkgrid", palette="muted")
    except ImportError:
        print("[WARN] matplotlib/seaborn not available; skipping plots")
        return

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = Path(model_label).stem

    ep_lengths = [ep["episode_length"] for ep in all_episodes]
    ep_rewards = [ep["total_reward"] for ep in all_episodes]

    # ── Plot 1: Balance time distribution ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ep_lengths, bins=25, edgecolor="black", alpha=0.8, color="#4c72b0")
    ax.axvline(999, color="green", linestyle="--", linewidth=1.5, label="Success threshold (999 steps)")
    ax.set_xlabel("Episode Length (steps)")
    ax.set_ylabel("Count")
    ax.set_title(f"Balance Time Distribution — {model_name}")
    ax.legend()
    fig.tight_layout()
    path1 = out_dir / f"{model_name}_balance_dist.png"
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path1}")

    # ── Plot 2: Phase portrait ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    for ep in all_episodes[:20]:  # First 20 episodes
        pitches = np.array(ep["pitches"])
        rates = np.array(ep["pitch_rates"])
        color = "#2ca02c" if ep["success"] else "#d62728"
        ax.plot(pitches, rates, alpha=0.4, linewidth=0.8, color=color)
    ax.scatter([0], [0], s=100, marker="*", color="gold", zorder=5, label="Equilibrium")
    ax.axvline(0.5, color="red", linestyle=":", alpha=0.5, label="Fall boundary")
    ax.axvline(-0.5, color="red", linestyle=":", alpha=0.5)
    ax.set_xlabel("Pitch Angle θ (rad)")
    ax.set_ylabel("Pitch Rate θ̇ (rad/s)")
    ax.set_title(f"Phase Portrait — {model_name}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path2 = out_dir / f"{model_name}_phase_portrait.png"
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path2}")

    # ── Plot 3: Best episode trajectory ───────────────────────────────────────
    best_ep = max(all_episodes, key=lambda e: e["episode_length"])
    t = np.arange(len(best_ep["pitches"])) * 0.01  # 100 Hz → seconds

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t, np.degrees(best_ep["pitches"]), label="Pitch angle (°)", color="#1f77b4")
    axes[0].axhline(0, color="gray", linewidth=0.5)
    axes[0].set_ylabel("Pitch (°)")
    axes[0].set_title(f"Best Episode Trajectory — {model_name}  (length={len(t)} steps)")

    axes[1].plot(t, best_ep["lw_speeds"], label="Left wheel", alpha=0.8, color="#ff7f0e")
    axes[1].plot(t, best_ep["rw_speeds"], label="Right wheel", alpha=0.8, color="#2ca02c")
    axes[1].set_ylabel("Wheel Speed (rad/s)")
    axes[1].legend(fontsize=8)

    axes[2].plot(t, best_ep["actions"], label="Motor action", color="#9467bd")
    axes[2].axhline(0, color="gray", linewidth=0.5)
    axes[2].set_ylabel("Action (normalized)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    path3 = out_dir / f"{model_name}_trajectory.png"
    fig.savefig(path3, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path3}")

    # ── Plot 4: Reward across episodes ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#2ca02c" if r > 500 else "#d62728" for r in ep_rewards]
    ax.bar(range(len(ep_rewards)), ep_rewards, color=colors, alpha=0.8)
    ax.axhline(np.mean(ep_rewards), color="navy", linestyle="--",
               label=f"Mean = {np.mean(ep_rewards):.0f}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title(f"Episode Rewards — {model_name}")
    ax.legend()
    fig.tight_layout()
    path4 = out_dir / f"{model_name}_rewards.png"
    fig.savefig(path4, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path4}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RL-Pendulum policy")
    parser.add_argument("--model", required=True, help="Path to SB3 .zip or ONNX .onnx model")
    parser.add_argument("--episodes", type=int, default=50, help="Number of eval episodes")
    parser.add_argument("--dr", action="store_true", help="Enable domain randomization during eval")
    parser.add_argument("--onnx", action="store_true", help="Load model as ONNX (not SB3)")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot generation")
    parser.add_argument("--output-dir", default="logs/eval", help="Directory for plot outputs")
    args = parser.parse_args()

    evaluate_policy_full(
        model_path=args.model,
        n_episodes=args.episodes,
        use_dr=args.dr,
        onnx_mode=args.onnx,
        plot=not args.no_plot,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
