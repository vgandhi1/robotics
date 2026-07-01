"""
Main PPO training script for RL-Pendulum.

Execution phases:
  Phase 2:  Train baseline PPO for 5M steps (no domain randomization).
  Phase 3:  Continue or retrain with domain randomization for 10M steps.

Usage:
    python training/train.py --config configs/ppo_config.yaml
    python training/train.py --config configs/ppo_config.yaml --no-dr
    python training/train.py --resume logs/best_model.zip --dr --total-timesteps 5000000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import yaml
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor, SubprocVecEnv, DummyVecEnv

# Ensure project root is on sys.path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.pendulum_env import PendulumBalanceEnv
from envs.domain_randomization import DomainRandomizationWrapper, DRConfig, make_env
from training.callbacks import EvalAndSaveCallback, DRParamLogCallback, ProgressCallback


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _get_activation(name: str):
    import torch.nn as nn
    return {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}[name]


def build_envs(
    cfg: dict,
    n_envs: int,
    use_dr: bool,
    seed: int,
    multiprocess: bool = False,
) -> VecMonitor:
    """Build vectorized training environments."""
    dr_config = DRConfig.from_dict(cfg) if use_dr else None
    env_fns = [
        make_env(rank=i, dr_config=dr_config, use_dr=use_dr, seed=seed)
        for i in range(n_envs)
    ]
    VecEnvCls = SubprocVecEnv if (multiprocess and n_envs > 1) else DummyVecEnv
    vec_env = VecEnvCls(env_fns)
    return VecMonitor(vec_env)


def build_eval_env(seed: int = 9999) -> VecMonitor:
    """Build a single evaluation environment (no DR, nominal physics)."""
    def _init():
        env = PendulumBalanceEnv()
        env.reset(seed=seed)
        return env

    return VecMonitor(DummyVecEnv([_init]))


def build_policy_kwargs(cfg: dict) -> dict:
    pk = cfg.get("ppo", {}).get("policy_kwargs", {})
    net_arch = pk.get("net_arch", {"pi": [64, 64], "vf": [64, 64]})
    activation_name = pk.get("activation_fn", "tanh")
    return {
        "net_arch": net_arch,
        "activation_fn": _get_activation(activation_name),
        "ortho_init": pk.get("ortho_init", True),
    }


def train(
    config_path: str = "configs/ppo_config.yaml",
    use_dr: bool = True,
    resume_from: Optional[str] = None,
    total_timesteps: Optional[int] = None,
    seed: int = 42,
    multiprocess: bool = False,
) -> PPO:
    """
    Full training pipeline.

    Args:
        config_path:      Path to YAML config.
        use_dr:           Whether to enable domain randomization.
        resume_from:      Path to a saved SB3 model to continue training from.
        total_timesteps:  Override timesteps from config.
        seed:             Random seed.
        multiprocess:     Use SubprocVecEnv for parallel environments.

    Returns:
        Trained PPO model.
    """
    cfg = load_config(config_path)
    ppo_cfg = cfg.get("ppo", {})
    train_cfg = cfg.get("training", {})

    np.random.seed(seed)
    torch.manual_seed(seed)

    n_envs = ppo_cfg.get("n_envs", 8)
    timesteps = total_timesteps or (
        ppo_cfg.get("total_timesteps_dr", 10_000_000)
        if use_dr
        else ppo_cfg.get("total_timesteps", 5_000_000)
    )

    log_dir = Path(train_cfg.get("log_dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    tb_log = str(log_dir / "tensorboard")
    save_path = str(log_dir / "best_model")

    print("=" * 60)
    print(f"  RL-Pendulum PPO Training")
    print(f"  Domain Randomization : {'ON' if use_dr else 'OFF'}")
    print(f"  Total timesteps      : {timesteps:,}")
    print(f"  Parallel envs        : {n_envs}")
    print(f"  Seed                 : {seed}")
    print("=" * 60)

    # ── Environments ─────────────────────────────────────────────────────────
    train_env = build_envs(cfg, n_envs=n_envs, use_dr=use_dr, seed=seed, multiprocess=multiprocess)
    eval_env = build_eval_env(seed=seed + 10000)

    # ── Model ─────────────────────────────────────────────────────────────────
    if resume_from:
        print(f"  Resuming from       : {resume_from}")
        model = PPO.load(
            resume_from,
            env=train_env,
            tensorboard_log=tb_log,
            device="auto",
        )
    else:
        policy_kwargs = build_policy_kwargs(cfg)
        model = PPO(
            policy=ppo_cfg.get("policy", "MlpPolicy"),
            env=train_env,
            learning_rate=ppo_cfg.get("learning_rate", 3e-4),
            n_steps=ppo_cfg.get("n_steps", 2048),
            batch_size=ppo_cfg.get("batch_size", 64),
            n_epochs=ppo_cfg.get("n_epochs", 10),
            gamma=ppo_cfg.get("gamma", 0.99),
            gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
            clip_range=ppo_cfg.get("clip_range", 0.2),
            ent_coef=ppo_cfg.get("ent_coef", 0.01),
            vf_coef=ppo_cfg.get("vf_coef", 0.5),
            max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
            normalize_advantage=ppo_cfg.get("normalize_advantage", True),
            policy_kwargs=policy_kwargs,
            tensorboard_log=tb_log,
            verbose=train_cfg.get("verbose", 1),
            seed=seed,
            device="auto",
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    callbacks = [
        EvalAndSaveCallback(
            eval_env=eval_env,
            eval_freq=train_cfg.get("eval_freq", 10_000),
            n_eval_episodes=train_cfg.get("n_eval_episodes", 20),
            save_path=save_path,
            verbose=1,
        ),
        ProgressCallback(log_freq=5000, verbose=1),
    ]
    if use_dr:
        callbacks.append(DRParamLogCallback(log_freq=1000))

    # ── Train ─────────────────────────────────────────────────────────────────
    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        reset_num_timesteps=(resume_from is None),
        progress_bar=True,
    )

    # Save final model
    final_path = str(log_dir / "final_model")
    model.save(final_path)
    print(f"\nTraining complete. Final model saved to {final_path}.zip")
    print(f"Best model saved to {save_path}.zip")

    train_env.close()
    eval_env.close()
    return model


def main():
    parser = argparse.ArgumentParser(description="Train RL-Pendulum PPO policy")
    parser.add_argument(
        "--config", type=str, default="configs/ppo_config.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--no-dr", action="store_true",
        help="Disable domain randomization (Phase 2 baseline)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to saved model to resume training from"
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=None,
        help="Override total training timesteps from config"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--multiprocess", action="store_true",
        help="Use SubprocVecEnv for parallel environments (faster on multi-core)"
    )
    args = parser.parse_args()

    train(
        config_path=args.config,
        use_dr=not args.no_dr,
        resume_from=args.resume,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        multiprocess=args.multiprocess,
    )


if __name__ == "__main__":
    main()
