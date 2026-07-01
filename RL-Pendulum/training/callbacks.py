"""
Custom Stable Baselines3 callbacks for PPO training.

Callbacks included:
  EvalAndSaveCallback:  Periodic evaluation on a held-out env; saves best model.
  DRParamLogCallback:   Logs the current domain randomization parameters to
                        TensorBoard for debugging.
  ProgressCallback:     Logs episode reward statistics to console + TensorBoard.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization


class EvalAndSaveCallback(BaseCallback):
    """
    Evaluates the policy every `eval_freq` training steps and saves the best
    model checkpoint based on mean episode reward.

    Args:
        eval_env:       Evaluation Gymnasium/VecEnv (no DR, nominal physics).
        eval_freq:      Evaluation interval in training timesteps.
        n_eval_episodes: Number of episodes per evaluation.
        save_path:      Directory to save the best model.
        verbose:        Verbosity level (0 = silent, 1 = info).
    """

    def __init__(
        self,
        eval_env: VecEnv,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 20,
        save_path: str = "logs/best_model",
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_path = Path(save_path)
        self.best_mean_reward = -np.inf
        self._last_eval_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step < self.eval_freq:
            return True

        self._last_eval_step = self.num_timesteps

        # Sync normalization statistics if using VecNormalize
        if hasattr(self.model, "get_vec_normalize_env"):
            sync_envs_normalization(self.training_env, self.eval_env)

        episode_rewards, episode_lengths = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=True,
            return_episode_rewards=True,
        )

        mean_reward = float(np.mean(episode_rewards))
        std_reward = float(np.std(episode_rewards))
        mean_length = float(np.mean(episode_lengths))
        success_rate = float(np.mean(np.array(episode_lengths) >= 999))

        # Log to TensorBoard
        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/std_reward", std_reward)
        self.logger.record("eval/mean_episode_length", mean_length)
        self.logger.record("eval/success_rate", success_rate)
        self.logger.dump(self.num_timesteps)

        if self.verbose >= 1:
            print(
                f"[Eval @ {self.num_timesteps:,}]  "
                f"reward={mean_reward:.1f}±{std_reward:.1f}  "
                f"len={mean_length:.0f}  "
                f"success={success_rate:.1%}"
            )

        # Save if improved
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(str(self.save_path))
            if self.verbose >= 1:
                print(f"  → New best ({mean_reward:.1f}), model saved to {self.save_path}")

        return True


class DRParamLogCallback(BaseCallback):
    """
    Logs the domain randomization parameters sampled for the current episode
    to TensorBoard for debugging and visualization.

    Requires the training env to be (or wrap) a DomainRandomizationWrapper.
    """

    def __init__(self, log_freq: int = 1000, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.log_freq = log_freq
        self._last_log_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_log_step < self.log_freq:
            return True

        self._last_log_step = self.num_timesteps

        # Try to extract DR params from the vec env
        try:
            env = self.training_env
            if hasattr(env, "envs"):
                dr_params = env.envs[0].current_dr_params
            elif hasattr(env, "current_dr_params"):
                dr_params = env.current_dr_params
            else:
                return True

            for key, val in dr_params.items():
                self.logger.record(f"dr/{key}", float(val))
        except (AttributeError, IndexError):
            pass

        return True


class ProgressCallback(BaseCallback):
    """
    Logs episode statistics (mean reward, episode length, success rate)
    to TensorBoard every `log_freq` timesteps. Also prints a progress bar
    summary.
    """

    def __init__(self, log_freq: int = 5000, verbose: int = 1) -> None:
        super().__init__(verbose)
        self.log_freq = log_freq
        self._episode_rewards: list[float] = []
        self._episode_lengths: list[int] = []
        self._last_log_step = 0

    def _on_step(self) -> bool:
        # Collect episode completions from the info dict
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._episode_rewards.append(info["episode"]["r"])
                self._episode_lengths.append(info["episode"]["l"])

        if (
            self._episode_rewards
            and self.num_timesteps - self._last_log_step >= self.log_freq
        ):
            self._last_log_step = self.num_timesteps
            mean_r = float(np.mean(self._episode_rewards[-100:]))
            mean_l = float(np.mean(self._episode_lengths[-100:]))
            success = float(
                np.mean(np.array(self._episode_lengths[-100:]) >= 999)
            )

            self.logger.record("train/mean_reward_100ep", mean_r)
            self.logger.record("train/mean_length_100ep", mean_l)
            self.logger.record("train/success_rate_100ep", success)

            if self.verbose >= 1:
                print(
                    f"  [t={self.num_timesteps:>9,}]  "
                    f"reward={mean_r:7.1f}  len={mean_l:5.0f}  "
                    f"success={success:.1%}"
                )

        return True
