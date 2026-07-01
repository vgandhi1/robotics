import gymnasium as gym

from .pendulum_env import PendulumBalanceEnv
from .domain_randomization import DomainRandomizationWrapper

gym.register(
    id="PendulumBalance-v0",
    entry_point="envs.pendulum_env:PendulumBalanceEnv",
    max_episode_steps=1000,
)

__all__ = ["PendulumBalanceEnv", "DomainRandomizationWrapper"]
