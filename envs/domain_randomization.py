"""
Domain Randomization (DR) wrapper for PendulumBalanceEnv.

On every episode reset, samples physics parameters from calibrated uniform
distributions that bracket the real-world uncertainty. The policy is thus
trained on a *distribution* of environments, making the learned behavior
robust to the imprecise real-world conditions it will face at deployment.

Randomized parameters and their physical interpretations:
  - body_mass:         Uncertainty from battery charge, part tolerances
  - motor_friction:    Brush wear, gearbox lubrication variability
  - imu_noise_std:     MPU-6050 noise varies with temperature / vibration
  - imu_latency_ms:    I2C bus congestion + filter delay jitter
  - wheel_slip_coeff:  Floor surface / rubber hardness variability
  - action_delay_steps: Actuator + communication lag variability
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import gymnasium as gym
import numpy as np

from .pendulum_env import PendulumBalanceEnv, _DT


@dataclass
class DRConfig:
    """Defines the randomization range for each physics parameter."""

    # Body mass (kg)
    mass_low: float = 0.40
    mass_high: float = 0.60

    # Motor viscous friction (Nm)
    friction_low: float = 0.0005
    friction_high: float = 0.0015

    # IMU measurement noise standard deviation (rad / rad·s⁻¹)
    imu_noise_low: float = 0.001
    imu_noise_high: float = 0.020

    # IMU observation latency (milliseconds)
    latency_ms_low: float = 5.0
    latency_ms_high: float = 15.0

    # Wheel slip coefficient [0 = no slip, 1 = full slip]
    slip_low: float = 0.0
    slip_high: float = 0.05

    # Actuator action delay (control steps)
    action_delay_low: int = 0
    action_delay_high: int = 2

    @classmethod
    def from_dict(cls, d: dict) -> "DRConfig":
        dr = d.get("domain_randomization", {})
        return cls(
            mass_low=dr.get("mass_kg", {}).get("low", 0.40),
            mass_high=dr.get("mass_kg", {}).get("high", 0.60),
            friction_low=dr.get("friction_nm", {}).get("low", 0.0005),
            friction_high=dr.get("friction_nm", {}).get("high", 0.0015),
            imu_noise_low=dr.get("imu_noise_std_rad", {}).get("low", 0.001),
            imu_noise_high=dr.get("imu_noise_std_rad", {}).get("high", 0.020),
            latency_ms_low=dr.get("imu_latency_ms", {}).get("low", 5.0),
            latency_ms_high=dr.get("imu_latency_ms", {}).get("high", 15.0),
            slip_low=dr.get("wheel_slip_coeff", {}).get("low", 0.0),
            slip_high=dr.get("wheel_slip_coeff", {}).get("high", 0.05),
            action_delay_low=dr.get("action_delay_steps", {}).get("low", 0),
            action_delay_high=dr.get("action_delay_steps", {}).get("high", 2),
        )


class DomainRandomizationWrapper(gym.Wrapper):
    """
    Wraps PendulumBalanceEnv and re-samples physics parameters on each reset.

    Usage:
        base_env = PendulumBalanceEnv()
        env = DomainRandomizationWrapper(base_env, DRConfig())
        obs, info = env.reset()   # physics randomized here
    """

    def __init__(
        self,
        env: PendulumBalanceEnv,
        config: Optional[DRConfig] = None,
        enabled: bool = True,
    ) -> None:
        super().__init__(env)
        self.dr_config = config or DRConfig()
        self.enabled = enabled
        self._current_params: dict = {}

    def reset(self, *, seed=None, options=None):
        if self.enabled:
            self._randomize_physics()
        return self.env.reset(seed=seed, options=options)

    def _randomize_physics(self) -> None:
        """Sample and apply a new set of physics parameters."""
        rng = self.env.np_random if hasattr(self.env, "np_random") else np.random

        cfg = self.dr_config

        mass = float(rng.uniform(cfg.mass_low, cfg.mass_high))
        friction = float(rng.uniform(cfg.friction_low, cfg.friction_high))
        imu_noise = float(rng.uniform(cfg.imu_noise_low, cfg.imu_noise_high))
        latency_ms = float(rng.uniform(cfg.latency_ms_low, cfg.latency_ms_high))
        slip = float(rng.uniform(cfg.slip_low, cfg.slip_high))
        action_delay = int(rng.integers(cfg.action_delay_low, cfg.action_delay_high + 1))

        # Convert latency from ms to control steps (round to nearest int)
        latency_steps = int(round(latency_ms / (DT_MS := _DT * 1000)))

        # Mutate the underlying env's physics parameters directly
        unwrapped: PendulumBalanceEnv = self.env.unwrapped  # type: ignore[assignment]
        unwrapped.body_mass = mass
        unwrapped.motor_friction = friction
        unwrapped.imu_noise_std = imu_noise
        unwrapped.imu_latency_steps = latency_steps
        unwrapped.wheel_slip_coeff = slip
        unwrapped.action_delay_steps = action_delay

        # Update derived inertia
        from .pendulum_env import _BODY_HEIGHT
        unwrapped._body_inertia = mass * _BODY_HEIGHT ** 2 / 3.0

        # Resize delay buffers to match new latency values
        from collections import deque
        unwrapped._obs_buffer = deque(maxlen=max(1, latency_steps + 1))
        unwrapped._action_buffer = deque(maxlen=max(1, action_delay + 1))

        self._current_params = {
            "mass_kg": mass,
            "friction_nm": friction,
            "imu_noise_std": imu_noise,
            "imu_latency_steps": latency_steps,
            "wheel_slip_coeff": slip,
            "action_delay_steps": action_delay,
        }

    @property
    def current_dr_params(self) -> dict:
        """Returns the physics parameters sampled for the current episode."""
        return self._current_params.copy()


def make_env(
    rank: int,
    dr_config: Optional[DRConfig] = None,
    use_dr: bool = True,
    seed: int = 0,
) -> gym.Env:
    """
    Factory for creating a (optionally DR-wrapped) env suitable for SB3
    VecEnv construction.

    Args:
        rank:      Index of this parallel environment (for seeding).
        dr_config: Domain randomization configuration.
        use_dr:    Whether to apply domain randomization.
        seed:      Base random seed.

    Returns:
        A Gymnasium environment instance.
    """
    def _init():
        env = PendulumBalanceEnv()
        if use_dr:
            env = DomainRandomizationWrapper(env, dr_config)
        env.reset(seed=seed + rank)
        return env

    return _init
