"""
Two-wheeled self-balancing robot Gymnasium environment.

Models a differential-drive inverted pendulum robot (similar to a Segway):
  - The body is an inverted pendulum that must stay upright.
  - Two motorized wheels provide the only actuation.
  - State is matched exactly to what the ESP32 can measure at runtime.

Physics model:
  - Rigid-body approximation with linearized motor dynamics.
  - Wheel slip accounted for via a configurable coefficient.
  - IMU observation noise and latency configurable for domain randomization.

Coordinate convention:
  - x:     forward displacement of the wheel contact point (m)
  - theta: pitch angle from vertical (rad), positive = forward lean
  - dot notation denotes time derivative
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# ─── Physical constants (nominal values) ────────────────────────────────────────
_GRAVITY = 9.81              # m/s²
_DT = 0.01                   # 10 ms control period (100 Hz)

# Robot geometry / inertia (nominal)
_BODY_MASS = 0.5             # kg
_BODY_HEIGHT = 0.12          # m   (center of mass above wheel axle)
_WHEEL_RADIUS = 0.02         # m
_WHEEL_MASS = 0.05           # kg (each)
_BODY_INERTIA = (            # kg·m²  (solid cylinder approximation)
    _BODY_MASS * _BODY_HEIGHT ** 2 / 3.0
)

# Motor model
_MOTOR_TORQUE_CONST = 0.03   # Nm/V   (back-EMF / torque constant)
_MOTOR_FRICTION = 0.001      # Nm     (viscous damping)
_MOTOR_MAX_VOLTAGE = 6.0     # V      (3S LiPo × gear ratio effective)

# Observation normalization limits
_PITCH_LIMIT = 0.5           # rad    — also terminal condition
_PITCH_RATE_LIMIT = 10.0     # rad/s
_WHEEL_SPEED_LIMIT = 20.0    # rad/s
_X_LIMIT = 2.0               # m

# Reward coefficients (see ref.md)
_ALPHA = 1.0     # upright reward
_BETA = 0.01     # motor effort penalty
_GAMMA = 0.1     # position drift penalty
_DELTA = 0.1     # alive bonus


class PendulumBalanceEnv(gym.Env):
    """
    Gymnasium environment for a two-wheeled self-balancing robot.

    Observation space (4D, normalized to [-1, 1]):
        [pitch_angle (rad), pitch_rate (rad/s),
         left_wheel_speed (rad/s), right_wheel_speed (rad/s)]

    Action space (1D, [-1, 1]):
        [target_motor_voltage]  — applied symmetrically to both wheels

    Reward:
        R = alpha*(1 - (theta/theta_max)^2)
          - beta * a^2
          - gamma * |x| / x_max
          + delta
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        # Physics overrides (used by DomainRandomizationWrapper)
        body_mass: float = _BODY_MASS,
        motor_friction: float = _MOTOR_FRICTION,
        imu_noise_std: float = 0.0,
        imu_latency_steps: int = 0,
        wheel_slip_coeff: float = 0.0,
        action_delay_steps: int = 0,
    ) -> None:
        super().__init__()

        self.render_mode = render_mode

        # Physics parameters (may be overridden by DR wrapper)
        self.body_mass = body_mass
        self.motor_friction = motor_friction
        self.imu_noise_std = imu_noise_std
        self.imu_latency_steps = max(0, imu_latency_steps)
        self.wheel_slip_coeff = wheel_slip_coeff
        self.action_delay_steps = max(0, action_delay_steps)

        # Derived inertia
        self._body_inertia = body_mass * _BODY_HEIGHT ** 2 / 3.0

        # Spaces
        obs_high = np.ones(4, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-obs_high, high=obs_high, dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Internal state (raw, un-normalized)
        self._pitch: float = 0.0
        self._pitch_rate: float = 0.0
        self._x_pos: float = 0.0
        self._x_vel: float = 0.0
        self._lw_speed: float = 0.0   # left wheel angular velocity (rad/s)
        self._rw_speed: float = 0.0

        # Latency buffers
        self._obs_buffer: deque[np.ndarray] = deque(
            maxlen=max(1, self.imu_latency_steps + 1)
        )
        self._action_buffer: deque[np.ndarray] = deque(
            maxlen=max(1, self.action_delay_steps + 1)
        )

        self._step_count: int = 0
        self._np_random: np.random.Generator = np.random.default_rng()

        # Rendering
        self._viewer = None

    # ─── Gymnasium API ───────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        rng = self.np_random

        # Randomize initial state near upright (small perturbation)
        self._pitch = rng.uniform(-0.05, 0.05)
        self._pitch_rate = rng.uniform(-0.02, 0.02)
        self._x_pos = rng.uniform(-0.1, 0.1)
        self._x_vel = 0.0
        self._lw_speed = 0.0
        self._rw_speed = 0.0
        self._step_count = 0

        # Pre-fill latency buffers with initial observation
        raw_obs = self._get_raw_obs()
        for _ in range(max(1, self.imu_latency_steps + 1)):
            self._obs_buffer.append(raw_obs.copy())

        zero_action = np.zeros(1, dtype=np.float32)
        for _ in range(max(1, self.action_delay_steps + 1)):
            self._action_buffer.append(zero_action.copy())

        observation = self._get_delayed_obs()
        info = self._get_info()
        return observation, info

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Store action in delay buffer; apply delayed action to physics
        self._action_buffer.append(action.copy())
        effective_action = self._action_buffer[0]

        # Physics step
        self._physics_step(effective_action[0])
        self._step_count += 1

        # Observation with noise + latency
        raw_obs = self._get_raw_obs()
        self._obs_buffer.append(raw_obs)
        observation = self._get_delayed_obs()

        # Reward
        reward = self._compute_reward(action[0])

        # Termination / truncation
        terminated = self._is_terminated()
        truncated = self._step_count >= 1000

        info = self._get_info()
        return observation, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    # ─── Physics ─────────────────────────────────────────────────────────────────

    def _physics_step(self, voltage_normalized: float) -> None:
        """
        Euler integration of two-wheeled inverted pendulum dynamics.

        Equations of motion (simplified planar model):
          theta_ddot = (m*g*L*sin(theta) - tau_wheel) / I_eff
          x_ddot     = (tau_wheel / r_wheel) / (m + 2*m_wheel)

        Where:
          tau_wheel = motor torque computed from applied voltage
          I_eff     = body_inertia + m*L² (parallel axis theorem contribution)
        """
        voltage = voltage_normalized * _MOTOR_MAX_VOLTAGE

        # Motor torque model: tau = K_t * V - friction * omega_wheel
        avg_wheel_speed = 0.5 * (self._lw_speed + self._rw_speed)
        torque = (
            _MOTOR_TORQUE_CONST * voltage
            - self.motor_friction * avg_wheel_speed
        )

        # Wheel slip: reduce effective torque by slip coefficient
        effective_torque = torque * (1.0 - self.wheel_slip_coeff)

        # Inverted pendulum angular acceleration
        sin_theta = np.sin(self._pitch)
        I_eff = self._body_inertia + self.body_mass * _BODY_HEIGHT ** 2
        theta_ddot = (
            self.body_mass * _GRAVITY * _BODY_HEIGHT * sin_theta
            - effective_torque
        ) / I_eff

        # Cart / wheel linear acceleration
        total_mass = self.body_mass + 2.0 * _WHEEL_MASS
        x_ddot = effective_torque / (_WHEEL_RADIUS * total_mass)

        # Wheel angular acceleration
        wheel_ddot = x_ddot / _WHEEL_RADIUS

        # Euler integration
        self._pitch += self._pitch_rate * _DT
        self._pitch_rate += theta_ddot * _DT
        self._x_pos += self._x_vel * _DT
        self._x_vel += x_ddot * _DT
        self._lw_speed += wheel_ddot * _DT
        self._rw_speed += wheel_ddot * _DT

        # Clamp to prevent numerical blow-up
        self._pitch_rate = np.clip(self._pitch_rate, -50.0, 50.0)
        self._x_vel = np.clip(self._x_vel, -10.0, 10.0)

    # ─── Observations ────────────────────────────────────────────────────────────

    def _get_raw_obs(self) -> np.ndarray:
        """Raw (un-normalized, pre-noise) observation matching sensor outputs."""
        obs = np.array(
            [self._pitch, self._pitch_rate, self._lw_speed, self._rw_speed],
            dtype=np.float32,
        )
        if self.imu_noise_std > 0.0:
            noise = self.np_random.normal(0.0, self.imu_noise_std, size=4).astype(
                np.float32
            )
            obs += noise
        return obs

    def _get_delayed_obs(self) -> np.ndarray:
        """Return the delayed, noisy, normalized observation for the policy."""
        raw = self._obs_buffer[0]  # oldest = most delayed
        normalized = np.array(
            [
                raw[0] / _PITCH_LIMIT,
                raw[1] / _PITCH_RATE_LIMIT,
                raw[2] / _WHEEL_SPEED_LIMIT,
                raw[3] / _WHEEL_SPEED_LIMIT,
            ],
            dtype=np.float32,
        )
        return np.clip(normalized, -1.0, 1.0)

    # ─── Reward ──────────────────────────────────────────────────────────────────

    def _compute_reward(self, action_scalar: float) -> float:
        r_upright = _ALPHA * (1.0 - (self._pitch / _PITCH_LIMIT) ** 2)
        r_effort = -_BETA * action_scalar ** 2
        r_position = -_GAMMA * abs(self._x_pos) / _X_LIMIT
        r_alive = _DELTA
        return float(r_upright + r_effort + r_position + r_alive)

    # ─── Termination ─────────────────────────────────────────────────────────────

    def _is_terminated(self) -> bool:
        return (
            abs(self._pitch) > _PITCH_LIMIT
            or abs(self._x_pos) > _X_LIMIT
        )

    # ─── Info ────────────────────────────────────────────────────────────────────

    def _get_info(self) -> dict:
        return {
            "pitch_rad": float(self._pitch),
            "pitch_rate_rads": float(self._pitch_rate),
            "x_pos_m": float(self._x_pos),
            "step": self._step_count,
        }

    # ─── Rendering ───────────────────────────────────────────────────────────────

    def _render_human(self) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        if not hasattr(self, "_fig"):
            plt.ion()
            self._fig, self._ax = plt.subplots(1, 1, figsize=(4, 6))

        self._ax.clear()
        self._ax.set_xlim(-0.5, 0.5)
        self._ax.set_ylim(-0.05, 0.3)
        self._ax.set_aspect("equal")
        self._ax.set_title(f"Step {self._step_count}  θ={self._pitch:.3f} rad")

        # Wheels
        cx = self._x_pos
        self._ax.add_patch(plt.Circle((cx - 0.04, _WHEEL_RADIUS), _WHEEL_RADIUS, color="k"))
        self._ax.add_patch(plt.Circle((cx + 0.04, _WHEEL_RADIUS), _WHEEL_RADIUS, color="k"))

        # Pendulum body
        bx = cx + _BODY_HEIGHT * np.sin(self._pitch)
        by = _WHEEL_RADIUS * 2 + _BODY_HEIGHT * np.cos(self._pitch)
        self._ax.plot([cx, bx], [_WHEEL_RADIUS * 2, by], "b-", lw=4)
        self._ax.add_patch(plt.Circle((bx, by), 0.015, color="r"))

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def _render_rgb_array(self) -> np.ndarray:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.backends.backend_agg as agg
        except ImportError:
            return np.zeros((400, 300, 3), dtype=np.uint8)

        fig, ax = plt.subplots(1, 1, figsize=(3, 4))
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.05, 0.3)
        ax.set_aspect("equal")
        ax.set_title(f"θ={self._pitch:.3f} rad")

        cx = self._x_pos
        ax.add_patch(plt.Circle((cx - 0.04, _WHEEL_RADIUS), _WHEEL_RADIUS, color="k"))
        ax.add_patch(plt.Circle((cx + 0.04, _WHEEL_RADIUS), _WHEEL_RADIUS, color="k"))
        bx = cx + _BODY_HEIGHT * np.sin(self._pitch)
        by = _WHEEL_RADIUS * 2 + _BODY_HEIGHT * np.cos(self._pitch)
        ax.plot([cx, bx], [_WHEEL_RADIUS * 2, by], "b-", lw=4)
        ax.add_patch(plt.Circle((bx, by), 0.015, color="r"))

        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        image = np.asarray(buf)[:, :, :3]
        plt.close(fig)
        return image
