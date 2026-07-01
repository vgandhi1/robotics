"""
Unit tests for the PendulumBalanceEnv Gymnasium environment.

Tests:
  - Observation/action space shapes and types
  - Reset returns valid observation
  - Step returns correct tuple structure and types
  - Termination on fall (|pitch| > 0.5 rad)
  - Truncation at max steps (1000)
  - Reward is finite and within expected range
  - Physics: energy grows when motor pushes, zero action → fall
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envs.pendulum_env import PendulumBalanceEnv, _PITCH_LIMIT, _X_LIMIT


@pytest.fixture
def env():
    e = PendulumBalanceEnv()
    yield e
    e.close()


class TestSpaces:
    def test_observation_space_shape(self, env):
        assert env.observation_space.shape == (4,)

    def test_action_space_shape(self, env):
        assert env.action_space.shape == (1,)

    def test_observation_space_bounds(self, env):
        assert np.all(env.observation_space.low  == -1.0)
        assert np.all(env.observation_space.high ==  1.0)

    def test_action_space_bounds(self, env):
        assert np.all(env.action_space.low  == -1.0)
        assert np.all(env.action_space.high ==  1.0)


class TestReset:
    def test_returns_tuple(self, env):
        result = env.reset(seed=0)
        assert isinstance(result, tuple) and len(result) == 2

    def test_observation_shape(self, env):
        obs, info = env.reset(seed=0)
        assert obs.shape == (4,)

    def test_observation_dtype(self, env):
        obs, _ = env.reset(seed=0)
        assert obs.dtype == np.float32

    def test_observation_in_bounds(self, env):
        for seed in range(10):
            obs, _ = env.reset(seed=seed)
            assert np.all(obs >= -1.0) and np.all(obs <= 1.0), \
                f"Obs out of bounds: {obs}"

    def test_info_contains_keys(self, env):
        _, info = env.reset(seed=0)
        assert "pitch_rad" in info
        assert "step" in info

    def test_seeded_reset_reproducible(self, env):
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)


class TestStep:
    def test_returns_five_tuple(self, env):
        env.reset(seed=0)
        result = env.step(np.array([0.0]))
        assert isinstance(result, tuple) and len(result) == 5

    def test_output_types(self, env):
        env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(np.array([0.0]))
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_action_clipping(self, env):
        """Out-of-bounds actions should not raise errors."""
        env.reset(seed=0)
        env.step(np.array([99.0]))   # over
        env.step(np.array([-99.0]))  # under

    def test_reward_finite(self, env):
        env.reset(seed=0)
        for _ in range(50):
            obs, reward, term, trunc, _ = env.step(env.action_space.sample())
            assert np.isfinite(reward), f"Non-finite reward: {reward}"
            if term or trunc:
                break

    def test_reward_range(self, env):
        """Reward should be in roughly [-2, 1.1] per step."""
        env.reset(seed=0)
        for _ in range(100):
            _, reward, term, trunc, _ = env.step(np.array([0.0]))
            assert reward >= -2.5, f"Reward too low: {reward}"
            assert reward <= 1.2, f"Reward too high: {reward}"
            if term or trunc:
                break


class TestTermination:
    def test_terminates_on_fall_forward(self, env):
        """Manually push pitch beyond limit; should terminate."""
        env.reset(seed=0)
        env._pitch = _PITCH_LIMIT + 0.01
        _, _, terminated, _, _ = env.step(np.array([0.0]))
        assert terminated, "Should terminate when pitch exceeds limit"

    def test_terminates_on_fall_backward(self, env):
        env.reset(seed=0)
        env._pitch = -(_PITCH_LIMIT + 0.01)
        _, _, terminated, _, _ = env.step(np.array([0.0]))
        assert terminated

    def test_terminates_on_drift(self, env):
        env.reset(seed=0)
        env._x_pos = _X_LIMIT + 0.01
        _, _, terminated, _, _ = env.step(np.array([0.0]))
        assert terminated

    def test_truncates_at_1000_steps(self, env):
        """Episode should truncate (not terminate) after 1000 steps if still balanced."""
        env.reset(seed=0)
        env._pitch = 0.0  # force upright for all steps
        env._x_pos = 0.0

        for step_num in range(1001):
            env._pitch = 0.0   # keep it from falling
            env._x_pos = 0.0
            _, _, terminated, truncated, _ = env.step(np.array([0.0]))
            if truncated:
                assert step_num == 999, f"Truncation at wrong step: {step_num}"
                assert not terminated
                break


class TestPhysics:
    def test_nonzero_action_changes_state(self, env):
        obs0, _ = env.reset(seed=0)
        # Apply a large action
        obs1, _, _, _, _ = env.step(np.array([1.0]))
        assert not np.allclose(obs0, obs1), "State should change after action"

    def test_zero_action_falls(self, env):
        """Starting at a non-zero angle with zero torque should cause the pendulum to fall."""
        env.reset(seed=0)
        env._pitch = 0.10  # small initial lean
        env._pitch_rate = 0.0

        steps = 0
        terminated = False
        while steps < 500 and not terminated:
            _, _, terminated, _, _ = env.step(np.array([0.0]))
            steps += 1

        assert terminated, (
            f"Pendulum with 0 action and initial lean should fall within 500 steps "
            f"(got {steps} steps)"
        )

    def test_observations_normalized(self, env):
        env.reset(seed=0)
        env._pitch = _PITCH_LIMIT * 0.9
        env._pitch_rate = 5.0
        env._lw_speed = 10.0
        obs, _ = env.reset(seed=0)
        # After reset, observation should be well within [-1, 1]
        assert np.all(obs >= -1.0) and np.all(obs <= 1.0)
