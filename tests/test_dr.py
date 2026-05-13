"""
Unit tests for Domain Randomization wrapper.

Tests:
  - DR wrapper applies without errors
  - Physics parameters change between episodes
  - DR does not change observation/action space shapes
  - make_env factory creates valid envs
  - DRConfig.from_dict correctly parses YAML config
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.pendulum_env import PendulumBalanceEnv
from envs.domain_randomization import (
    DomainRandomizationWrapper,
    DRConfig,
    make_env,
)


@pytest.fixture
def wrapped_env():
    base = PendulumBalanceEnv()
    env = DomainRandomizationWrapper(base, DRConfig())
    env.reset(seed=0)
    yield env
    env.close()


class TestDRConfig:
    def test_default_config_valid(self):
        cfg = DRConfig()
        assert cfg.mass_low < cfg.mass_high
        assert cfg.friction_low < cfg.friction_high
        assert cfg.imu_noise_low < cfg.imu_noise_high
        assert cfg.latency_ms_low < cfg.latency_ms_high
        assert cfg.slip_low <= cfg.slip_high
        assert cfg.action_delay_low <= cfg.action_delay_high

    def test_from_dict_parses_correctly(self):
        d = {
            "domain_randomization": {
                "mass_kg":            {"low": 0.45, "high": 0.55},
                "friction_nm":        {"low": 0.001, "high": 0.002},
                "imu_noise_std_rad":  {"low": 0.005, "high": 0.015},
                "imu_latency_ms":     {"low": 6.0,   "high": 12.0},
                "wheel_slip_coeff":   {"low": 0.01,  "high": 0.04},
                "action_delay_steps": {"low": 0,     "high": 1},
            }
        }
        cfg = DRConfig.from_dict(d)
        assert cfg.mass_low == pytest.approx(0.45)
        assert cfg.mass_high == pytest.approx(0.55)
        assert cfg.imu_noise_low == pytest.approx(0.005)
        assert cfg.latency_ms_high == pytest.approx(12.0)


class TestDRWrapper:
    def test_spaces_preserved(self, wrapped_env):
        """DR wrapper must not change observation/action space."""
        assert wrapped_env.observation_space.shape == (4,)
        assert wrapped_env.action_space.shape == (1,)

    def test_reset_returns_valid_obs(self, wrapped_env):
        obs, info = wrapped_env.reset(seed=1)
        assert obs.shape == (4,)
        assert obs.dtype == np.float32
        assert np.all(obs >= -1.0) and np.all(obs <= 1.0)

    def test_params_change_between_resets(self):
        """DR should sample different physics params on each reset."""
        base = PendulumBalanceEnv()
        env = DomainRandomizationWrapper(base, DRConfig())

        env.reset(seed=0)
        params1 = env.current_dr_params.copy()

        env.reset(seed=1)
        params2 = env.current_dr_params.copy()

        # At least one parameter should differ across resets with different seeds
        different = any(
            abs(params1[k] - params2[k]) > 1e-8 for k in params1
        )
        assert different, "DR params should differ between resets"
        env.close()

    def test_params_within_bounds(self):
        """Sampled DR params must respect configured ranges."""
        cfg = DRConfig()
        base = PendulumBalanceEnv()
        env = DomainRandomizationWrapper(base, cfg)

        for seed in range(20):
            env.reset(seed=seed)
            p = env.current_dr_params
            assert cfg.mass_low <= p["mass_kg"] <= cfg.mass_high
            assert cfg.friction_low <= p["friction_nm"] <= cfg.friction_high
            assert cfg.imu_noise_low <= p["imu_noise_std"] <= cfg.imu_noise_high
            assert cfg.action_delay_low <= p["action_delay_steps"] <= cfg.action_delay_high

        env.close()

    def test_disabled_dr_keeps_nominal_params(self):
        """When DR is disabled, physics params should remain at defaults."""
        base = PendulumBalanceEnv()
        env = DomainRandomizationWrapper(base, DRConfig(), enabled=False)
        env.reset(seed=0)
        # With DR disabled, current_dr_params should be empty
        assert env.current_dr_params == {}
        env.close()

    def test_step_works_after_dr_reset(self, wrapped_env):
        obs, _ = wrapped_env.reset(seed=5)
        for _ in range(10):
            obs, reward, term, trunc, info = wrapped_env.step(np.array([0.1]))
            assert np.isfinite(reward)
            assert obs.shape == (4,)
            if term or trunc:
                break


class TestMakeEnv:
    def test_make_env_no_dr(self):
        env_fn = make_env(rank=0, use_dr=False, seed=0)
        env = env_fn()
        obs, _ = env.reset()
        assert obs.shape == (4,)
        env.close()

    def test_make_env_with_dr(self):
        cfg = DRConfig()
        env_fn = make_env(rank=0, dr_config=cfg, use_dr=True, seed=42)
        env = env_fn()
        obs, _ = env.reset()
        assert obs.shape == (4,)
        env.close()

    def test_make_env_different_ranks_different_seeds(self):
        """Different rank values should produce different initial states."""
        env_fn_0 = make_env(rank=0, use_dr=False, seed=0)
        env_fn_1 = make_env(rank=1, use_dr=False, seed=0)
        e0 = env_fn_0()
        e1 = env_fn_1()
        obs0, _ = e0.reset()
        obs1, _ = e1.reset()
        # Different seeds → different initial perturbations
        assert not np.allclose(obs0, obs1)
        e0.close()
        e1.close()
