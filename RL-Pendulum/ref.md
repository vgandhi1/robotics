# RL-Pendulum: Sim-to-Real Reinforcement Learning for Dynamic Balance

> An AI-first approach to physical control systems — replacing hand-tuned PID controllers with a PyTorch-trained PPO policy deployed on an ESP32 microcontroller at 100 Hz inference.

---

## Table of Contents

1. [Project Overview](#overview)
2. [Hardware Architecture](#hardware)
3. [Software Architecture](#software)
4. [AI Pipeline](#ai-pipeline)
5. [Environment & State Space](#environment)
6. [Reward Engineering](#reward)
7. [Domain Randomization](#domain-randomization)
8. [Sim-to-Real Transfer](#sim-to-real)
9. [Edge Deployment](#edge-deployment)
10. [Project Structure](#structure)
11. [Setup & Installation](#setup)
12. [Training Guide](#training)
13. [Evaluation](#evaluation)
14. [Deployment Guide](#deployment)
15. [Troubleshooting](#troubleshooting)
16. [References](#references)

---

## Overview

Traditional robotics relies on manually tuned mathematical controllers (PID, LQR) to balance an inverted pendulum. This project replaces that with a **Proximal Policy Optimization (PPO)** agent that:

1. Learns to balance through **millions of simulated interactions** in a custom Gymnasium environment.
2. Overcomes the **sim-to-real gap** via domain randomization — injecting calibrated noise so the policy generalizes to physical hardware.
3. **Deploys zero-shot** onto an ESP32 microcontroller using quantized INT8 ONNX / TensorFlow Lite Micro inference at under 10 ms per step.

### Why RL over PID?

| Property              | PID Controller      | RL Policy (PPO)              |
|-----------------------|---------------------|------------------------------|
| Tuning effort         | Manual, tedious     | Automated via reward signal  |
| Non-linear dynamics   | Approximate only    | Natively handles non-linearity |
| Noise robustness      | Requires re-tuning  | Trained with noise injection |
| Adaptation            | Static gains        | Robust across parameter range |
| Deployment size       | Trivial             | ~50 KB quantized model       |

---

## Hardware Architecture

### Bill of Materials

| Component               | Model/Spec             | Role                        | Qty |
|-------------------------|------------------------|-----------------------------|-----|
| Microcontroller         | ESP32-WROOM-32 (240MHz) | Policy inference + PWM     | 1   |
| IMU                     | MPU-6050 (I2C)         | Pitch angle + angular rate  | 1   |
| DC Motors               | N20 Gear Motor 100RPM  | Drive wheels (left/right)   | 2   |
| Motor Driver            | L298N or DRV8833       | H-bridge PWM amplifier      | 1   |
| Wheel Encoders          | Hall-effect, 20 CPR    | Wheel speed feedback        | 2   |
| Power Supply            | 3S LiPo 11.1V 1000mAh  | Robot power                 | 1   |
| Logic Level Converter   | 3.3V ↔ 5V Bi-dir       | IMU / ESP32 interface       | 1   |
| Chassis                 | Custom 3D-printed      | Body structure              | 1   |

### Wiring Diagram (ASCII)

```
                 ┌─────────────────────────────────┐
                 │         ESP32-WROOM-32           │
                 │                                  │
  MPU-6050       │  GPIO21 (SDA) ──── SDA           │
  ┌─────────┐    │  GPIO22 (SCL) ──── SCL           │
  │ IMU     │────│  3.3V ──────────── VCC           │
  │ pitch   │    │  GND ───────────── GND           │
  │ rate    │    │                                  │
  └─────────┘    │  GPIO25 (PWM_L) ──┐              │
                 │  GPIO26 (DIR_L) ──┤─── L298N ───── Motor L
                 │  GPIO27 (PWM_R) ──┤              │
                 │  GPIO14 (DIR_R) ──┘─── L298N ───── Motor R
                 │                                  │
  Encoders       │  GPIO34 (ENC_LA) ── Encoder L A  │
  ┌─────────┐    │  GPIO35 (ENC_LB) ── Encoder L B  │
  │  Hall   │────│  GPIO32 (ENC_RA) ── Encoder R A  │
  │ sensors │    │  GPIO33 (ENC_RB) ── Encoder R B  │
  └─────────┘    │                                  │
                 └─────────────────────────────────┘
                              │
                         3S LiPo
                         11.1V / Buck → 5V for logic
```

### Physical Robot Dimensions

```
Side View (balanced upright):

        ●  ← Center of Mass (top-heavy, unstable)
        │
      ┌─┴─┐
      │ESP│  ← Controller board  (~12cm height)
      │IMU│
      └─┬─┘
    ────┼────  ← Wheel axle
   ◯    │    ◯
  Left       Right
  Wheel       Wheel
  (40mm dia)
```

---

## Software Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        TRAINING (Host PC / GPU)                  │
│                                                                  │
│  ┌─────────────┐    ┌──────────────────┐    ┌────────────────┐  │
│  │  MuJoCo /   │    │  PPO Agent       │    │  Domain        │  │
│  │  Custom     │───▶│  (SB3 / PyTorch) │◀───│  Randomization │  │
│  │  Gym Env    │    │                  │    │  Wrapper       │  │
│  └─────────────┘    └────────┬─────────┘    └────────────────┘  │
│                              │                                   │
│                     ┌────────▼─────────┐                        │
│                     │  Trained Policy  │                        │
│                     │  (PyTorch .pt)   │                        │
│                     └────────┬─────────┘                        │
└──────────────────────────────┼───────────────────────────────────┘
                               │
              ┌────────────────▼────────────────┐
              │         EXPORT PIPELINE          │
              │                                  │
              │  PyTorch → ONNX → TFLite INT8    │
              │  (quantization + layer fusion)   │
              └────────────────┬────────────────┘
                               │
              ┌────────────────▼────────────────┐
              │    EDGE INFERENCE (ESP32)        │
              │                                  │
              │  IMU → State Vector → NN (INT8)  │
              │       → Motor PWM Outputs        │
              │   Loop time: < 10ms @ 100Hz      │
              └──────────────────────────────────┘
```

---

## AI Pipeline

### Phase 1 — Digital Twin & Environment Setup

- Build a lightweight **URDF** model capturing the robot's inertia, wheel geometry, and motor dynamics.
- Wrap physics simulation into a **Gymnasium-compatible environment**.
- Ensure simulated IMU outputs match the **MPU-6050 frequency (200 Hz)** and noise characteristics (Allan deviation-matched Gaussian).

**Key design decision:** The observation vector is carefully matched to the data available on the physical ESP32 at inference time — no simulated quantities that cannot be measured on hardware.

### Phase 2 — Reward Shaping & Baseline Training

Train a PPO agent for **5M steps** with the reward:

```
R(t) = α·(1 − θ²/θ_max²)         # upright angle reward
     − β·||a||²                   # motor effort penalty (anti-jitter)
     − γ·|ẋ|                      # position drift penalty
     + δ·alive_bonus              # +1 per step the robot stays balanced
```

Parameters: `α=1.0, β=0.01, θ_max=0.5 rad, γ=0.1, δ=0.1`

The reward landscape encourages:
- **Stability:** Minimizing pole tilt.
- **Efficiency:** Penalizing motor energy (prevents violent oscillation).
- **Positional integrity:** Preventing the robot from drifting off the table.

### Phase 3 — Domain Randomization (DR)

At each episode reset, sample physics parameters from uniform distributions:

| Parameter            | Nominal  | Randomization Range |
|----------------------|----------|---------------------|
| Body mass            | 0.5 kg   | ±20% → [0.4, 0.6]  |
| Motor friction (Nm)  | 0.001    | ±50% → [0.0005, 0.0015] |
| IMU noise σ (rad)    | 0.005    | Uniform [0.001, 0.02] |
| IMU latency (ms)     | 0        | Uniform [5, 15]     |
| Wheel slip coeff.    | 0.0      | Uniform [0.0, 0.05] |
| Action delay (steps) | 0        | Uniform [0, 2]      |

**Retrain for 10M steps** with DR enabled.

### Phase 4 — Edge Deployment

```
PyTorch FP32 (policy MLP: 4 → 64 → 64 → 1)
    │
    ▼  torch.onnx.export()
ONNX FP32 (verified via onnxruntime)
    │
    ▼  onnxruntime quantize_dynamic() → INT8
Quantized ONNX (4× smaller, ~2× faster on ARM)
    │
    ▼  tf2onnx + tflite converter
TFLite FlatBuffer (.tflite)
    │
    ▼  xxd → C header array
ESP32 firmware embeds model weights as const uint8_t[]
```

---

## Environment

### State Space

```
observation = [
    pitch_angle       (rad)    — from MPU-6050 complementary filter
    pitch_rate        (rad/s)  — gyroscope reading
    left_wheel_speed  (rad/s)  — from hall encoder, filtered
    right_wheel_speed (rad/s)  — from hall encoder, filtered
]
```

All values are normalized to `[-1, 1]` using known physical limits before being fed to the neural network.

### Action Space

```
action = [target_motor_voltage]  ∈ [-1.0, +1.0]
```

A single scalar controls both motors symmetrically (forward/backward balance). Differential steering for turning is a planned extension.

### Episode Termination

The episode ends (failure) when:
- `|pitch_angle| > 0.5 rad (≈ 28.6°)` — robot has fallen
- `|x_position| > 2.0 m` — robot has drifted off course
- `t > 1000 steps` (10 seconds at 100 Hz) — success

---

## Reward Engineering

The full shaped reward at each timestep:

```python
# Upright angle reward (quadratic cost → 1.0 when perfectly upright)
r_upright  = 1.0 - (pitch_angle / PITCH_LIMIT) ** 2

# Motor effort penalty (discourages jitter and high power draw)
r_effort   = -0.01 * action ** 2

# Position drift penalty (keeps robot roughly centered)
r_position = -0.1 * abs(x_position) / X_LIMIT

# Alive bonus (encourages long balanced episodes)
r_alive    = 0.1

total_reward = r_upright + r_effort + r_position + r_alive
```

Maximum achievable reward per episode (1000 steps):
`R_max ≈ 1000 × (1.0 + 0.1) = 1100` (no drift, perfectly still, zero effort)

---

## Domain Randomization

Domain Randomization (DR) is the primary technique for closing the sim-to-real gap without access to a precise physical model. By training on a *distribution* of environments, the policy learns to generalize.

```
Episode start:
    sample mass    ~ Uniform(0.4, 0.6)      kg
    sample friction ~ Uniform(5e-4, 1.5e-3) Nm
    sample IMU_std ~ Uniform(1e-3, 2e-2)    rad
    sample latency ~ Uniform(5, 15)         ms
    
Each step:
    obs_noisy = obs + N(0, IMU_std²)        (sensor noise)
    obs_delayed = buffer[t - latency_steps] (observation delay)
    action_delayed = action_buffer[t - action_delay] (actuator lag)
```

---

## Sim-to-Real Transfer

After DR training, **zero-shot transfer** is tested on the physical robot:

1. Flash the quantized TFLite model to ESP32.
2. Run the 100 Hz inference loop.
3. Log IMU + motor data via Serial/UART.
4. If balance fails: identify the dominant failure mode and tune DR ranges.

**Common failure modes and fixes:**

| Symptom                        | Cause                  | Fix                                |
|--------------------------------|------------------------|------------------------------------|
| Oscillates at high frequency   | Too low friction in DR | Increase `friction_max`            |
| Slowly drifts and falls        | Gyro drift / bias      | Add bias estimation to DR          |
| Overcorrects, crashes backward | Actuator lag mismatch  | Increase `action_delay_max`        |
| Works briefly, then fails      | Thermal motor lag      | Add motor thermal model to sim     |

---

## Edge Deployment

### Model Size Budget (ESP32 constraints)

| Format             | Size    | Latency (ESP32 @ 240MHz) |
|--------------------|---------|--------------------------|
| PyTorch FP32       | ~200 KB | Not runnable              |
| ONNX FP32          | ~150 KB | Not runnable              |
| ONNX INT8          | ~40 KB  | ~8–12 ms                 |
| TFLite INT8        | ~32 KB  | ~6–9 ms ✓                |

### Inference Loop (100 Hz = 10ms budget)

```
┌─────────────────────────────────────────────────────┐
│  ESP32 Inference Loop (10ms tick)                   │
│                                                     │
│  1. Read MPU-6050 via I2C          (~1.0ms)         │
│  2. Complementary filter (pitch)   (~0.1ms)         │
│  3. Read encoder counts            (~0.2ms)         │
│  4. Normalize state vector         (~0.1ms)         │
│  5. TFLite Invoke() [INT8 MLP]    (~6–9ms)          │
│  6. Clip + scale action to PWM     (~0.1ms)         │
│  7. Write PWM to L298N             (~0.1ms)         │
│  Total:                           < 10ms ✓          │
└─────────────────────────────────────────────────────┘
```

---

## Project Structure

```
RL-Pendulum/
├── ref.md                       # This document
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
├── configs/
│   └── ppo_config.yaml          # Hyperparameters
├── envs/
│   ├── __init__.py
│   ├── pendulum_env.py          # Custom Gymnasium environment
│   └── domain_randomization.py # DR wrapper
├── training/
│   ├── __init__.py
│   ├── train.py                 # Main PPO training script
│   └── callbacks.py             # SB3 training callbacks
├── export/
│   ├── __init__.py
│   ├── export_onnx.py           # PyTorch → ONNX
│   └── quantize.py              # FP32 → INT8 quantization
├── evaluation/
│   ├── __init__.py
│   └── evaluate.py              # Policy evaluation + plots
├── firmware/
│   ├── README.md                # Flashing guide
│   └── inference_loop/
│       ├── inference_loop.ino   # Arduino sketch
│       ├── imu_driver.h         # MPU-6050 driver
│       └── motor_driver.h       # L298N PWM driver
├── urdf/
│   └── pendulum_robot.urdf      # Robot description
├── scripts/
│   ├── run_training.sh          # Full training pipeline
│   └── export_model.sh          # Export + quantize
└── tests/
    ├── test_env.py
    ├── test_dr.py
    └── test_export.py
```

---

## Setup

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (optional, for GPU training)
- Arduino IDE 2.x or PlatformIO (for firmware)
- MuJoCo 3.x (optional, for advanced simulation)

### Python Environment

```bash
git clone https://github.com/yourusername/RL-Pendulum.git
cd RL-Pendulum
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Verify Environment

```bash
python -c "from envs import PendulumEnv; e = PendulumEnv(); print(e.reset())"
```

---

## Training

### Quick Start (5M steps, no DR)

```bash
python training/train.py --config configs/ppo_config.yaml --total-timesteps 5000000
```

### Full Pipeline with Domain Randomization

```bash
bash scripts/run_training.sh
```

### Monitor Training (TensorBoard)

```bash
tensorboard --logdir logs/
```

### Expected Training Curve

```
Reward
1100 |                                         ╭─────────────
     |                               ╭─────────╯
 600 |                    ╭──────────╯
     |         ╭──────────╯
 100 |─────────╯
   0 +──────────────────────────────────────────────────────→
     0        1M       2M       3M       4M      5M   Steps
```

---

## Evaluation

```bash
python evaluation/evaluate.py --model logs/best_model.zip --episodes 50
```

Outputs:
- **Success rate** (episodes balanced for full 10s)
- **Mean episode reward**
- **Balance time distribution** (histogram)
- **Phase portrait** (pitch angle vs. angular velocity)

---

## Deployment

### 1. Export and Quantize

```bash
bash scripts/export_model.sh --model logs/best_model.zip
# Outputs: export/model.onnx, export/model_int8.onnx, export/model.tflite
```

### 2. Generate C Header

```bash
xxd -i export/model.tflite > firmware/inference_loop/rl_policy_data.h
```

### 3. Flash ESP32

Open `firmware/inference_loop/inference_loop.ino` in Arduino IDE.
Install required libraries:
- `TensorFlowLite_ESP32` (v0.9.0+)
- `MPU6050` by Electronic Cats

Select board: `ESP32 Dev Module` → Flash (115200 baud).

### 4. Monitor

```bash
screen /dev/ttyUSB0 115200
# Expected output every 10ms:
# pitch=0.012 rate=-0.003 lw=2.341 rw=2.338 action=0.021 loop_ms=7.4
```

---

## Troubleshooting

| Issue                            | Solution                                                      |
|----------------------------------|---------------------------------------------------------------|
| `gymnasium` env fails to reset  | Check MuJoCo installation: `pip install mujoco`              |
| Policy NaN during training       | Reduce learning rate; clip reward range in config            |
| ONNX export shape mismatch       | Verify `obs_dim=4` in export script matches training config  |
| ESP32 runs out of flash          | Enable PSRAM; reduce hidden layer size to 32×32              |
| Robot oscillates badly           | Increase DR `friction_max`; reduce `kp` in action scaling    |
| IMU values drifting              | Implement gyro bias estimation (Madgwick/Mahony filter)       |

---

## References

1. Schulman et al. (2017) — [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
2. Tobin et al. (2017) — [Domain Randomization for Transferring Deep Neural Networks](https://arxiv.org/abs/1703.06907)
3. Tan et al. (2018) — [Sim-to-Real: Learning Agile Locomotion For Quadruped Robots](https://arxiv.org/abs/1804.10332)
4. Antonova et al. (2017) — [Reinforcement Learning for Pivoting Task](https://arxiv.org/abs/1703.00472)
5. [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
6. [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers)
7. [ESP32 TFLite Library](https://github.com/tanakamasayuki/tfmicro)
