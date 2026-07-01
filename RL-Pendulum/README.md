<div align="center">

# ⚖️ RL-Pendulum

### Sim-to-Real Reinforcement Learning for Dynamic Balance

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Stable Baselines3](https://img.shields.io/badge/SB3-PPO-10B981?style=flat-square)](https://stable-baselines3.readthedocs.io)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29%2B-0077B5?style=flat-square)](https://gymnasium.farama.org)
[![ESP32](https://img.shields.io/badge/ESP32-TFLite%20Micro-E7352C?style=flat-square&logo=espressif&logoColor=white)](https://www.espressif.com)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

<br/>

**An AI-first approach to physical control** — replacing hand-tuned PID controllers with a PyTorch PPO policy that learns to balance through 10M simulated interactions, then deploys zero-shot onto a $5 ESP32 microcontroller at 100 Hz.

<br/>

```
      Simulation (Host GPU)              Edge Inference (ESP32)
  ┌────────────────────────┐         ┌────────────────────────┐
  │  MuJoCo / Custom Gym   │         │  MPU-6050 IMU          │
  │  ┌──────────────────┐  │  ONNX   │  ┌──────────────────┐  │
  │  │   PPO Agent      │──┼─INT8──▶│  │  TFLite MLP      │  │
  │  │  (PyTorch MLP)   │  │  ───▶  │  │  4→64→64→1       │  │
  │  └──────────────────┘  │        │  └────────┬─────────┘  │
  │  Domain Randomization  │        │  PWM → L298N → Motors  │
  └────────────────────────┘        └────────────────────────┘
         Training                        ~7ms inference
```

</div>

---

## 🎯 What & Why

Traditional inverted pendulum control uses hand-tuned **PID** or **LQR** controllers — brittle math that breaks when hardware ages, floors get slippery, or batteries drain. This project takes a fundamentally different approach:

| | PID / LQR | RL Policy (This Project) |
|---|---|---|
| **Tuning** | Manual, expert-dependent | Automated via reward signal |
| **Non-linear dynamics** | Linearized approximation | Natively handles non-linearity |
| **Noise robustness** | Re-tune for each condition | Domain-randomized for all conditions |
| **Adaptation** | Static gains | Robust policy over parameter range |
| **Deployment size** | Trivial | ~32 KB quantized TFLite |

---

## 🏗️ Architecture at a Glance

```
Phase 1 ── Digital Twin ──────────────────────────────────────────────────────
           Custom Gymnasium environment   │  pendulum_env.py
           Physics: Euler-integrated      │  2-wheel pendulum dynamics
           IMU model: complementary filter│  matched to MPU-6050 specs

Phase 2 ── Baseline Training (5M steps) ─────────────────────────────────────
           Algorithm: PPO (Stable Baselines3)
           Policy:    MLP  4 → [64, 64] → 1   (tanh activations)
           Reward:    R = α(1-θ²) − β·a² − γ|x| + δ

Phase 3 ── Domain Randomization (10M steps) ─────────────────────────────────
           Randomizes body mass ±20%, motor friction ±50%,
           IMU noise σ ∈ [1,20] mrad, observation latency 5–15 ms,
           wheel slip 0–5%, action delay 0–2 steps

Phase 4 ── Export Pipeline ───────────────────────────────────────────────────
           PyTorch FP32  ──▶  ONNX FP32  ──▶  ONNX INT8  ──▶  TFLite INT8
                ~200 KB          ~150 KB         ~40 KB          ~32 KB ✓

Phase 5 ── Edge Inference (100 Hz) ──────────────────────────────────────────
           ESP32 @ 240 MHz: IMU → normalize → TFLite invoke → PWM
           Loop budget: 10 ms │ Actual: ~7–9 ms ✓
```

---

## 📁 Project Structure

```
RL-Pendulum/
├── 📄 README.md                     ← You are here
├── 📄 TECHNICAL_GUIDE.md            ← Deep-dive: math, algorithms, hardware
├── 📄 ref.md                        ← Original project brief + architecture
│
├── ⚙️  configs/
│   └── ppo_config.yaml              ← All hyperparameters, DR ranges, env limits
│
├── 🏋️  envs/
│   ├── pendulum_env.py              ← Custom Gymnasium environment (physics sim)
│   └── domain_randomization.py     ← DR wrapper + DRConfig dataclass
│
├── 🧠  training/
│   ├── train.py                     ← Main PPO training script (CLI)
│   └── callbacks.py                 ← Eval, DR logging, progress callbacks
│
├── 📦  export/
│   ├── export_onnx.py               ← PyTorch actor → verified ONNX
│   └── quantize.py                  ← ONNX INT8 + TFLite + C header generator
│
├── 📊  evaluation/
│   └── evaluate.py                  ← Metrics + 4 diagnostic plots
│
├── 🔌  firmware/
│   ├── inference_loop/
│   │   ├── inference_loop.ino       ← Main Arduino sketch (100 Hz loop)
│   │   ├── imu_driver.h             ← MPU-6050 I2C + complementary filter
│   │   └── motor_driver.h           ← L298N LEDC PWM driver
│   └── README.md                    ← Flashing guide
│
├── 🤖  urdf/
│   └── pendulum_robot.urdf          ← Robot description for sim / ROS
│
├── 🛠️  scripts/
│   ├── run_training.sh              ← Phase 2 → Phase 3 pipeline
│   └── export_model.sh             ← ONNX → TFLite → C header
│
└── 🧪  tests/
    ├── test_env.py                  ← 22 environment unit tests
    ├── test_dr.py                   ← 11 domain randomization tests
    └── test_export.py               ← 10 ONNX / quantization tests
```

---

## 🤖 Hardware

### Bill of Materials (~$45 total)

| Component | Model | Role | ~Cost |
|-----------|-------|------|-------|
| Microcontroller | **ESP32-WROOM-32** (240 MHz) | Policy inference + PWM | $5 |
| IMU | **MPU-6050** (I2C, 200 Hz) | Pitch angle + rate | $2 |
| Motors | N20 Gear Motor 100 RPM × 2 | Drive wheels | $8 |
| Motor Driver | **L298N** or DRV8833 | H-bridge PWM | $3 |
| Encoders | Hall-effect 20 CPR × 2 | Wheel speed feedback | $6 |
| Battery | 3S LiPo 11.1V 1000 mAh | Robot power | $12 |
| Chassis | Custom 3D-printed | Body structure | $9 |

### Wiring (ESP32 Pin Map)

```
MPU-6050  ──┬── SDA → GPIO21
            └── SCL → GPIO22

L298N     ──┬── ENA (PWM Left)  → GPIO27   (LEDC ch0, 20 kHz)
            ├── IN1 (Dir Left A) → GPIO25
            ├── IN2 (Dir Left B) → GPIO26
            ├── ENB (PWM Right) → GPIO13   (LEDC ch1, 20 kHz)
            ├── IN3 (Dir Right A)→ GPIO14
            └── IN4 (Dir Right B)→ GPIO12

Encoders  ──┬── Left  A/B → GPIO34 / GPIO35
            └── Right A/B → GPIO32 / GPIO33
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+, Git
- Arduino IDE 2.x (for firmware flashing)
- CUDA GPU recommended (not required)

### 1 — Install

```bash
git clone https://github.com/vgandhi1/RL-Pendulum.git
cd RL-Pendulum

python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2 — Verify Environment

```bash
python3 -c "
from envs import PendulumBalanceEnv
env = PendulumBalanceEnv()
obs, _ = env.reset(seed=42)
print('Observation shape:', obs.shape)
print('Sample obs:', obs)
print('✓ Environment OK')
"
```

### 3 — Train

```bash
# Phase 2: Baseline (5M steps, ~30 min on GPU)
python3 training/train.py --config configs/ppo_config.yaml --no-dr

# Phase 3: With domain randomization (10M steps)
python3 training/train.py --config configs/ppo_config.yaml

# Full pipeline in one command
bash scripts/run_training.sh
```

Monitor live in TensorBoard:
```bash
tensorboard --logdir logs/tensorboard
```

### 4 — Evaluate

```bash
python3 evaluation/evaluate.py \
    --model logs/best_model.zip \
    --episodes 50

# Output: success rate, 4 diagnostic plots saved to logs/eval/
```

### 5 — Export for ESP32

```bash
bash scripts/export_model.sh --model logs/best_model.zip

# Generates:
#   export/model.onnx          (~150 KB, FP32)
#   export/model_int8.onnx     (~40 KB,  INT8)
#   export/model.tflite        (~32 KB,  INT8)
#   firmware/inference_loop/rl_policy_data.h   ← flash-ready C header
```

### 6 — Flash ESP32

```
Arduino IDE → Open firmware/inference_loop/inference_loop.ino
Board: ESP32 Dev Module  |  Port: /dev/ttyUSB0  |  Upload
```

Serial Monitor (115200 baud) shows live telemetry:
```
pitch_deg,pitch_rate_rads,lw_spd_rads,rw_spd_rads,action,loop_ms
0.412,0.031,0.234,0.238,0.0214,7
```

---

## 📐 State Space & Reward

### Observation Vector (4D, normalized to [−1, 1])

| Index | Signal | Source | Normalization |
|-------|--------|--------|--------------|
| 0 | `pitch_angle` (rad) | MPU-6050 complementary filter | ÷ 0.5 rad |
| 1 | `pitch_rate` (rad/s) | MPU-6050 gyroscope | ÷ 10 rad/s |
| 2 | `left_wheel_speed` (rad/s) | Hall encoder | ÷ 20 rad/s |
| 3 | `right_wheel_speed` (rad/s) | Hall encoder | ÷ 20 rad/s |

### Shaped Reward (per 10 ms step)

```
R(t) = 1.0 × (1 − θ²/θ²_max)     ← upright angle reward
     − 0.01 × a²                   ← motor effort penalty
     − 0.10 × |x| / x_max         ← position drift penalty
     + 0.10                        ← alive bonus

Max episode reward ≈ 1100  (1000 steps × 1.1)
```

---

## 🎲 Domain Randomization

Re-sampled every episode reset to close the sim-to-real gap:

| Parameter | Nominal | Randomization Range | Physical Rationale |
|-----------|---------|--------------------|--------------------|
| Body mass | 0.50 kg | ±20% → [0.40, 0.60] | Battery charge, part tolerance |
| Motor friction | 0.001 Nm | ±50% → [0.0005, 0.0015] | Brush wear, lubrication |
| IMU noise σ | 0 rad | [0.001, 0.020] | Temperature, vibration |
| IMU latency | 0 ms | [5, 15] ms | I2C bus congestion |
| Wheel slip | 0 | [0.0, 0.05] | Floor surface variation |
| Action delay | 0 steps | [0, 2] steps | Actuator lag |

---

## 📊 Expected Training Curve

```
Reward
 1100 ┤                                          ╭─────────
      │                               ╭──────────╯
  600 ┤                   ╭───────────╯
      │        ╭──────────╯          Phase 3 (DR on)
  100 ┤────────╯           
    0 └────────┬──────────┬──────────┬──────────┬────────→ Steps
               1M         3M         5M         8M
```

---

## 🧪 Tests

```bash
python3 -m pytest tests/ -v
# 43 passed in ~7s
```

| Test Module | Coverage |
|-------------|----------|
| `test_env.py` | Spaces, reset, step, reward, termination, physics |
| `test_dr.py` | Config parsing, param bounds, factory, seed isolation |
| `test_export.py` | ONNX inference, INT8 quantization, C header generation |

---

## 📚 Key References

| Paper | Relevance |
|-------|-----------|
| [PPO — Schulman et al. (2017)](https://arxiv.org/abs/1707.06347) | Core RL algorithm |
| [Domain Randomization — Tobin et al. (2017)](https://arxiv.org/abs/1703.06907) | Sim-to-real transfer technique |
| [Sim-to-Real Locomotion — Tan et al. (2018)](https://arxiv.org/abs/1804.10332) | DR for legged robots |

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built with PyTorch · Stable Baselines3 · Gymnasium · TensorFlow Lite Micro

</div>
