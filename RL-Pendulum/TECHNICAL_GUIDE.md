# RL-Pendulum — Comprehensive Technical Guide

> Deep-dive into the mathematics, algorithms, hardware design, and implementation decisions behind the sim-to-real RL pipeline.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Physics Model](#2-physics-model)
3. [Gymnasium Environment](#3-gymnasium-environment)
4. [Reward Function Design](#4-reward-function-design)
5. [PPO Algorithm](#5-ppo-algorithm)
6. [Policy Network Architecture](#6-policy-network-architecture)
7. [Domain Randomization](#7-domain-randomization)
8. [Sim-to-Real Transfer Analysis](#8-sim-to-real-transfer-analysis)
9. [ONNX Export Pipeline](#9-onnx-export-pipeline)
10. [INT8 Quantization](#10-int8-quantization)
11. [Edge Inference on ESP32](#11-edge-inference-on-esp32)
12. [IMU Signal Processing](#12-imu-signal-processing)
13. [Motor Control](#13-motor-control)
14. [Hyperparameter Guide](#14-hyperparameter-guide)
15. [Evaluation Methodology](#15-evaluation-methodology)
16. [Failure Modes & Debugging](#16-failure-modes--debugging)
17. [Extending the Project](#17-extending-the-project)

---

## 1. System Overview

### The Control Problem

An inverted pendulum on two wheels is a **canonically unstable** system. The equilibrium point (upright) is unstable: any small perturbation, if uncontrolled, causes exponential divergence. The linearized dynamics have a positive eigenvalue, meaning energy flows away from equilibrium.

```
Physical system state:
  x       — horizontal cart position (m)
  ẋ       — cart velocity (m/s)
  θ       — pole pitch angle from vertical (rad)
  θ̇       — pole angular velocity (rad/s)

Unstable equilibrium: x=0, ẋ=0, θ=0, θ̇=0
Time to fall from 1° perturbation ≈ 0.3 s (open-loop)
```

### Why Reinforcement Learning?

Classical controllers (PID, LQR, MPC) require:
- **PID**: Three gains tuned per operating condition; no model of the system.
- **LQR**: Linearized model; optimal only near the equilibrium.
- **MPC**: Full model + optimization at each step; computationally heavy for embedded.

An RL policy:
- Requires **no explicit dynamics model** at inference time.
- Handles **non-linear** regimes (large angles, high speeds).
- Generalizes across a **distribution** of hardware variations via domain randomization.
- Inference cost: a single MLP forward pass (~200 FLOPS) — orders of magnitude cheaper than MPC.

---

## 2. Physics Model

### 2.1 Equations of Motion

The two-wheeled balancing robot is modeled as a **planar rigid-body inverted pendulum** with motorized wheels. The simplified equations of motion (neglecting lateral dynamics and differential drive):

**Pole angular acceleration:**
```
θ̈ = (m·g·L·sin(θ) − τ_eff) / I_eff

where:
  m      = body mass (kg)
  g      = 9.81 m/s²
  L      = 0.12 m  (CoM height above wheel axle)
  τ_eff  = effective wheel torque (Nm)
  I_eff  = m·L² + I_body  (parallel axis theorem)
  I_body = m·L²/3  (slender rod approximation)
```

**Cart linear acceleration:**
```
ẍ = τ_eff / (r_w · M_total)

where:
  r_w      = 0.02 m  (wheel radius)
  M_total  = m + 2·m_w  (body + both wheels)
```

**Motor torque model (back-EMF):**
```
τ_wheel = K_t · V_applied − b_friction · ω_wheel

where:
  K_t          = 0.03 Nm/V  (motor torque constant)
  b_friction   = 0.001 Nm   (viscous friction coefficient)
  V_applied    = action · V_max  ∈ [−6, 6] V
  ω_wheel      = mean wheel angular velocity (rad/s)
```

**Wheel slip model:**
```
τ_eff = τ_wheel · (1 − k_slip)

where k_slip ∈ [0, 1) is the slip coefficient (randomized in DR)
```

### 2.2 Numerical Integration

Euler forward integration at 100 Hz (Δt = 10 ms):

```python
θ    ← θ    + θ̇ · Δt
θ̇   ← θ̇   + θ̈ · Δt
x    ← x    + ẋ · Δt
ẋ   ← ẋ   + ẍ · Δt
ω_w ← ω_w  + ω̈_w · Δt
```

**Why Euler and not RK4?** At 100 Hz the step size is small enough that Euler errors are negligible for a rigid pendulum. RK4 would require 4 physics evaluations per step (4× slower), with no meaningful accuracy gain for this system at this frequency.

### 2.3 Key Simplifications

The simulation omits:
- **Gyroscopic effects** from spinning wheels (small at low wheel speeds)
- **Flexible chassis deformation** (robot body is rigid)
- **Stiction / coulomb friction** (only viscous modeled; compensated by DR)
- **3D dynamics / lateral tipping** (planar model only)

These omissions are intentional — domain randomization compensates for modeling errors at the cost of training robustness rather than simulation fidelity.

---

## 3. Gymnasium Environment

### 3.1 Environment Class: `PendulumBalanceEnv`

The environment inherits from `gymnasium.Env` and implements the standard API:

```
reset(seed) → (obs, info)
step(action) → (obs, reward, terminated, truncated, info)
render()     → (frame | None)
close()
```

### 3.2 Observation Space

```python
observation_space = Box(low=-1.0, high=1.0, shape=(4,), dtype=float32)
```

Raw sensor values are **normalized** to `[−1, 1]` before exposure to the policy:

```
obs[0] = pitch_angle     / 0.5      ← θ / θ_max
obs[1] = pitch_rate      / 10.0     ← θ̇ / θ̇_max
obs[2] = left_wheel_spd  / 20.0     ← ωL / ω_max
obs[3] = right_wheel_spd / 20.0     ← ωR / ω_max
```

**Design rationale for normalization:**
- Neural networks train faster and more stably when inputs are O(1).
- The same normalization constants are hard-coded in the ESP32 firmware, ensuring exact sim-to-real numerical alignment.

### 3.3 Action Space

```python
action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=float32)
```

A single scalar commands both motors symmetrically. The policy outputs a value in `[−1, 1]`, which the firmware maps to motor voltage via:

```
V_motor = action × V_max  where V_max = 6.0 V
duty    = |action| × 900  (10-bit PWM, 900/1023 ≈ 88% max)
```

### 3.4 Episode Lifecycle

```
reset():
    θ  ~ Uniform(−0.05, 0.05) rad   (small random initial lean)
    θ̇  ~ Uniform(−0.02, 0.02) rad/s
    x  ~ Uniform(−0.10, 0.10) m
    ẋ  = 0
    ωL = ωR = 0

step(a):
    1. Apply action delay (pop from circular buffer)
    2. Physics step (Euler, 10 ms)
    3. Add IMU noise N(0, σ²) to [θ, θ̇, ωL, ωR]
    4. Observe delayed observation (pop from obs buffer)
    5. Compute reward
    6. Check termination: |θ| > 0.5 rad OR |x| > 2.0 m
    7. Check truncation: step >= 1000

terminated ← robot fell or drifted
truncated  ← survived 10 seconds (success)
```

### 3.5 Observation Latency Buffer

Physical IMU latency (I2C bus + DLPF) is modeled as a circular buffer:

```python
# Buffer size = max_latency_steps + 1
obs_buffer = deque(maxlen=latency_steps + 1)

# At each step:
obs_buffer.append(current_raw_obs)
delayed_obs = obs_buffer[0]           # oldest = most delayed
```

This ensures the policy trains on **the same causal structure** it will experience on hardware — it cannot "see the future."

---

## 4. Reward Function Design

### 4.1 Full Reward Formula

```
R(t) = α·r_upright(θ) + β·r_effort(a) + γ·r_position(x) + δ·r_alive

r_upright(θ) =  1.0 − (θ / θ_max)²       ∈ [0, 1]
r_effort(a)  = −a²                         ∈ [−1, 0]
r_position(x)= −|x| / x_max               ∈ [−1, 0]
r_alive      =  1.0                        (constant)

Default: α=1.0, β=0.01, γ=0.10, δ=0.10, θ_max=0.5, x_max=2.0
```

### 4.2 Component Analysis

**r_upright — Quadratic angle cost:**
The `1 − (θ/θ_max)²` formulation is preferred over `−|θ|` (absolute) because:
- Smooth gradient everywhere (no cusp at θ=0)
- Reward is 1.0 at perfect balance, 0.0 at the fall boundary
- Penalizes large angles disproportionately (quadratic), discouraging near-miss behavior

**r_effort — Motor energy penalty:**
Penalizes `a²` (proportional to motor power). Without this term, the agent learns oscillatory "chattering" behavior — rapidly switching motor direction. The coefficient β=0.01 provides 1% of the upright reward's weight, ensuring effort is secondary to stability.

**r_position — Drift penalty:**
Prevents the robot from drifting to the edge of the arena. Without this term, a biased policy could "lean and run" — maintaining balance by constant forward motion. The coefficient γ=0.10 gives moderate discouragement.

**r_alive — Alive bonus:**
A constant +0.10 per step encourages the agent to survive as long as possible. This provides a dense reward signal even when the robot is already near-perfect (otherwise, small angle improvements yield negligible upright reward improvement).

### 4.3 Reward Scaling and Range

```
Best case (θ=0, a=0, x=0):   R = 1.0 + 0 + 0 + 0.1 = 1.10 / step
Worst case (|θ|→θ_max):      R = 0.0 − 0.01 − 0.1 + 0.1 = −0.01 / step

Max episode reward: 1000 × 1.10 = 1100 (truncation = success)
Min episode reward: ~0  (immediate fall at θ=θ_max)
```

---

## 5. PPO Algorithm

### 5.1 Policy Gradient Background

PPO optimizes the policy π_θ(a|s) by maximizing the expected return:

```
J(θ) = E_{τ~π_θ} [Σ_t γ^t r_t]
```

The policy gradient is estimated using the **advantage function** A(s,a) = Q(s,a) − V(s), which measures how much better action `a` is compared to the average under the current policy.

### 5.2 PPO Clipped Surrogate Objective

PPO avoids destructively large policy updates via a clipped importance ratio:

```
L^CLIP(θ) = E_t [min(r_t(θ)·Â_t,  clip(r_t(θ), 1−ε, 1+ε)·Â_t)]

where:
  r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)   ← probability ratio
  Â_t    = advantage estimate (GAE)
  ε      = 0.2                                  ← clip range
```

The clipping prevents the new policy from moving too far from the old policy in a single update step, providing stable monotonic improvement.

### 5.3 Generalized Advantage Estimation (GAE)

The advantage is estimated using GAE with λ=0.95:

```
Â_t = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}

where δ_t = r_t + γ·V(s_{t+1}) − V(s_t)   ← TD residual
      γ   = 0.99                             ← discount factor
      λ   = 0.95                             ← GAE parameter
```

**γ·λ = 0.94** provides a balance between low variance (short-horizon returns) and low bias (long-horizon returns).

### 5.4 Full Loss Function

```
L(θ) = −L^CLIP(θ) + c_vf·L^VF(θ) − c_ent·S[π_θ](s_t)

where:
  L^VF(θ) = (V_θ(s_t) − V_target)²     ← value function MSE loss
  S[π_θ]  = entropy of the policy       ← exploration bonus
  c_vf    = 0.5                         ← value coefficient
  c_ent   = 0.01                        ← entropy coefficient
```

### 5.5 Training Configuration

```yaml
n_envs:     8          # 8 parallel environments (DummyVecEnv)
n_steps:    2048       # steps collected per env per rollout
batch_size: 64         # minibatch size for gradient updates
n_epochs:   10         # passes over each rollout buffer
lr:         3e-4       # Adam optimizer learning rate
```

**Effective batch size:** 8 envs × 2048 steps = 16,384 transitions per rollout, split into 16,384 / 64 = 256 minibatches, each updated 10 times before the next rollout.

---

## 6. Policy Network Architecture

### 6.1 Network Topology

```
Input: obs ∈ ℝ⁴  (normalized state vector)
         │
         ▼
  Linear(4 → 64)   → Tanh   ← actor MLP (pi)
         │
         ▼
  Linear(64 → 64)  → Tanh
         │
         ├─── Linear(64 → 1)  → Tanh   ← action mean (deterministic output)
         │                              ← ∈ [−1, 1] for motor voltage
         │
         └─── log_std                  ← learned scalar (exploration noise)

Separate critic (vf) with identical topology → scalar V(s)
```

**Why Tanh over ReLU?**
For physical control tasks, Tanh is preferred because:
1. **Bounded outputs** — the final tanh squashes action to `[−1, 1]` naturally.
2. **Smooth everywhere** — no discontinuous gradient at activation boundaries.
3. **Symmetric about zero** — positive and negative motor commands are treated symmetrically.

### 6.2 Orthogonal Initialization

All weight matrices are initialized with orthogonal initialization:

```
W ~ Orthogonal(gain=√2)  for hidden layers
W ~ Orthogonal(gain=0.01) for output layer
```

Orthogonal initialization is empirically better for RL than Glorot/He because:
- Preserves gradient norms through early training
- The small output gain (0.01) means the policy starts near-random, exploring broadly

### 6.3 Parameter Count

```
Layer             Parameters
────────────────────────────
Linear(4→64)      4×64 + 64 = 320
Linear(64→64)     64×64 + 64 = 4,160
Linear(64→1)      64×1 + 1 = 65
log_std (scalar)  1
────────────────────────────
Actor total:      4,546

Critic (same arch → 1):
  + 4,547 parameters

Total policy:     ~9,100 parameters
```

At INT8 quantization: 9,100 weights × 1 byte = **~9 KB of weight data** (plus ONNX/TFLite overhead ≈ 32 KB total).

---

## 7. Domain Randomization

### 7.1 The Sim-to-Real Gap

The sim-to-real gap arises because the simulation is an imperfect model of the physical world. Key sources:

| Gap Source | Simulation | Reality | Impact |
|---|---|---|---|
| Body mass | Fixed 0.5 kg | ±10% from battery/wear | Pitch control gain |
| Motor friction | Fixed 0.001 Nm | Varies ±50% with temp | Response speed |
| IMU noise | Zero (ideal) | σ ≈ 0.005–0.020 rad | State estimation error |
| I2C latency | Zero | 5–15 ms | Observation delay |
| Wheel slip | Zero | 0–5% on smooth floor | Action effectiveness |

### 7.2 Randomization Distributions

All parameters sampled from **uniform distributions** (not Gaussian) because:
- Gaussian could assign near-zero probability to extreme values that exist in reality
- Uniform with manually specified bounds covers the physically possible range
- Simpler to reason about for hardware characterization

```python
mass    ~ U(0.40, 0.60)        # ±20% of nominal
friction ~ U(5e-4, 1.5e-3)     # ±50% of nominal
imu_σ   ~ U(1e-3, 2e-2)        # measured Allan deviation range
latency  ~ U(5, 15) ms          # measured I2C + DLPF latency
slip    ~ U(0, 0.05)            # empirical floor tests
delay   ~ {0, 1, 2} steps       # uniform discrete
```

### 7.3 Why DR Enables Zero-Shot Transfer

Without DR, the policy overfits to the nominal simulation dynamics. Adding DR forces the policy to learn a **robust** strategy that works across the entire distribution of environments. The optimal policy under DR is a minimax strategy:

```
π* = argmax_π E_{Φ~P(Φ)} [J(π, Φ)]

where Φ is the physics parameter vector and P(Φ) is the DR distribution.
```

This is equivalent to finding a policy that performs well in the **worst expected case** over the randomization distribution — which is precisely the physical robot with its uncertain parameters.

### 7.4 DR Scheduling (Optional Extension)

An advanced technique is **Automatic Domain Randomization (ADR)** — expanding DR ranges only when the policy performs well enough on the current range. The implementation supports this via the `DRConfig` dataclass, which can be updated dynamically during training.

---

## 8. Sim-to-Real Transfer Analysis

### 8.1 Transfer Protocol

1. **Train with DR** → policy π* robust over P(Φ)
2. **Export** → INT8 TFLite model, ~32 KB
3. **Flash** → ESP32, no retraining
4. **Test** → power on, run; measure balance time

### 8.2 Identifying Transfer Failures

If the zero-shot transfer fails, use this diagnostic protocol:

**Step 1: Log hardware data**
Flash the firmware with `VERBOSE_MODE=true`. Connect serial monitor. Let the robot attempt to balance. Record: `pitch_rad, pitch_rate_rads, action, loop_ms`.

**Step 2: Replay in simulation**
Feed the recorded observations through the policy in simulation. Compare actions — if actions match, the policy is consistent. If they diverge, there is a state estimation problem.

**Step 3: Identify failure mode**

| Symptom | Diagnosis | Fix |
|---|---|---|
| Oscillates rapidly then falls | Motor friction too low in DR | Increase `friction_high` |
| Slowly drifts, overcorrects | IMU gyro bias present | Add bias estimation to DR + firmware |
| Initial balance then drifts off | No position correction | Add position sensor (encoder odometry) |
| Never stabilizes, violent motion | Action delay mismatch | Increase `action_delay_high` |
| Oscillates at low frequency | High mass / low friction combo | Widen DR mass range |
| Works on hard floor, fails on carpet | Insufficient wheel slip DR | Increase `slip_high` |

### 8.3 Quantization Effect on Sim-to-Real

The INT8 quantization introduces a systematic approximation error:

```
ε_quant ≈ Δ/2  where Δ = (max − min) / 255  ← weight quantization step

For typical weights ∈ [−1, 1]: Δ ≈ 2/255 ≈ 0.008
Maximum action error:             ε_quant ≈ 0.004 (0.4% of action range)
```

This is well within the DR noise budget and empirically does not affect balance performance.

---

## 9. ONNX Export Pipeline

### 9.1 Extracting the Actor

SB3's `ActorCriticPolicy` contains both actor and critic networks. For edge deployment only the deterministic actor path is needed:

```python
# SB3 policy forward pass (training):
#   obs → features → (mean_action, log_std, value)

# Export path (deterministic):
#   obs → mlp_extractor.forward_actor(obs) → action_net(features) → tanh(mean)
```

The `ActorWrapper` class in `export/export_onnx.py` extracts this subgraph and applies the final `tanh` squashing, producing actions in `[−1, 1]`.

### 9.2 ONNX Verification

After export, every model is verified against the original PyTorch implementation on 20 random test vectors:

```python
max_diff = max(|PyTorch_output - ONNX_output|) for 20 random inputs
assert max_diff < 1e-5  # numerical equivalence check
```

This catches:
- Operator translation errors (rare with standard MLP ops)
- Shape mismatches
- Constant folding that accidentally removes parameters

### 9.3 ONNX Graph Structure

The exported MLP graph contains:
```
Input [batch, 4]
    │
  Gemm  [4→64]    ← Linear (fused weights + bias)
    │
  Tanh
    │
  Gemm  [64→64]
    │
  Tanh
    │
  Gemm  [64→1]
    │
  Tanh
    │
Output [batch, 1]
```

---

## 10. INT8 Quantization

### 10.1 Dynamic vs. Static Quantization

| Type | When quantized | Calibration needed | Best for |
|------|----------------|-------------------|----------|
| Dynamic (weights only) | Export time | No | MLPs, RNNs |
| Static (weights + activations) | Export time | Yes (calibration dataset) | CNNs |

This project uses **dynamic quantization** for the ONNX model because:
- No calibration dataset needed (weights are fully determined post-training)
- For small MLPs, activation quantization provides minimal additional benefit
- Simpler export pipeline

The TFLite model uses **full integer quantization** (static) via a representative dataset for ESP32 deployment, since the TFLite INT8 runtime requires statically quantized activations.

### 10.2 Quantization Mathematics

Each FP32 weight `w` is mapped to INT8 `q` via:

```
q = round(w / scale) + zero_point

where:
  scale      = (w_max − w_min) / 255
  zero_point = round(−w_min / scale)

Dequantization: w ≈ (q − zero_point) × scale
```

For weights symmetric around zero (typical for neural networks):

```
zero_point ≈ 0
scale      = w_max / 127
q          = round(w × 127 / w_max)
```

### 10.3 INT8 MatMul Execution

On ARM Cortex-M (ESP32 dual-core Xtensa LX6), INT8 matrix multiplication is computed using:

```
y_i = Σ_j (W_ij_int8 × x_j_int8)   [integer accumulation]
y_float = (y_int − zero_point_out) × scale_out
```

Modern ARM cores execute INT8 multiply-accumulate operations at **2–4× higher throughput** than FP32 due to SIMD packing (4 INT8 values in one 32-bit register vs. 1 FP32).

---

## 11. Edge Inference on ESP32

### 11.1 Memory Budget

ESP32-WROOM-32 memory:
```
SRAM:   320 KB total
  ├── Firmware code:     ~100 KB
  ├── TFLite tensor arena: ~16 KB  (configured in firmware)
  ├── Model weights:      ~32 KB   (embedded in flash, loaded to PSRAM)
  └── Stack/heap:         ~172 KB  available

Flash:  4 MB
  ├── Firmware + model:  ~200 KB
  └── Remaining:         ~3.8 MB  (SPIFFS/OTA available)
```

### 11.2 Control Loop Timing Budget

```
Loop budget: 10 ms (100 Hz)

Operation                    Typical Time
─────────────────────────────────────────
I2C read MPU-6050 (6 bytes)     0.9 ms   (400 kHz I2C, 6 bytes)
Complementary filter            0.1 ms
Encoder counter read            0.1 ms   (ISR + atomic read)
State normalization             0.05 ms
TFLite Invoke() [4→64→64→1]    6–9 ms   (INT8, no PSRAM penalty)
Action scaling + PWM write      0.1 ms
Serial print (every 5th loop)   0.3 ms   (amortized)
─────────────────────────────────────────
Total:                          7.6–10.5 ms

Margin:                        ~0–2 ms
```

**If loop overruns 10 ms:** Reduce tensor arena (triggers model recompilation), or move to 90 Hz control rate.

### 11.3 TFLite Micro Operator Resolution

The `AllOpsResolver` includes all TFLite ops but adds ~30 KB to firmware. For production, replace with `MicroMutableOpResolver` registering only:

```cpp
tflite::MicroMutableOpResolver<4> resolver;
resolver.AddFullyConnected();
resolver.AddTanh();
resolver.AddQuantize();
resolver.AddDequantize();
```

This saves ~25 KB of flash.

### 11.4 Tensor Arena Sizing

The tensor arena must hold all intermediate activations simultaneously:

```
Minimum arena size for 4→64→64→1 INT8 MLP:
  Input tensor:   1 × 4 × 1 byte   =    4 B
  Hidden 1:       1 × 64 × 1 byte  =   64 B
  Hidden 2:       1 × 64 × 1 byte  =   64 B
  Output:         1 × 1 × 1 byte   =    1 B
  TFLite overhead:                 ≈ 12 KB
  ─────────────────────────────────
  Total:                           ≈ 12.1 KB

Configured: 16 KB (33% headroom)
```

---

## 12. IMU Signal Processing

### 12.1 MPU-6050 Configuration

Register configuration in `imu_driver.h`:

```
PWR_MGMT_1  = 0x01   → Wake up; use gyro X clock (more stable)
SMPLRT_DIV  = 0x04   → Sample rate = 1000 / (1+4) = 200 Hz
CONFIG      = 0x03   → DLPF BW = 44 Hz (+4.9 ms group delay)
GYRO_CONFIG = 0x00   → ±250 deg/s full scale (131 LSB/deg/s)
ACCEL_CONFIG= 0x00   → ±2g full scale (16384 LSB/g)
```

**DLPF (Digital Low Pass Filter) trade-off:**
- Lower bandwidth → less noise, more delay
- Setting 3 (44 Hz) chosen: sufficiently suppresses >100 Hz motor vibration while keeping delay manageable (4.9 ms < 10 ms control period)

### 12.2 Complementary Filter

The complementary filter fuses gyroscope integration (accurate at high frequency, drifts slowly) with accelerometer pitch estimate (accurate at DC, noisy at high frequency):

```
θ[t] = α × (θ[t−1] + ω_gyro × Δt) + (1−α) × θ_accel

where:
  α = 0.98           ← trust gyro for 98% of bandwidth
  θ_accel = atan2(a_x, √(a_y² + a_z²))   ← pitch from gravity
  ω_gyro  = raw_gyro − bias               ← bias-corrected rate
```

**Filter frequency response:**
- Gyro path (high-pass): cutoff ≈ (1−α) / (2π×Δt) ≈ 0.32 Hz
- Accel path (low-pass): same cutoff 0.32 Hz
- The two paths sum to unity gain at all frequencies

### 12.3 Gyro Bias Calibration

At startup, 500 samples are averaged to estimate the zero-rate offset:

```
bias = (1/500) × Σ_{i=1}^{500} ω_raw[i]   (robot stationary)

Typical MPU-6050 bias: ±1 deg/s = ±0.017 rad/s
After calibration:     ±0.002 rad/s
```

This calibration is **blocking** (1 second) and should be performed after the robot is placed upright. If the calibration routine is skipped, gyro drift will cause the pitch estimate to slowly drift, eventually triggering the fall termination.

---

## 13. Motor Control

### 13.1 L298N H-Bridge

The L298N is a dual H-bridge motor driver. Each motor channel requires:
- **1 PWM pin** (enable, controls speed via duty cycle)
- **2 direction pins** (IN1/IN2 or IN3/IN4, set motor polarity)

Truth table (motor forward):
```
IN1=HIGH, IN2=LOW, ENA=PWM → Forward at `duty/1023 × V_supply` effective voltage
IN1=LOW, IN2=HIGH, ENA=PWM → Reverse
IN1=LOW, IN2=LOW, ENA=LOW  → Brake (short circuit across motor)
```

### 13.2 PWM Configuration

```
Frequency: 20,000 Hz  ← above audible range (prevents motor whine)
Resolution: 10-bit    ← 1024 levels, duty 0..1023
Max duty:   900       ← 87.9% headroom prevents H-bridge shoot-through
                         (L298N needs dead-time when switching direction)
Deadband:   50        ← duty < 50 → motor off (stiction region)
```

**Why 20 kHz?** Motor inductance L (typically 0.5–2 mH) acts as a low-pass filter for the PWM current ripple. At 20 kHz, ripple current ΔI = V_supply / (2fL) ≈ 2–5 mA (negligible). At 1 kHz, ΔI ≈ 40–100 mA causing significant heating and noise.

### 13.3 Encoder Processing

Hall-effect encoders generate quadrature pulses (A and B channels, 90° phase offset). Using only the rising edge of channel A:

```
Interrupt on A_RISING:
    count += (B == HIGH) ? +1 : −1

Wheel speed: ω = (Δcount / CPR) × 2π / Δt [rad/s]
```

With CPR=600 (N20 motor 20 CPR × 30:1 gear ratio), at 100 Hz control:

```
Min detectable speed: 1 count / (600 × 0.01 s) = 0.167 rad/s (1.6 RPM)
Max reliable speed:   limited by interrupt rate ≈ 600 Hz → 1 rad/s = 95 RPM
```

---

## 14. Hyperparameter Guide

### 14.1 Critical Hyperparameters

| Parameter | Value | Sensitivity | Notes |
|-----------|-------|-------------|-------|
| `learning_rate` | 3e-4 | High | Too high → unstable; reduce 10× if reward collapses |
| `n_steps` | 2048 | Medium | Larger = lower variance gradients, more memory |
| `gamma` | 0.99 | Medium | Reduce to 0.95 for faster convergence but shorter-horizon |
| `gae_lambda` | 0.95 | Low | Rarely needs tuning |
| `clip_range` | 0.2 | Low | Increase to 0.3 for more aggressive updates |
| `ent_coef` | 0.01 | Medium | Increase if agent gets stuck (local optima) |
| `n_epochs` | 10 | Low | Fewer = more stable but slower |
| `batch_size` | 64 | Low | Larger improves GPU utilization |

### 14.2 Reward Coefficient Tuning

The reward coefficients interact:

```
If robot oscillates but stays balanced:
    Increase β (motor effort penalty) → smoother control

If robot falls slowly to one side (not correcting):
    Decrease β (was over-penalizing correction) OR increase α

If robot balances but drifts linearly:
    Increase γ (position penalty)

If training is slow (reward plateaus early):
    Increase δ (alive bonus) → denser learning signal
```

### 14.3 DR Range Tuning

**Expanding DR too aggressively** makes the task too hard and prevents learning. Strategy:
1. Train baseline without DR (verify env works, baseline reward ~1000)
2. Add DR with narrow ranges (10% of nominal for each parameter)
3. Evaluate success rate; if >80%, expand ranges by 50%
4. Repeat until reaching the target DR bounds in `ppo_config.yaml`

---

## 15. Evaluation Methodology

### 15.1 Success Metric

**Primary:** Success rate = fraction of episodes lasting ≥ 999 steps (10 seconds at 100 Hz).

A 10-second balance is a meaningful physical target: it demonstrates sustained closed-loop control through minor disturbances, sensor noise, and motor variability.

### 15.2 Diagnostic Plots

**Balance time distribution:**
Histogram of episode lengths. A good policy shows a bimodal distribution: episodes that fall quickly (unlucky initial conditions) and episodes that reach truncation. A mediocre policy shows a smooth distribution with few truncations.

**Phase portrait (θ vs. θ̇):**
Plots trajectories in the angle–rate plane. A good policy converges to the origin (0, 0) from various starting points. Limit cycles indicate oscillatory behavior. Diverging trajectories indicate failure.

**Best episode trajectory:**
Time-series of all state variables for the best episode. Reveals control characteristics: settle time, steady-state oscillation amplitude, motor effort profile.

**Episode reward bar chart:**
Per-episode rewards colored by success (green) / failure (red). Reveals variance across episodes and identifies whether failures are systematic or random.

### 15.3 DR Stress Test

Evaluate the final policy with DR enabled (`--dr` flag) to measure robustness:

```bash
python3 evaluation/evaluate.py --model logs/best_model.zip --episodes 100 --dr
```

A success rate >70% under DR conditions indicates the policy is robust enough for zero-shot deployment.

---

## 16. Failure Modes & Debugging

### 16.1 Training Failures

**Reward collapses to ~0 after initial progress:**
- Cause: Learning rate too high; policy destroys itself with large update
- Fix: Reduce `learning_rate` by 5×; enable `normalize_advantage=True`

**Policy never learns (reward stays at baseline):**
- Cause: Reward too sparse or wrong sign; physics bug
- Fix: Run `python3 -c "from envs import PendulumBalanceEnv; e=PendulumBalanceEnv(); ..."`; manually verify reward values for known good states

**NaN in loss:**
- Cause: Exploding gradients; reward normalization issues
- Fix: Set `max_grad_norm=0.5`; clip reward to [-10, 10] in config

**Overfitting to simulator (fails under DR):**
- Cause: Trained without DR or DR ranges too narrow
- Fix: Retrain Phase 3 with full DR ranges; check `DRParamLogCallback` in TensorBoard

### 16.2 ESP32 Failures

**ESP32 crashes immediately after `AllocateTensors()`:**
- Cause: Tensor arena too small
- Fix: Increase `kTensorArenaSize` to `32 * 1024`; or enable PSRAM

**IMU reads garbage values (all zeros or ±32767):**
- Cause: I2C wiring error or voltage mismatch
- Fix: Verify 3.3V on MPU-6050 VCC; check SDA/SCL connections; verify address (AD0=GND → 0x68)

**Motors don't respond to PWM:**
- Cause: Wrong pin assignment or L298N enable not connected
- Fix: Probe PWM pin with oscilloscope/LED; verify `ENA` pin is connected to GPIO27

**Robot balances briefly then oscillates and falls (10–30 s):**
- Cause: Gyro bias drift (not calibrated at startup with `imu_calibrate_gyro()`)
- Fix: Ensure calibration runs before control loop; keep robot stationary during calibration

---

## 17. Extending the Project

### 17.1 Differential Steering (Turning)

Extend the action space to `[left_voltage, right_voltage]` ∈ ℝ²:

```python
action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=float32)
```

Modify the reward to include heading alignment:

```
R += ρ · cos(ψ − ψ_target)   ← heading reward (ρ = 0.5)
```

Train the policy to follow waypoints by providing heading error as an additional observation.

### 17.2 MuJoCo Integration

For higher-fidelity simulation, replace `pendulum_env.py` with a MuJoCo environment using the provided URDF:

```python
import gymnasium as gym
env = gym.make("MuJoCo-PendulumBalance-v0",
               xml_file="urdf/pendulum_robot.urdf")
```

MuJoCo uses RK4 integration and contact dynamics, providing more realistic collision/friction modeling. Expect ~5–10× slower simulation speed.

### 17.3 Sim-to-Real with System Identification

Instead of purely random DR, measure the physical robot's parameters:
1. Record motor step response → estimate K_t and b_friction
2. Perform IMU Allan deviation analysis → estimate noise σ
3. Use measured values as DR distribution centers, physical tolerance as ranges

This "informed DR" typically achieves higher success rates with narrower DR bands.

### 17.4 Online Adaptation (Meta-RL)

Replace standard PPO with **Model-Agnostic Meta-Learning (MAML)** or **Rapid Motor Adaptation (RMA)**:
- Train a base policy + adaptation module
- The adaptation module receives recent (action, obs) pairs and produces a latent context vector
- The context vector is fed to the policy, enabling online identification of current physics parameters
- This eliminates the need for DR and enables real-time adaptation

### 17.5 PSRAM Utilization

For larger policies (e.g., 128×128 hidden layers), enable PSRAM on supported ESP32 variants:

```cpp
// platformio.ini
board_build.extra_flags = -DBOARD_HAS_PSRAM
board_build.f_flash = 80000000L

// firmware
#include "esp_heap_caps.h"
static uint8_t* tensor_arena = 
    (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM);
```

This allows up to 4 MB for model weights — sufficient for policies with 256×256 hidden layers.

---

*Generated alongside the RL-Pendulum codebase. See [README.md](README.md) for quick start.*
