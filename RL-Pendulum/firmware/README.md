# Firmware — ESP32 Edge Inference

## Overview

The `inference_loop/` directory contains an Arduino sketch that runs the
quantized TFLite INT8 policy on an ESP32 at 100 Hz.

## Prerequisites

### Arduino IDE Setup

1. Install [Arduino IDE 2.x](https://www.arduino.cc/en/software)
2. Add ESP32 board package:
   - File → Preferences → Additional Board URLs:
     `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
   - Tools → Board Manager → Install **esp32** by Espressif Systems (v2.0.x)

### Required Libraries (Library Manager)

| Library                  | Version | Source               |
|--------------------------|---------|----------------------|
| TensorFlowLite_ESP32     | 0.9.0+  | Arduino Library Mgr  |
| Wire (built-in)          | -       | ESP32 core           |

## Generating the Model Header

Before building, you must export and convert the trained model:

```bash
# From project root
bash scripts/export_model.sh --model logs/best_model.zip

# This generates: firmware/inference_loop/rl_policy_data.h
# containing: const unsigned char model_tflite[] = { 0x1c, 0x00, ... };
```

If TensorFlow is not available for TFLite conversion, a placeholder header
with a dummy model is provided for compilation testing.

## Flashing

1. Open `inference_loop/inference_loop.ino` in Arduino IDE
2. Select **Tools → Board → ESP32 Dev Module**
3. Select **Tools → Port → /dev/ttyUSB0** (or COM port on Windows)
4. Upload (Ctrl+U)

## Serial Monitor

Open at **115200 baud**. Expected output every 50 ms:

```
===== RL-Pendulum Edge Inference =====
[IMU] MPU-6050 initialized at 200 Hz
Calibrating gyro — hold robot stationary for 1 second...
[IMU] Gyro bias: 0.000213 rad/s
[TF] Model loaded. Arena used: 12480 / 16384 bytes
[TF] Input  type=9 dims=[1,4]
[TF] Output type=9 dims=[1,1]
Starting 100 Hz control loop...
pitch_deg,pitch_rate_rads,lw_spd_rads,rw_spd_rads,action,loop_ms
0.412,0.031,0.234,0.238,0.0214,7
-0.195,-0.012,0.112,0.113,-0.0098,8
```

## Troubleshooting

| Issue                          | Solution                                  |
|--------------------------------|-------------------------------------------|
| `AllocateTensors() failed`     | Increase `kTensorArenaSize` to 32*1024    |
| `Schema version mismatch`      | Rebuild model with matching TFLite version |
| IMU reads all zeros            | Check I2C wiring; verify VCC is 3.3V      |
| Motors don't respond           | Check PWM pin assignments in motor_driver.h |
| Loop overrun warnings          | Reduce `kTensorArenaSize`; use PSRAM      |
