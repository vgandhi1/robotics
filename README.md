<div align="center">

# Semantic SLAM Rover

**Edge AI · LiDAR SLAM · Semantic Navigation**

[![ROS 2 Humble](https://img.shields.io/badge/ROS%202-Humble-blue?logo=ros&logoColor=white)](https://docs.ros.org/en/humble/)
[![Jetson Orin](https://img.shields.io/badge/NVIDIA-Jetson%20Orin%20Nano-76b900?logo=nvidia&logoColor=white)](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)
[![TensorRT FP16](https://img.shields.io/badge/TensorRT-FP16%20%E2%89%A530%20FPS-76b900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/tensorrt)
[![Tests](https://img.shields.io/badge/tests-46%20passed-brightgreen?logo=pytest&logoColor=white)](tests/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

*A ground rover that doesn't just map space — it understands what's in it.*

</div>

---

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SEMANTIC SLAM ROVER                                │
│                                                                             │
│   Camera ──► YOLOv8 TRT ──►╮                                               │
│                              ╠──► Fusion Node ──► Semantic Map              │
│   LiDAR  ──► SLAM Toolbox ──╯         │                                     │
│                                       ▼                                     │
│                           "navigate_to: blue_box" ──► Nav2 ──► Motors      │
└─────────────────────────────────────────────────────────────────────────────┘
```

A standard warehouse robot sees LiDAR geometry — walls, columns, obstacles. It doesn't know if an obstacle is a concrete pillar or a human. This project bridges that gap: **YOLOv8 running at 45 FPS on a Jetson Orin Nano** fuses every detection with the LiDAR scan to place class-labeled 3D landmarks into a live map. Nav2 can then act on semantic commands like *"find the fire extinguisher and stop 0.5 m in front of it."*

---

## Contents

- [Architecture](#architecture)
- [Hardware](#hardware)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Running Tests](#running-tests)
- [Key Algorithms](#key-algorithms)
- [ROS 2 Interface](#ros-2-interface)
- [Docker Deployment](#docker-deployment)
- [Configuration](#configuration)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)

---

## Architecture

```
                         ┌────────────────────────────────────────────┐
                         │            NVIDIA Jetson Orin Nano          │
                         │                                             │
  ┌──────────┐           │  ┌─────────────┐    ┌────────────────────┐ │
  │  Camera  │──/image──►│  │  yolo_node  │    │   slam_toolbox     │ │
  │  IMX219  │           │  │  TensorRT   │    │   (async SLAM)     │ │
  └──────────┘           │  │  FP16 45FPS │    │                    │ │
                         │  └──────┬──────┘    └────────┬───────────┘ │
  ┌──────────┐           │         │ /detections         │ /map /tf    │
  │ RPLiDAR  │──/scan───►│         └──────────┬──────────┘            │
  │  A1/A2   │           │                    │                        │
  └──────────┘           │            ┌───────▼────────┐              │
                         │            │  fusion_node   │              │
                         │            │  projection    │              │
                         │            │  math + EMA    │              │
                         │            └───────┬────────┘              │
                         │                    │ /semantic/landmarks   │
                         │            ┌───────▼────────────────┐     │
                         │            │  semantic_navigator     │     │
                         │            │  class_label → Nav2    │     │
                         │            └───────┬────────────────┘     │
                         │                    │ /navigate_to_pose     │
                         │            ┌───────▼────────┐             │
                         │            │    Nav2 Stack  │             │
                         │            │  DWB + Costmap │             │
                         └────────────└───────┬────────┘─────────────┘
                                              │ /cmd_vel
                              ┌───────────────▼──────────────────┐
                              │           ESP32                   │
                              │  Serial bridge · Motor PWM        │
                              │  Encoder ISR · IMU I2C            │
                              └──────────────────────────────────┘
```

---

## Hardware

| Component | Part | Role |
|-----------|------|------|
| Edge Compute | NVIDIA Jetson Orin Nano 8 GB | TensorRT inference · ROS 2 host |
| Microcontroller | ESP32 DevKit v1 | Motor PWM · Encoder odometry · IMU |
| LiDAR | RPLiDAR A1/A2 (360°) | 2D scan for SLAM + sensor fusion |
| Camera | IMX219 / OV9281 CSI | YOLOv8 input stream (30 FPS) |
| Motor Driver | L298N / DRV8833 | Differential drive H-bridge |
| IMU | MPU-6050 I2C | Orientation assist for SLAM |
| Chassis | 2-wheel differential | ~30 × 25 cm footprint |
| Power | 3S LiPo 11.1 V + buck converters | Motors 7.4 V · Jetson 5 V/4 A |

---

## Project Structure

```
Semantic-SLAM-Rover/
│
├── ros2_ws/src/
│   ├── rover_msgs/                 # Custom message & service definitions
│   │   ├── msg/Detection2D.msg
│   │   ├── msg/SemanticLandmark.msg
│   │   ├── msg/SemanticMap.msg
│   │   └── srv/NavigateToClass.srv
│   │
│   ├── rover_perception/           # YOLOv8 inference (TensorRT + PyTorch)
│   │   └── rover_perception/
│   │       ├── tensorrt_engine.py  # GPU buffer mgmt, letterbox, NMS
│   │       └── yolo_node.py        # ROS 2 subscriber / publisher
│   │
│   ├── rover_fusion/               # Camera-LiDAR semantic fusion ★
│   │   └── rover_fusion/
│   │       ├── projection_math.py  # Pixel → bearing → LiDAR → XYZ
│   │       ├── landmark_tracker.py # EMA fusion, dedup, aging
│   │       └── fusion_node.py      # ROS 2 node + service server
│   │
│   ├── rover_slam/                 # SLAM Toolbox async config + launch
│   ├── rover_navigation/           # Nav2 params + SemanticNavigator
│   ├── rover_hardware/             # ESP32 serial bridge + odometry
│   └── rover_bringup/              # Full-system launch + RViz config
│
├── firmware/esp32/motor_controller/
│   └── motor_controller.ino        # Encoder ISR · PWM · IMU · watchdog
│
├── models/
│   └── export_tensorrt.py          # YOLOv8 → ONNX → TRT FP16 pipeline
│
├── tools/calibration/
│   └── camera_lidar_calibration.py # Color blob / ArUco yaw calibration
│
├── tests/
│   ├── unit/                       # 38 pure-Python math tests
│   └── integration/                # End-to-end pipeline tests (no ROS)
│
└── docker/
    ├── Dockerfile.jetson            # JetPack 6 + ROS 2 Humble image
    └── docker-compose.yml
```

---

## Quick Start

### 1 — Build the ROS 2 Workspace

```bash
# Clone
git clone https://github.com/vgandhi1/semantic-SLAM-Rover.git
cd semantic-SLAM-Rover/ros2_ws

# Install ROS 2 package dependencies
rosdep install --from-paths src --ignore-src -r -y

# Build (Release mode for best performance on Jetson)
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

### 2 — Export YOLOv8 TensorRT Engine *(Jetson only)*

```bash
pip3 install ultralytics

python3 models/export_tensorrt.py \
    --model yolov8n          \
    --output /opt/rover/models \
    --fp16                   \
    --validate
```

```
Validation (20 runs): mean=22.3 ms  p50=22.1 ms  p95=24.8 ms  FPS=45
PASS: Latency target met (22.3 ms < 30 ms).
Engine: /opt/rover/models/yolov8n_640.engine  (16.4 MB)
```

### 3 — Flash ESP32 Firmware

Open `firmware/esp32/motor_controller/motor_controller.ino` in Arduino IDE.
Adjust the chassis constants, select **ESP32 Dev Module**, upload.

```cpp
static constexpr float WHEEL_BASE_M     = 0.22f;   // centre-to-centre (m)
static constexpr float WHEEL_RADIUS_M   = 0.033f;  // wheel radius (m)
static constexpr int   ENCODER_TICKS_REV = 1120;   // 28 pulse × 40:1 gear
```

### 4 — Calibrate Camera ↔ LiDAR

Place a bright target (red object or ArUco marker) 1–2 m ahead of the robot, aligned with the camera centre. Then run:

```bash
python3 tools/calibration/camera_lidar_calibration.py \
    --target-color red \
    --samples 30
```

Copy the reported `camera_yaw_offset_deg` into `rover_bringup/config/rover_params.yaml`.

### 5 — Launch Everything

```bash
# Single command — launches hardware, SLAM, perception, fusion, Nav2
ros2 launch rover_bringup rover_full_bringup.launch.py \
    use_hardware:=true                                  \
    engine_path:=/opt/rover/models/yolov8n_640.engine

# Optional: open RViz on a remote machine
export ROS_DOMAIN_ID=42
ros2 launch rover_bringup rover_full_bringup.launch.py use_rviz:=true
```

### 6 — Send a Semantic Goal

```bash
# Navigate to the nearest "person" (explore if not yet detected)
ros2 service call /rover/navigate_to_class rover_msgs/srv/NavigateToClass \
  "{class_label: 'person', approach_distance: 0.5, explore_if_not_found: true}"

# Query the live semantic map
ros2 service call /rover/get_semantic_landmarks rover_msgs/srv/GetSemanticLandmarks \
  "{class_label: '', max_age_seconds: 60.0}"

# Watch mission status in real time
ros2 topic echo /rover/mission/status
```

---

## Running Tests

No ROS 2 or GPU required — pure Python math and logic:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install pytest numpy opencv-python-headless

pytest tests/ -v
```

```
tests/integration/test_full_pipeline.py ....                      [  8%]
tests/unit/test_landmark_tracker.py ..............                [ 39%]
tests/unit/test_projection_math.py ....................            [ 82%]
tests/unit/test_tensorrt_engine.py ........                       [100%]

46 passed in 1.29s
```

---

## Key Algorithms

### Sensor Fusion — Pixel to 3D Map Point

```python
# 1. Bounding box centroid
u = (x_min + x_max) / 2.0
v = (y_min + y_max) / 2.0

# 2. Pixel → normalised image plane (camera frame)
x_n = (u - cx) / fx
y_n = (v - cy) / fy
bearing_cam = [x_n, y_n, 1.0] / norm([x_n, y_n, 1.0])

# 3. Rotate to robot frame via extrinsic matrix
bearing_robot = R_cam_to_robot @ bearing_cam
theta = atan2(bearing_robot[1], bearing_robot[0])

# 4. LiDAR scan index
idx = round((theta - angle_min) / angle_increment)
r = median(scan.ranges[idx-3 : idx+4])   # 7-ray noise suppression

# 5. Robot-frame XY
x_robot = r * cos(theta)
y_robot = r * sin(theta)

# 6. Map frame via TF
x_map, y_map = apply_2d_tf(x_robot, y_robot, tf_base_link_to_map)
```

### Landmark Deduplication

Observations of the same class within 0.5 m are merged via **Exponential Moving Average** (α = 0.3):

```
x_map ← 0.3 · x_new  +  0.7 · x_existing
```

Landmarks age out after 30 s without re-observation and are pruned at 0.2 Hz.

---

## ROS 2 Interface

### Topics

| Topic | Type | Direction | Description |
|-------|------|-----------|-------------|
| `/camera/image_raw` | `sensor_msgs/Image` | In | Camera frames |
| `/scan` | `sensor_msgs/LaserScan` | In | LiDAR scan |
| `/rover/detections` | `rover_msgs/Detection2DArray` | Out | YOLO results |
| `/rover/semantic/landmarks` | `rover_msgs/SemanticMap` | Out | Live semantic map |
| `/rover/semantic/markers` | `visualization_msgs/MarkerArray` | Out | RViz spheres + labels |
| `/rover/mission/status` | `std_msgs/String` (JSON) | Out | Navigation state |
| `/cmd_vel` | `geometry_msgs/Twist` | In/Out | Velocity commands |
| `/odom` | `nav_msgs/Odometry` | Out | Wheel odometry |

### Services

| Service | Type | Description |
|---------|------|-------------|
| `/rover/navigate_to_class` | `rover_msgs/NavigateToClass` | Send semantic nav goal |
| `/rover/get_semantic_landmarks` | `rover_msgs/GetSemanticLandmarks` | Query semantic map |

### Custom Messages

```
Detection2DArray
  ├── header
  ├── inference_latency_ms  float32
  └── detections[]
        ├── class_label     string
        ├── confidence      float32
        ├── x_min/y_min     uint32
        ├── x_max/y_max     uint32
        └── image_width/height uint32

SemanticLandmark
  ├── class_label           string
  ├── confidence            float32
  ├── position              geometry_msgs/Point  (map frame)
  ├── range                 float32
  ├── observation_count     uint32
  ├── first_seen / last_seen
  └── landmark_id           string
```

---

## Docker Deployment

```bash
# Build image on Jetson (aarch64)
docker buildx build --platform linux/arm64 \
    -f docker/Dockerfile.jetson           \
    -t semantic-slam-rover:latest .

# Launch full stack
docker-compose up rover

# Run tests inside container
docker-compose --profile test up test
```

---

## Configuration

All parameters live in `ros2_ws/src/rover_bringup/config/rover_params.yaml`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `conf_threshold` | `0.45` | Detection confidence floor |
| `merge_radius_m` | `0.5` | Merge radius for landmark dedup |
| `max_landmark_age_s` | `30.0` | Landmark expiry (seconds) |
| `min_range_m` | `0.30` | Near-field LiDAR rejection floor |
| `camera_yaw_offset_deg` | `0.0` | Camera-LiDAR alignment offset |
| `camera_pitch_offset_deg` | `-5.0` | Camera downward tilt |
| `wheel_base_m` | `0.22` | Chassis width (m) |
| `encoder_ticks_rev` | `1120` | Encoder resolution |
| `cmd_timeout_s` | `0.5` | ESP32 command watchdog |

---

## Performance

Benchmarked on **Jetson Orin Nano 8 GB**, JetPack 6.0, 640 × 480 camera, RPLiDAR A1:

| Stage | Latency | Rate |
|-------|---------|------|
| YOLOv8n TensorRT FP16 | ~22 ms | **45 FPS** |
| Fusion node (per frame) | ~3 ms | 10 Hz (scan-limited) |
| SLAM map update | — | 5 Hz |
| Nav2 path planning | ~50 ms | 20 Hz |
| Jetson power draw | — | ~9 W (inference) |

Target: landmark position error ≤ 0.3 m at 1–4 m range (validated with ruler measurements).

---

## Troubleshooting

<details>
<summary><strong>No detections on /rover/detections</strong></summary>

```bash
# Check inference rate
ros2 topic hz /rover/detections

# Verify engine file
ls -lh /opt/rover/models/*.engine

# Fall back to PyTorch (development mode)
# In rover_params.yaml: use_tensorrt: false
```
</details>

<details>
<summary><strong>Landmarks at wrong positions</strong></summary>

```bash
# Run calibration tool
python3 tools/calibration/camera_lidar_calibration.py --target-color red

# Verify TF tree is complete
ros2 run tf2_ros tf2_echo map base_link
ros2 run tf2_ros tf2_echo base_link laser
```
</details>

<details>
<summary><strong>ESP32 not connecting</strong></summary>

```bash
ls /dev/ttyUSB*            # identify port
sudo usermod -aG dialout $USER && newgrp dialout
# Update port in rover_params.yaml: port: /dev/ttyUSB1
```
</details>

<details>
<summary><strong>SLAM not building map</strong></summary>

```bash
ros2 topic hz /scan        # must be ~10 Hz
ros2 run tf2_ros tf2_echo odom base_link   # must be publishing
```
</details>

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).

---

<div align="center">
Built with ROS 2 Humble · NVIDIA TensorRT · slam_toolbox · Nav2
</div>
