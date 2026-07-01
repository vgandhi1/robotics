# Semantic-SLAM-Rover: Edge Vision and Spatial Fusion

An autonomous edge-compute rover that fuses AI vision (semantic "what is this?") with LiDAR-based SLAM (spatial "where is it?") to build richly annotated 3D maps and execute goal-directed navigation via natural language class labels.

---

## System Overview

A standard warehouse robot uses LiDAR to build a map of obstacles as geometry — it sees walls, columns, and boxes, but cannot distinguish a concrete pillar from a human. This project runs YOLOv8 on an NVIDIA Jetson Orin Nano at 30+ FPS (via TensorRT FP16), fuses each detection's bounding box with the LiDAR point cloud using camera intrinsic projection math, and writes class-labeled 3D landmarks into a live ROS 2 occupancy grid. Nav2 then uses those semantic coordinates to navigate commands like "go to the blue box."

---

## Hardware Bill of Materials

| Component | Model | Purpose |
|-----------|-------|---------|
| Edge Compute | NVIDIA Jetson Orin Nano 8GB | TensorRT inference, ROS 2 host |
| Microcontroller | ESP32 DevKit | Motor PWM, encoder odometry |
| LiDAR | RPLiDAR A1/A2 (360°) | 2D scan for SLAM + fusion |
| Camera | IMX219 / OV9281 (CSI) | YOLOv8 input stream |
| Motor Driver | L298N / DRV8833 | Differential drive H-bridge |
| IMU | MPU-6050 (I2C) | Orientation + acceleration |
| Chassis | 2-wheel differential drive | ~30×25 cm platform |
| Power | 3S LiPo 11.1V + buck converters | Motors (7.4V) + Jetson (5V/4A) |

---

## Software Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Jetson Orin Nano                         │
│  ┌──────────────┐   ┌──────────────┐   ┌─────────────────────┐  │
│  │  Perception  │   │  Spatial     │   │  Fusion             │  │
│  │  Engine      │   │  Engine      │   │  Node               │  │
│  │              │   │              │   │                     │  │
│  │ /camera/raw  │   │ /scan        │   │ bbox + scan → XYZ   │  │
│  │ ↓ YOLOv8    │   │ ↓ SLAM       │   │ → /semantic/        │  │
│  │   TensorRT   │   │   Toolbox    │   │    landmarks        │  │
│  │ → /detections│   │ → /map       │   │ → /semantic/map     │  │
│  └──────────────┘   │   /tf        │   └─────────────────────┘  │
│                     └──────────────┘                             │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                   Nav2 Stack                               │  │
│  │  SemanticNavigator → goal pose lookup → action client      │  │
│  └────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
           │ USB Serial (micro-ROS / serial bridge)
┌──────────▼──────────────────────────────────────────────────────┐
│                          ESP32                                   │
│  Motor PWM  │  Encoder ISR  │  IMU I2C  │  /cmd_vel → wheels    │
└─────────────────────────────────────────────────────────────────┘
```

---

## ROS 2 Node Graph

```
/camera/image_raw  ──►  [yolo_tensorrt_node]  ──►  /rover/detections
/scan              ──►  [slam_toolbox]         ──►  /map, /tf
                   ──►  [fusion_node]          ──►  /rover/semantic/landmarks
                                                     /rover/semantic/map
/rover/semantic/landmarks  ──►  [semantic_navigator]  ──►  /navigate_to_pose
/cmd_vel           ──►  [esp32_bridge_node]    ──►  (serial → ESP32)
/odom              ◄──  [esp32_bridge_node]    ◄──  (serial ← ESP32)
```

---

## Custom Message Types

### `rover_msgs/msg/Detection2D.msg`
```
std_msgs/Header header
string class_label
float32 confidence
uint32 x_min
uint32 y_min
uint32 x_max
uint32 y_max
uint32 image_width
uint32 image_height
```

### `rover_msgs/msg/Detection2DArray.msg`
```
std_msgs/Header header
rover_msgs/Detection2D[] detections
```

### `rover_msgs/msg/SemanticLandmark.msg`
```
std_msgs/Header header
string class_label
float32 confidence
geometry_msgs/Point position       # map frame XYZ
float32 range                      # distance from robot at time of detection
builtin_interfaces/Time last_seen
```

### `rover_msgs/msg/SemanticMap.msg`
```
std_msgs/Header header
rover_msgs/SemanticLandmark[] landmarks
```

---

## Pipeline Detail

### Phase 1 — Edge AI Optimization (Perception Engine)

**Goal:** YOLOv8 at ≥ 30 FPS on Jetson Orin Nano.

**Steps:**
1. Start with `yolov8n.pt` (nano, 3.2M params). Benchmark PyTorch inference: ~8–12 FPS on Jetson.
2. Export to ONNX: `model.export(format='onnx', imgsz=640, simplify=True)`
3. Build TensorRT engine with FP16: `trtexec --onnx=yolov8n.onnx --fp16 --saveEngine=yolov8n.engine`
4. Validate: latency drops to ≤ 25ms/frame (≥40 FPS). Memory footprint ~300 MB.
5. ROS 2 node subscribes `/camera/image_raw`, runs TRT inference, publishes `Detection2DArray`.

**TensorRT inference loop:**
```python
# Load engine → allocate GPU buffers → copy image → execute → copy results
engine = trt.Runtime(trt.Logger()).deserialize_cuda_engine(open('yolov8n.engine','rb').read())
context = engine.create_execution_context()
# host/device buffer management via pycuda
```

### Phase 2 — Data Pipeline (ROS 2 Hardware Bringup)

**Components to verify:**
- `/scan` from RPLiDAR at 10 Hz, 360 rays, ~0.3–12 m range
- `/camera/image_raw` at 30 FPS, 640×480 BGR8
- `/odom` from ESP32 encoder counts → differential drive kinematics
- `/imu/data` from MPU-6050 (optional, improves SLAM)

**ESP32 serial protocol:**
```
Jetson→ESP32: "CMD,v_linear,v_angular\n"   (m/s, rad/s → PWM)
ESP32→Jetson: "ODO,left_ticks,right_ticks,dt_ms\n"
```

### Phase 3 — Sensor Fusion Math (Core Algorithm)

**Camera intrinsics (K matrix):**
```
K = [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]
```
Calibrated with `ros2 run camera_calibration cameracalibrator`.

**Bounding box → LiDAR ray projection:**
```python
# 1. Centroid pixel of detection
u = (x_min + x_max) / 2.0
v = (y_min + y_max) / 2.0

# 2. Pixel → normalized image plane (camera frame)
x_n = (u - cx) / fx
y_n = (v - cy) / fy

# 3. Horizontal angle in camera frame
theta_cam = math.atan(x_n)

# 4. Add extrinsic offset (camera → LiDAR transform, yaw component)
theta_lidar = theta_cam + camera_to_lidar_yaw_offset

# 5. LiDAR scan index
idx = int((theta_lidar - scan.angle_min) / scan.angle_increment)
idx = max(0, min(idx, len(scan.ranges) - 1))

# 6. Range at that angle
r = scan.ranges[idx]
if math.isnan(r) or r < scan.range_min or r > scan.range_max:
    return None  # invalid measurement

# 7. Robot-frame XY
x_robot = r * math.cos(theta_lidar)
y_robot = r * math.sin(theta_lidar)

# 8. Transform to map frame via TF
robot_to_map = tf_buffer.lookup_transform('map', 'base_link', stamp)
x_map, y_map = apply_2d_transform(x_robot, y_robot, robot_to_map)
```

**Landmark deduplication:** landmarks of the same class within 0.5 m are merged (position averaged over last N sightings). Age-out after 30 s without re-detection.

### Phase 4 — Autonomous Semantic Navigation

**Nav2 integration:**
1. `SemanticNavigator` node subscribes `/rover/semantic/landmarks`.
2. Accepts goal: `"navigate_to_class": "blue_box"` via ROS 2 service.
3. Looks up the most recently seen landmark of that class.
4. Sends `NavigateToPose` action to Nav2 with the landmark's map-frame pose.
5. Monitors feedback; on success publishes `/rover/mission/status`.

**Exploration strategy (no prior landmark):**
- Frontier-based exploration using `slam_toolbox` unknown-space frontiers.
- After each frontier is visited, re-check landmark map for target class.

---

## Project Directory Structure

```
semantic_slam_rover/
├── ref.md                          # This document
├── README.md                       # Setup and run guide
├── ros2_ws/
│   └── src/
│       ├── rover_msgs/             # Custom ROS 2 message definitions
│       ├── rover_perception/       # YOLOv8 TensorRT inference node
│       ├── rover_fusion/           # Camera-LiDAR sensor fusion
│       ├── rover_slam/             # SLAM Toolbox config + launch
│       ├── rover_navigation/       # Nav2 + semantic navigator node
│       ├── rover_hardware/         # ESP32 serial bridge + hardware
│       └── rover_bringup/          # Full-system launch files
├── firmware/
│   └── esp32/motor_controller/     # Arduino/PlatformIO sketch
├── models/
│   └── export_tensorrt.py          # YOLOv8 → TensorRT export script
├── tools/
│   ├── calibration/                # Camera-LiDAR calibration utilities
│   └── visualization/              # RViz configs
├── tests/
│   ├── unit/                       # Math and node unit tests
│   └── integration/                # Full pipeline integration tests
└── docker/
    ├── Dockerfile.jetson           # JetPack 6 + ROS 2 Humble image
    └── docker-compose.yml
```

---

## Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| ROS 2 Humble | LTS | Middleware, nav stack |
| slam_toolbox | ≥2.6 | Async SLAM, lifelong mapping |
| nav2 | ≥1.1 | Path planning, obstacle avoidance |
| ultralytics | ≥8.0 | YOLOv8 model + ONNX export |
| tensorrt | ≥8.6 | FP16 engine on Jetson |
| pycuda | ≥2022.1 | GPU buffer management |
| opencv-python | ≥4.8 | Image pre/post processing |
| numpy | ≥1.24 | Array math |
| pyserial | ≥3.5 | ESP32 serial bridge |

---

## Performance Targets

| Metric | Target | How Measured |
|--------|--------|--------------|
| YOLO inference latency | ≤ 25 ms | `ros2 topic hz /rover/detections` |
| End-to-end fusion latency | ≤ 50 ms | header stamp delta |
| SLAM map update rate | ≥ 5 Hz | `ros2 topic hz /map` |
| Landmark position error | ≤ 0.3 m | Ground-truth ruler measurement |
| Nav2 goal success rate | ≥ 90% | 10-run average in test arena |
| Jetson power draw | ≤ 10 W (inference) | `tegrastats` |

---

## Camera–LiDAR Extrinsic Calibration

The camera is mounted above or beside the LiDAR. The key extrinsic is the **yaw offset** (rotation around Z) and **translational offset** (meters, robot frame). Calibrate by pointing both sensors at a known reflector/target and measuring the angular difference between the projected camera ray and the LiDAR return at the same point.

```yaml
# config/sensor_extrinsics.yaml
camera_to_lidar:
  translation: [0.05, 0.0, 0.12]   # x (forward), y (left), z (up) in meters
  rotation_yaw_deg: 0.0             # camera optical axis aligned with LiDAR forward
  rotation_pitch_deg: -5.0          # camera tilted 5° down
```

---

## Safety and Edge Cases

- **Invalid LiDAR range:** `nan`, `inf`, or out-of-`[range_min, range_max]` values are rejected; detection is skipped.
- **Stale transforms:** TF lookup uses `rospy.Duration(0.1)` tolerance; if TF is not available within 100 ms, detection is dropped with a warning.
- **Near-field detections:** objects closer than 0.3 m produce unreliable LiDAR returns; suppressed by range floor check.
- **Multiple detections same class:** landmarks within 0.5 m of the same class are merged; prevents map bloat from stationary objects.
- **ESP32 disconnect:** hardware bridge node enters safe-stop mode (publishes zero velocity) and retries serial connection every 5 seconds.
