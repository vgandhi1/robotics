# Technical Guide — Semantic SLAM Rover

> **Audience:** Robotics engineers, ML practitioners, and embedded systems developers who want a deep understanding of how every component works, why design decisions were made, and how to extend or replicate the system.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Hardware Design Rationale](#2-hardware-design-rationale)
3. [ROS 2 Architecture](#3-ros-2-architecture)
4. [Perception Engine — YOLOv8 + TensorRT](#4-perception-engine--yolov8--tensorrt)
5. [Sensor Fusion — The Core Algorithm](#5-sensor-fusion--the-core-algorithm)
6. [Landmark Tracking](#6-landmark-tracking)
7. [SLAM — slam_toolbox](#7-slam--slam_toolbox)
8. [Navigation — Nav2 + Semantic Goals](#8-navigation--nav2--semantic-goals)
9. [Hardware Bridge — ESP32 Serial Protocol](#9-hardware-bridge--esp32-serial-protocol)
10. [Camera–LiDAR Extrinsic Calibration](#10-cameralidar-extrinsic-calibration)
11. [TensorRT Export and Optimization](#11-tensorrt-export-and-optimization)
12. [Custom ROS 2 Messages](#12-custom-ros-2-messages)
13. [Testing Strategy](#13-testing-strategy)
14. [Docker Deployment](#14-docker-deployment)
15. [Performance Analysis](#15-performance-analysis)
16. [Known Limitations and Future Work](#16-known-limitations-and-future-work)

---

## 1. System Overview

The Semantic SLAM Rover solves a fundamental limitation of standard mobile robots: classical SLAM produces occupancy grids where every occupied cell is geometrically equivalent. There is no distinction between a cardboard box and a person. This project adds a semantic layer by:

1. Running an object detection neural network on camera frames at real-time rates.
2. Fusing each 2D detection with the simultaneous LiDAR scan to recover the 3D position of the detected object in map coordinates.
3. Accumulating these 3D observations into a persistent semantic landmark map.
4. Exposing that map to a navigation stack that can route the robot to any object class.

The result is a robot that can receive a command like `"navigate_to_class: fire_extinguisher"` and autonomously locate and approach that object, even if it was first observed several rooms away.

### Design Philosophy

- **Modularity:** Each capability (perception, fusion, SLAM, navigation, hardware) lives in a separate ROS 2 package with its own message types and launch file. Any module can be swapped independently.
- **Hardware-portability:** TensorRT is used on Jetson for maximum performance, but the `UltralyticsEngine` fallback means the full pipeline (minus TRT) runs on any x86 Linux machine with a webcam and a LiDAR.
- **Testability without ROS:** All mathematical core code (`projection_math.py`, `landmark_tracker.py`) is pure Python with zero ROS imports. The 46-test suite runs on any Python 3.10+ environment in ~1.3 seconds.
- **Safety first:** Motor command watchdog on ESP32 (stops after 500 ms), TF timeout in fusion node, scan age check before fusion, near-field rejection.

---

## 2. Hardware Design Rationale

### Why Jetson Orin Nano vs Raspberry Pi / Coral TPU?

| Platform | YOLOv8n FP16 FPS | Notes |
|----------|-------------------|-------|
| Jetson Orin Nano 8 GB | ~45 FPS | TensorRT, full ROS 2, CUDA |
| Jetson Nano 4 GB | ~12 FPS | older TensorRT, memory-limited |
| Raspberry Pi 5 | ~5 FPS | CPU only, no CUDA |
| Coral Dev Board | ~80 FPS | INT8 only, limited model flexibility |

The Orin Nano hits the 30 FPS minimum requirement with headroom for richer models (`yolov8s`), runs the full ROS 2 + Nav2 stack, and supports CUDA-accelerated map operations.

### Why RPLiDAR A1/A2?

The A1 provides 360° at 10 Hz with ~0.3–12 m range, which is sufficient for indoor navigation up to ~10 m corridors. The scan geometry (single 2D plane at ~15 cm height) means the scan line passes through most obstacles at a useful height. The A2 doubles the scan rate to 20 Hz and extends range — a drop-in upgrade.

### Why ESP32 vs Direct PWM from Jetson?

The Jetson GPIO is 3.3 V and limited; its Python GPIO libraries lack real-time guarantees. Delegating motor PWM and encoder counting to the ESP32 provides:

- **Deterministic interrupt timing:** ESP32 ISRs fire in microseconds, critical for encoder accuracy at high RPM.
- **Safety isolation:** The ESP32's watchdog stops motors if the serial link drops, independent of the Jetson.
- **Clean electrical isolation:** The L298N/DRV8833 motor power rail is separate from the Jetson supply.

---

## 3. ROS 2 Architecture

### Node Graph

```
[/camera/image_raw] ─────────────────────► [yolo_node]
                                                │
                                         /rover/detections
                                                │
[/scan] ──────────────────┬────────────► [fusion_node] ──► /rover/semantic/landmarks
                          │                                 /rover/semantic/markers
                          ▼
                   [slam_toolbox] ──────► /map
                                          /tf (map→odom→base_link)

[/camera/camera_info] ──────────────────► [fusion_node] (intrinsics, once)

/rover/semantic/landmarks ──────────────► [semantic_navigator]
                                                │
                                         /navigate_to_pose
                                                │
                                          [Nav2 stack] ──► /cmd_vel
                                                               │
                                                      [esp32_bridge_node] ──► (serial)
                                                               │
                                                          [/odom] ◄── (serial)
```

### QoS Policies

| Topic | Reliability | History | Rationale |
|-------|-------------|---------|-----------|
| `/camera/image_raw` | BEST_EFFORT | KEEP_LAST (1) | Tolerate dropped frames; never block on old images |
| `/scan` | BEST_EFFORT | KEEP_LAST (1) | Same — always want freshest scan |
| `/rover/detections` | RELIABLE | KEEP_LAST (10) | Fusion node needs every detection |
| `/rover/semantic/landmarks` | RELIABLE | KEEP_LAST (10) | Navigator must not miss updates |
| `/map` | RELIABLE | TRANSIENT_LOCAL | New subscribers receive last map immediately |

### Parameter Architecture

All runtime-tunable parameters are declared with `declare_parameter()` in each node and loaded from `rover_params.yaml` through the launch system. This enables:

```bash
# Live parameter update without restarting
ros2 param set /fusion_node merge_radius_m 0.3
```

---

## 4. Perception Engine — YOLOv8 + TensorRT

### Model Selection

YOLOv8 Nano (`yolov8n`) is chosen for the edge target:

| Model | Parameters | Jetson FP16 FPS | mAP50 (COCO) |
|-------|-----------|-----------------|--------------|
| yolov8n | 3.2 M | ~45 FPS | 37.3 |
| yolov8s | 11.2 M | ~25 FPS | 44.9 |
| yolov8m | 25.9 M | ~12 FPS | 50.2 |

For most indoor navigation use cases (person, chair, bottle, etc.), `yolov8n` at 37.3 mAP is sufficient. For higher-stakes applications, `yolov8s` still meets the 30 FPS requirement.

### TensorRT FP16 Pipeline

```
yolov8n.pt (PyTorch checkpoint)
    │
    ▼ model.export(format='onnx', imgsz=640, simplify=True)
yolov8n.onnx (simplified ONNX graph)
    │
    ▼ trtexec --onnx=... --fp16 --saveEngine=...
yolov8n.engine (TRT serialized engine, ~16 MB)
    │
    ▼ Runtime: deserialize → allocate GPU buffers → execute
Latency: ~22 ms/frame (FP32: ~85 ms/frame on same hardware)
```

**Why FP16?** FP16 halves memory bandwidth requirements and enables Tensor Core acceleration on Ampere/Orin GPUs. Accuracy loss on YOLOv8 detection tasks is typically <1% mAP.

### Inference Loop (TensorRTEngine)

```python
# One-time setup (constructor)
engine = trt.Runtime(logger).deserialize_cuda_engine(engine_bytes)
context = engine.create_execution_context()
h_input  = cuda.pagelocked_empty(...)  # pinned host memory
h_output = cuda.pagelocked_empty(...)
d_input  = cuda.mem_alloc(h_input.nbytes)
d_output = cuda.mem_alloc(h_output.nbytes)
stream   = cuda.Stream()

# Per-frame (< 25 ms total)
#   1. Letterbox + normalize: BGR uint8 → RGB float16 CHW
#   2. Async H→D copy (pinned memory → GPU)
#   3. TRT execute_async_v3 (non-blocking)
#   4. Async D→H copy
#   5. stream.synchronize()
#   6. Decode output tensor (84, 8400) → bounding boxes → NMS
```

### Output Tensor Decoding

YOLOv8 ONNX export produces a `(1, 84, 8400)` tensor:
- First 4 rows: `[cx, cy, w, h]` per anchor
- Rows 4–83: per-class scores (80 COCO classes)

Post-processing:
1. `argmax` over class dimension → `class_id`, `confidence`
2. Threshold by `conf_threshold` (0.45)
3. Convert `xywh → xyxy` and undo letterbox padding
4. Apply `cv2.dnn.NMSBoxes` with `nms_threshold` (0.50)

### Preprocessing — Letterbox

```
Original (W×H) → resize keeping aspect ratio → new (nW × nH)
Canvas of (640 × 640, value=114) ← paste at (pad_x, pad_y)
Normalize: [0,255] → [0,1]
CHW layout for TRT input
```

Scale and offset are stored to undo the transform during postprocessing.

---

## 5. Sensor Fusion — The Core Algorithm

This is the mathematical centrepiece of the project.

### Coordinate Frames

| Frame | Origin | X direction | Z direction |
|-------|--------|-------------|-------------|
| Camera optical | Camera centre | Right | Forward (optical axis) |
| Robot (base_link) | Robot centre floor | Forward | Up |
| Odom | Initial pose | Forward | Up |
| Map | SLAM origin | East / Forward | Up |

### Step-by-Step Pipeline

#### Step 1 — Bounding Box Centroid

```python
u = (x_min + x_max) / 2.0   # horizontal pixel
v = (y_min + y_max) / 2.0   # vertical pixel
```

The centroid is a reasonable approximation for the object's 2D position in the image, assuming the object subtends a modest solid angle from the camera.

#### Step 2 — Pixel → Camera-Frame Bearing

The pinhole camera model maps pixel `(u, v)` to a normalised point on the image plane:

```
x_n = (u - c_x) / f_x
y_n = (v - c_y) / f_y
```

The bearing vector in camera optical frame is:

```
bearing_cam = [x_n, y_n, 1.0]^T  (then normalised to unit length)
```

This vector points from the camera centre through the detected object.

#### Step 3 — Camera → Robot Frame Rotation

The extrinsic transform `R_cam→robot` is a 3×3 rotation matrix composed of:

1. **Axis realignment**: Camera optical frame (Z=forward, X=right, Y=down) → Robot frame (X=forward, Y=left, Z=up).
2. **Yaw rotation**: Horizontal alignment offset measured during calibration.
3. **Pitch rotation**: Camera downward tilt (typically −5°).

```python
R_cam_to_robot = R_yaw @ R_pitch @ R_axis_align
bearing_robot = R_cam_to_robot @ bearing_cam
```

#### Step 4 — Horizontal Angle

```python
theta = atan2(bearing_robot[1], bearing_robot[0])
```

This is the angle in the robot's horizontal plane (XY plane) pointing toward the detected object.

#### Step 5 — LiDAR Scan Index

The RPLiDAR publishes `sensor_msgs/LaserScan` with:
- `angle_min`, `angle_max`: scan arc (typically −π to +π)
- `angle_increment`: radians between consecutive rays
- `ranges[]`: distance in metres per ray

```python
idx = round((theta - angle_min) / angle_increment)
idx = clip(idx, 0, len(ranges) - 1)
```

#### Step 6 — Range Extraction with Median Filter

A single laser ray can return `nan` (no return) or a spurious value from specular reflection. A 7-ray median window (±3 around the computed index) reduces noise:

```python
window = [r for r in ranges[idx-3:idx+4]
          if not isnan(r) and range_min <= r <= range_max]
r = median(window)  # None if all invalid
```

#### Step 7 — Polar to Cartesian (Robot Frame)

```python
x_robot = r * cos(theta)   # forward (m)
y_robot = r * sin(theta)   # left (m)
```

#### Step 8 — Transform to Map Frame

```python
# 2D rigid transform: (x_robot, y_robot) → (x_map, y_map)
# Using TF2 lookup: base_link → map at detection timestamp

yaw = atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y² + q.z²))

x_map = cos(yaw)*x_robot - sin(yaw)*y_robot + t.x
y_map = sin(yaw)*x_robot + cos(yaw)*y_robot + t.y
```

The TF lookup uses the timestamp from the detection message header to account for any lag between image capture and fusion processing (bounded to 100 ms).

### Failure Modes and Mitigations

| Failure | Cause | Mitigation |
|---------|-------|------------|
| NaN range | No LiDAR return (glass, mirror) | Median window; drop detection |
| Stale scan | Scan >150 ms old | Age check; skip fusion frame |
| TF unavailable | SLAM not converged | 100 ms timeout; drop detection |
| Near-field return | Object <30 cm (LiDAR minimum) | `min_range_m` floor rejection |
| Large yaw offset | Camera misaligned | Calibration procedure |
| Glass wall | Camera detects object behind glass, LiDAR passes through | Out-of-scope; future work |

---

## 6. Landmark Tracking

### Data Structure

Each `LandmarkEntry` stores:
- `class_label`: string from YOLO
- `x_map`, `y_map`: EMA-fused map position
- `confidence`: max observed confidence
- `observation_count`: total merges (proxy for certainty)
- `first_seen`, `last_seen`: Unix timestamps
- `landmark_id`: 8-char UUID fragment

### Spatial Deduplication

On each `observe(class_label, x, y, conf, range)` call:

1. Search existing landmarks of the same class.
2. Find the nearest within `merge_radius_m` (default 0.5 m).
3. If found → EMA update. If not → create new entry.

The 0.5 m radius was chosen so that:
- A person standing still and slowly drifting in YOLO position estimates merges correctly.
- Two separate people >0.5 m apart create separate entries.

### Exponential Moving Average (EMA) Position Fusion

```
x_new_stored = α · x_observed  +  (1 − α) · x_stored
```

With α = 0.3, after 5 consistent observations the stored position has converged to within ~5% of the true position from any starting point. This filters LiDAR noise and SLAM drift without requiring a full Kalman filter.

### Aging and Pruning

A timer fires at 0.2 Hz (configurable). Landmarks not seen for `max_landmark_age_s` (30 s) are removed. This:
- Handles moved objects (e.g. a chair pushed aside).
- Keeps memory bounded in continuous operation.
- Can be tuned longer for static environments.

---

## 7. SLAM — slam_toolbox

### Why slam_toolbox?

`slam_toolbox` provides:
- **Async mode**: scan processing does not block ROS callbacks, critical for fusion latency.
- **Lifelong mapping**: can serialize and reload maps, enabling the rover to resume from a known map.
- **Loop closure**: Ceres-based graph solver corrects drift when returning to visited areas.
- **Nav2 integration**: publishes `/map` and TF tree (`map→odom→base_link`) directly consumable by Nav2.

### Configuration Highlights

```yaml
minimum_travel_distance: 0.10   # Only process scans after 10 cm movement
minimum_travel_heading: 0.10    # Or 5.7° rotation
do_loop_closing: true
loop_match_minimum_chain_size: 10  # Require 10 consecutive matching scans
resolution: 0.05                # 5 cm per cell
max_laser_range: 10.0           # Clip returns beyond 10 m
```

The `minimum_travel_distance/heading` filters prevent the SLAM from wasting CPU reprocessing scans when the robot is stationary.

### TF Tree

```
map ──(slam_toolbox)──► odom ──(esp32_bridge_node)──► base_link
                                                         │
                                            ┌────────────┴──────────┐
                                         (static TF)            (static TF)
                                           laser                camera_link
```

The `odom→base_link` transform is published by `esp32_bridge_node` from wheel encoder odometry. SLAM corrects drift by publishing the `map→odom` correction transform at ~50 Hz.

---

## 8. Navigation — Nav2 + Semantic Goals

### Nav2 Stack Components

| Component | Role |
|-----------|------|
| `bt_navigator` | Behavior Tree executor for goal handling |
| `planner_server` | Global path planning (NavFn A* / Dijkstra) |
| `controller_server` | Local trajectory following (DWB planner) |
| `local_costmap` | Obstacle-inflated rolling window for local planning |
| `global_costmap` | Full map costmap for global planning |
| `behavior_server` | Recovery behaviors: spin, back-up, wait |

### Semantic Navigator Design

`SemanticNavigator` is a thin ROS 2 service wrapper over Nav2's `NavigateToPose` action:

```
Service call: NavigateToClass
    │
    ▼
GetSemanticLandmarks service call → most-observed landmark of requested class
    │
    ├── Found → compute approach pose → NavigateToPose action goal → Nav2
    │
    └── Not found + explore_if_not_found=True
            │
            ▼
        Frontier waypoint loop (background thread)
            ├── Send exploration waypoint → Nav2
            ├── Sleep 8 s (rover navigates)
            ├── Re-check landmark map
            └── Repeat until found or timeout (300 s)
```

### Approach Pose Computation

The goal pose is placed `approach_distance` metres in front of the landmark, with the robot facing the landmark:

```python
goal_x = landmark.x_map - approach_distance  # default: along map X-axis
goal_y = landmark.y_map
yaw    = atan2(landmark.y_map - goal_y, landmark.x_map - goal_x)
```

A production system would compute the approach vector from the robot's current position. The default implementation backs off along map X — straightforward and effective when the robot is roughly aligned with its targets.

### DWB Planner Tuning for Differential Drive

Key parameters for a small differential-drive rover:

```yaml
max_vel_x: 0.26        # m/s (conservative for indoor)
max_vel_theta: 1.0     # rad/s
acc_lim_x: 2.5         # m/s²
decel_lim_x: -2.5
robot_radius: 0.17     # m (inflation buffer)
xy_goal_tolerance: 0.25
yaw_goal_tolerance: 0.25
```

---

## 9. Hardware Bridge — ESP32 Serial Protocol

### Protocol Specification

All messages are newline-terminated ASCII, comma-delimited:

```
Direction        Format                          Example
─────────────────────────────────────────────────────────
Jetson → ESP32:  "CMD,<v_lin>,<v_ang>\n"         "CMD,0.2000,-0.3000\n"
ESP32  → Jetson: "ODO,<l_ticks>,<r_ticks>,<dt>\n" "ODO,42,-40,100\n"
ESP32  → Jetson: "IMU,ax,ay,az,gx,gy,gz\n"       "IMU,0.12,-0.05,9.78,0.001,0.002,-0.003\n"
ESP32  → Jetson: "ERR,<message>\n"               "ERR,i2c_timeout\n"
```

**Baud rate:** 115200. At 115200 bps, an ODO line (~20 chars) takes ~1.7 ms. At 10 Hz, this represents <2% bus utilisation.

### Odometry Calculation (Differential Drive)

```
# Per 100 ms tick
d_left  = left_ticks  × (2π × r_wheel) / ticks_per_rev
d_right = right_ticks × (2π × r_wheel) / ticks_per_rev

d_center = (d_right + d_left) / 2
d_theta  = (d_right - d_left) / wheel_base

# Integrate pose
x     += d_center × cos(θ + d_theta/2)
y     += d_center × sin(θ + d_theta/2)
θ     += d_theta
```

Using the **mid-point rule** (integrating at θ + dθ/2) reduces error compared to Euler integration, especially through turns.

### Inverse Kinematics (Jetson → ESP32)

```cpp
v_left  = (v_lin - v_ang × wheel_base/2) / wheel_radius  // rad/s
v_right = (v_lin + v_ang × wheel_base/2) / wheel_radius  // rad/s

// Normalise to [–1, +1] and scale to PWM
pwm = |v_wheel| / max_wheel_speed × 255
```

### Watchdog Behaviour

If no `CMD` message arrives for 500 ms:
- ESP32 sets `g_cmd_v_lin = 0`, `g_cmd_v_ang = 0`.
- Both motor PWM channels set to 0.
- All H-bridge direction pins set LOW (freewheeling, not braking).

This prevents runaway motion if the Jetson crashes or the serial cable disconnects.

---

## 10. Camera–LiDAR Extrinsic Calibration

### Why Calibration Matters

A 5° yaw misalignment between the camera and LiDAR causes a position error of:

```
Δy = r × sin(5°) ≈ 0.087 × r
```

At 3 m range: **26 cm error**. The 30 cm position accuracy target requires yaw error below ~5°.

### Calibration Procedure

1. Place a **retroreflective target** (or bright coloured object, or ArUco marker) at 1–3 m, centred in the camera frame.
2. Run `camera_lidar_calibration.py`.
3. The tool:
   a. Detects the target centroid in the image (`cv2.inRange` HSV blob, or ArUco detector).
   b. Computes the camera-frame horizontal angle: `θ_cam = atan((u − c_x) / f_x)`.
   c. Finds the minimum-range LiDAR return within ±60° of forward.
   d. Reports `offset = θ_cam − θ_lidar` in degrees.
4. Collect 30+ samples. Use the **median** (robust to outliers from spurious returns).
5. Set `camera_yaw_offset_deg: <median>` in `rover_params.yaml`.

### Camera Intrinsic Calibration

Use the standard ROS 2 checkerboard calibration:

```bash
ros2 run camera_calibration cameracalibrator \
    --size 8x6 --square 0.025 \
    image:=/camera/image_raw \
    camera:=/camera
```

This produces `camera_info.yaml` with `K` (3×3 intrinsic matrix), `D` (distortion coefficients), and `P` (projection matrix). The fusion node reads `K` from the `sensor_msgs/CameraInfo` topic.

---

## 11. TensorRT Export and Optimization

### Export Pipeline

```python
# Step 1: PyTorch → ONNX (simplified)
model = YOLO('yolov8n.pt')
model.export(format='onnx', imgsz=640, simplify=True, opset=17)

# Step 2: ONNX → TRT engine (FP16)
# trtexec --onnx=yolov8n_640.onnx --fp16 --saveEngine=yolov8n_640.engine
```

**`simplify=True`** runs `onnxsim` to fold constants, eliminate dead nodes, and standardise op sets — typically reduces ONNX graph complexity by 30–40%.

**`opset=17`** is the highest opset TensorRT 8.6+ handles reliably. Lower opsets may trigger fallbacks.

### FP16 vs INT8

| Precision | Advantages | Disadvantages |
|-----------|-----------|---------------|
| FP32 | Highest accuracy | ~2× slower, ~2× more memory |
| FP16 | 2× faster, half memory, minimal accuracy loss | Requires Tensor Core GPU |
| INT8 | Up to 4× faster | Requires calibration dataset, ~2–5% mAP drop |

FP16 is chosen as the best accuracy/speed trade-off. INT8 is supported by the export script (`--int8` flag) but requires a representative calibration dataset.

### Memory Management

TensorRT requires pinned (page-locked) host memory for zero-copy DMA to GPU:

```python
h_input  = cuda.pagelocked_empty(n, dtype=np.float16)  # GPU DMA source
d_input  = cuda.mem_alloc(h_input.nbytes)               # GPU buffer
stream   = cuda.Stream()                                 # async CUDA stream

# Copy: pinned host → GPU
cuda.memcpy_htod_async(d_input, h_input, stream)
context.execute_async_v3(stream_handle=stream.handle)
cuda.memcpy_dtoh_async(h_output, d_output, stream)
stream.synchronize()
```

Pinned memory enables DMA transfer without a kernel-to-user-space copy. For a 640×640×3 FP16 input, this saves ~6 MB of memcpy overhead per frame.

---

## 12. Custom ROS 2 Messages

### Design Decisions

**`Detection2D` vs `vision_msgs/Detection2D`:** The standard `vision_msgs` message was not used because it lacks `inference_latency_ms` (needed for pipeline profiling) and uses a different bounding box representation requiring conversion. A minimal custom message was preferred.

**`SemanticLandmark.observation_count`:** This field serves as a confidence proxy. A landmark seen only once could be a false positive; one seen 10+ times is reliably present. `SemanticNavigator` uses the most-observed landmark when multiple instances of a class exist.

**`landmark_id`:** An 8-character UUID fragment is sufficient for deduplication within a session. Full UUIDs are not needed since landmarks are not persisted across reboots in the current implementation (future work: serialize to YAML).

### Message Dependencies

```
rover_msgs/SemanticLandmark
  ├── std_msgs/Header
  ├── geometry_msgs/Point
  └── builtin_interfaces/Time

rover_msgs/SemanticMap
  ├── std_msgs/Header
  └── rover_msgs/SemanticLandmark[]

rover_msgs/Detection2DArray
  ├── std_msgs/Header
  └── rover_msgs/Detection2D[]
```

---

## 13. Testing Strategy

### Test Pyramid

```
         ▲
        /·\      Integration tests (4)
       /···\     — Full detection→fusion→landmark pipeline
      /·····\    — No ROS, no GPU
     /·······\
    /·········\  Unit tests (42)
   /···········\ — projection_math: pixel↔bearing↔scan↔pose math
  /·············\— landmark_tracker: EMA, dedup, aging, pruning
 /···············\— tensorrt_engine: preprocessing, postprocessing mocks
```

### Running Tests

```bash
# No ROS 2 installation required
python3 -m venv .venv && source .venv/bin/activate
pip install pytest numpy opencv-python-headless

pytest tests/ -v                        # all 46 tests
pytest tests/unit/test_projection_math.py -v  # just fusion math
pytest tests/ -k "forward"              # keyword filter
```

### Test Design Principles

1. **No external dependencies in unit tests:** TensorRT and pycuda are guarded with `try/except ImportError`; tests use mock engines.
2. **Synthetic sensor data:** The `flat_scan` fixture creates a perfect 360-ray scan with all returns at 2.0 m — predictable, fast, no file I/O.
3. **Mathematical property tests:** Not just "does it run" but "is the geometry correct" — e.g., `test_principal_point_gives_forward_bearing` asserts that the optical axis projects to bearing `[0, 0, 1]`.
4. **Edge cases first:** NaN ranges, empty scan windows, below-floor ranges, TF timeout — all tested before happy-path.

### Adding New Tests

For a new sensor fusion feature:

```python
# tests/unit/test_projection_math.py

def test_my_new_case(self, intrinsics, extrinsics_aligned, flat_scan):
    result = project_detection_to_3d(
        x_min=..., y_min=..., x_max=..., y_max=...,
        intrinsics=intrinsics,
        extrinsics=extrinsics_aligned,
        **flat_scan,
    )
    assert result.is_valid
    assert abs(result.x_robot - expected_x) < tolerance
```

---

## 14. Docker Deployment

### Image Layers

```
nvcr.io/nvidia/l4t-ros:humble-ros-base-r36.x.0   # NVIDIA base (JetPack 6 + ROS 2)
    ├── System packages (slam_toolbox, nav2, rplidar_ros, cv_bridge, ...)
    ├── Python AI stack (ultralytics, opencv, pycuda)
    ├── ROS 2 workspace source copy
    ├── rosdep install + colcon build
    └── Entrypoint (source setup.bash + CMD)
```

### Volumes

| Volume | Purpose |
|--------|---------|
| `/opt/rover/models` | TRT engine files — persist between container restarts |
| `/tmp` | SLAM map serialization (slam_toolbox `map_file_name`) |

### Device Passthrough

```yaml
devices:
  - /dev/ttyUSB0:/dev/ttyUSB0   # RPLiDAR
  - /dev/ttyUSB1:/dev/ttyUSB1   # ESP32
  - /dev/video0:/dev/video0     # USB camera
```

The `privileged: true` flag is required for NVIDIA runtime access on Jetson. For production, tighten this using `device_cgroup_rules`.

### Multi-Architecture Builds

```bash
# Build for Jetson (aarch64) from x86 development machine
docker buildx build --platform linux/arm64 \
    --build-arg BASE=nvcr.io/nvidia/l4t-jetpack:r36.3.0 \
    -f docker/Dockerfile.jetson \
    -t registry.example.com/semantic-slam-rover:arm64 \
    --push .
```

Requires `qemu-user-static` and a buildx builder with ARM support.

---

## 15. Performance Analysis

### Latency Budget (10 Hz fusion cycle)

```
100 ms budget (10 Hz LiDAR)
 ├── Camera frame capture       ~0 ms  (async, camera at 30 Hz)
 ├── YOLOv8 TRT inference      ~22 ms  (GPU, parallel with LiDAR)
 ├── Scan age check             ~0 ms
 ├── TF lookup                  ~1 ms
 ├── Projection math            ~0.5 ms (per detection, n≤10)
 ├── Landmark tracker update    ~0.1 ms
 └── Publish landmarks          ~0.5 ms
 Total: ~25 ms (25% of 100 ms budget)
```

### Jetson Power

```
Idle (ROS 2 running, no inference):  ~4 W
SLAM only:                           ~5 W
SLAM + YOLOv8 TRT FP16:             ~9 W
SLAM + YOLOv8 + Nav2 + fusion:     ~11 W
```

At 11 W, a 3S LiPo 2200 mAh (81.48 Wh usable at 80% DoD) powers the Jetson for ~6 hours. Motors draw separately from the motor rail.

### Landmark Position Error Analysis

Sources of error in order of magnitude:

| Source | Typical Magnitude | Mitigation |
|--------|-------------------|------------|
| SLAM drift (pre-loop-closure) | 0.05–0.15 m | Loop closure, good initial odom |
| LiDAR range noise (RPLiDAR A1) | ±0.02 m | 7-ray median window |
| Centroid to object centre error | 0–0.1 m | Wider detections are less accurate |
| Yaw calibration residual | r × sin(Δθ) | Careful calibration |
| EMA convergence lag | ~0.05 m | α=0.3, converges in ~5 observations |

Combined RSS at 3 m range with good calibration: **~0.15–0.20 m**, well within the 0.30 m target.

---

## 16. Known Limitations and Future Work

### Current Limitations

| Limitation | Impact | Root Cause |
|------------|--------|------------|
| 2D LiDAR only | Cannot detect objects above/below scan plane | Planar RPLiDAR |
| Static landmark map | Moved objects linger until age-out (30 s) | No object tracking |
| Single-plane fusion | Vertical position always set to z=0 | 2D scan |
| Hardcoded exploration waypoints | Suboptimal exploration coverage | Placeholder implementation |
| No map persistence | Landmarks lost on restart | In-memory only |

### Roadmap

**Near-term (low effort, high value):**
- [ ] **3D LiDAR support** (Velodyne VLP-16 or Livox Mid-360): extend fusion to use full point cloud, enabling vertical position estimates.
- [ ] **Map serialization**: save/load `LandmarkTracker` state to YAML at shutdown/startup.
- [ ] **Frontier-based exploration**: replace hardcoded waypoints with proper frontier detection from the occupancy grid.

**Medium-term:**
- [ ] **Object tracking**: integrate DeepSORT or ByteTrack to associate `track_id` across frames, enabling velocity estimation and trajectory prediction for dynamic objects.
- [ ] **Active calibration**: online estimation of extrinsic yaw from mutual information between image edges and LiDAR edge returns.
- [ ] **Multi-class navigation policy**: define priority ordering (e.g., always navigate to "person" before "chair") and conflict resolution.

**Long-term:**
- [ ] **3D semantic mapping**: voxel grid (OctoMap) with per-voxel class distribution, enabling volumetric semantic queries.
- [ ] **Language interface**: wrap `SemanticNavigator` with an LLM that translates natural-language commands to `NavigateToClass` service calls.
- [ ] **Multi-robot coordination**: shared semantic map via a central `SemanticMapServer` using DDS multi-cast.

---

*For implementation questions, open a GitHub issue. For hardware replication, see the [BOM and wiring diagram](docs/hardware/) (coming soon).*
