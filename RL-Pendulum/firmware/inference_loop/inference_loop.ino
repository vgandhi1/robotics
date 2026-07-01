/**
 * inference_loop.ino — RL-Pendulum Edge Inference on ESP32
 *
 * Runs a TensorFlow Lite Micro INT8 MLP at 100 Hz to balance an inverted
 * pendulum robot using a PPO policy trained in simulation.
 *
 * Control loop (10 ms budget):
 *   1. Read MPU-6050 via I2C         (~1.0 ms)
 *   2. Complementary filter (pitch)  (~0.1 ms)
 *   3. Read encoder counts           (~0.2 ms)
 *   4. Normalize state vector        (~0.1 ms)
 *   5. TFLite Invoke()               (~6–9 ms)
 *   6. Clip + write PWM              (~0.1 ms)
 *   Total:                          < 10 ms ✓
 *
 * Required Arduino libraries:
 *   - TensorFlowLite_ESP32 (v0.9.0+)
 *   - MPU6050 by Electronic Cats (not used directly — we use raw I2C)
 *
 * Generated model header:
 *   rl_policy_data.h  — produced by: xxd -i export/model.tflite > rl_policy_data.h
 *
 * Serial output (115200 baud, CSV):
 *   pitch,pitch_rate,lw_speed,rw_speed,action,loop_ms
 */

#include <Wire.h>
#include "imu_driver.h"
#include "motor_driver.h"

// TFLite Micro headers
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Generated model data header (produced by export/quantize.py → generate_c_header)
// If not yet generated, run: bash scripts/export_model.sh
#include "rl_policy_data.h"

// ─── Configuration ───────────────────────────────────────────────────────────

// Control loop timing
#define CTRL_PERIOD_MS     10    // 100 Hz
#define FALL_ANGLE_RAD     0.45f // Emergency stop threshold (< 0.5 rad limit)

// Observation normalization limits (must match training config)
#define OBS_PITCH_LIMIT      0.5f
#define OBS_PITCH_RATE_LIMIT 10.0f
#define OBS_WHEEL_SPD_LIMIT  20.0f

// Encoder pins
#define ENC_L_A   34
#define ENC_L_B   35
#define ENC_R_A   32
#define ENC_R_B   33

// ─── TFLite Micro setup ──────────────────────────────────────────────────────

// Tensor arena: allocate enough SRAM for model activations + buffers.
// For a 4×64×64×1 INT8 MLP, ~16 KB is sufficient.
// Increase if the interpreter reports "AllocateTensors() failed".
constexpr int kTensorArenaSize = 16 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

static const tflite::Model*           tfl_model    = nullptr;
static tflite::AllOpsResolver         tfl_resolver;
static tflite::MicroInterpreter*      tfl_interp   = nullptr;
static TfLiteTensor*                  tfl_input    = nullptr;
static TfLiteTensor*                  tfl_output   = nullptr;

// ─── Encoder state ───────────────────────────────────────────────────────────
static volatile int32_t enc_l_count = 0;
static volatile int32_t enc_r_count = 0;
static int32_t enc_l_prev = 0, enc_r_prev = 0;
static float lw_speed_rads = 0.0f;
static float rw_speed_rads = 0.0f;

// Encoder: N20 motor + 20 CPR encoder + gear ratio ~30:1 ≈ 600 CPR at wheel
#define ENCODER_CPR        600
#define TWO_PI             6.28318530f

IRAM_ATTR void enc_l_isr() { enc_l_count += digitalRead(ENC_L_B) ? -1 : 1; }
IRAM_ATTR void enc_r_isr() { enc_r_count += digitalRead(ENC_R_B) ? -1 : 1; }

void encoder_init() {
    pinMode(ENC_L_A, INPUT_PULLUP); pinMode(ENC_L_B, INPUT_PULLUP);
    pinMode(ENC_R_A, INPUT_PULLUP); pinMode(ENC_R_B, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(ENC_L_A), enc_l_isr, RISING);
    attachInterrupt(digitalPinToInterrupt(ENC_R_A), enc_r_isr, RISING);
    Serial.println("[ENC] Encoder interrupts attached");
}

void encoder_update_speeds(float dt_s) {
    int32_t dl = enc_l_count - enc_l_prev;
    int32_t dr = enc_r_count - enc_r_prev;
    enc_l_prev = enc_l_count;
    enc_r_prev = enc_r_count;

    // counts/s → rad/s
    lw_speed_rads = ((float)dl / ENCODER_CPR) * TWO_PI / dt_s;
    rw_speed_rads = ((float)dr / ENCODER_CPR) * TWO_PI / dt_s;
}

// ─── TFLite initialization ───────────────────────────────────────────────────

bool tflite_init() {
    // Map the model data from the generated header
    tfl_model = tflite::GetModel(model_tflite);
    if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("[TF] Schema version mismatch: model=%d, runtime=%d\n",
                      tfl_model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }

    tfl_interp = new tflite::MicroInterpreter(
        tfl_model, tfl_resolver, tensor_arena, kTensorArenaSize
    );

    if (tfl_interp->AllocateTensors() != kTfLiteOk) {
        Serial.println("[TF] AllocateTensors() failed — increase kTensorArenaSize");
        return false;
    }

    tfl_input  = tfl_interp->input(0);
    tfl_output = tfl_interp->output(0);

    Serial.printf("[TF] Model loaded. Arena used: %u / %u bytes\n",
                  tfl_interp->arena_used_bytes(), kTensorArenaSize);
    Serial.printf("[TF] Input  type=%d dims=[%d,%d]\n",
                  tfl_input->type,
                  tfl_input->dims->data[0], tfl_input->dims->data[1]);
    Serial.printf("[TF] Output type=%d dims=[%d,%d]\n",
                  tfl_output->type,
                  tfl_output->dims->data[0], tfl_output->dims->data[1]);
    return true;
}

// ─── Policy inference ────────────────────────────────────────────────────────

/**
 * Run one forward pass through the policy network.
 *
 * Input:  normalized state vector [pitch, pitch_rate, lw_speed, rw_speed]
 *         all in [-1.0, 1.0]
 * Output: action in [-1.0, 1.0]
 *
 * For INT8 models, the framework handles quantization/dequantization
 * automatically via the scale/zero_point stored in the tensor metadata.
 */
float run_inference(float pitch_norm, float pitch_rate_norm,
                    float lw_norm, float rw_norm) {
    // Write normalized float inputs (TFLite handles quant internally)
    if (tfl_input->type == kTfLiteFloat32) {
        tfl_input->data.f[0] = pitch_norm;
        tfl_input->data.f[1] = pitch_rate_norm;
        tfl_input->data.f[2] = lw_norm;
        tfl_input->data.f[3] = rw_norm;
    } else {
        // INT8: quantize manually using scale/zero_point from tensor
        float scale    = tfl_input->params.scale;
        int32_t zp     = tfl_input->params.zero_point;
        float inputs[4] = {pitch_norm, pitch_rate_norm, lw_norm, rw_norm};
        for (int i = 0; i < 4; i++) {
            int q = (int)roundf(inputs[i] / scale) + zp;
            tfl_input->data.int8[i] = (int8_t)constrain(q, -128, 127);
        }
    }

    // Run the model
    if (tfl_interp->Invoke() != kTfLiteOk) {
        Serial.println("[TF] Invoke() failed");
        return 0.0f;
    }

    // Read output
    float action;
    if (tfl_output->type == kTfLiteFloat32) {
        action = tfl_output->data.f[0];
    } else {
        float scale = tfl_output->params.scale;
        int32_t zp  = tfl_output->params.zero_point;
        action = (tfl_output->data.int8[0] - zp) * scale;
    }

    return constrain(action, -1.0f, 1.0f);
}

// ─── Arduino entry points ────────────────────────────────────────────────────

void setup() {
    Serial.begin(115200);
    delay(500);
    Serial.println("\n===== RL-Pendulum Edge Inference =====");

    // I2C for IMU
    Wire.begin(21, 22);
    Wire.setClock(400000);  // 400 kHz Fast Mode

    // Subsystem init
    if (!imu_init()) { Serial.println("FATAL: IMU init failed"); while(1); }
    motor_init();
    encoder_init();

    // IMU gyro bias calibration (keep robot stationary)
    Serial.println("Calibrating gyro — hold robot stationary for 1 second...");
    delay(200);
    imu_calibrate_gyro();

    // TFLite model init
    if (!tflite_init()) { Serial.println("FATAL: TFLite init failed"); while(1); }

    Serial.println("\nStarting 100 Hz control loop...");
    Serial.println("pitch_deg,pitch_rate_rads,lw_spd_rads,rw_spd_rads,action,loop_ms");
}

void loop() {
    uint32_t loop_start = millis();

    // ── 1. Read IMU ──────────────────────────────────────────────────────────
    float pitch_rad, pitch_rate_rads;
    imu_update(&pitch_rad, &pitch_rate_rads);

    // ── 2. Read encoders ─────────────────────────────────────────────────────
    float dt = CTRL_PERIOD_MS * 0.001f;
    encoder_update_speeds(dt);

    // ── 3. Emergency stop if robot has fallen ────────────────────────────────
    if (fabsf(pitch_rad) > FALL_ANGLE_RAD) {
        motor_stop();
        Serial.printf("FALLEN  pitch=%.3f\n", pitch_rad);
        delay(CTRL_PERIOD_MS);
        return;
    }

    // ── 4. Normalize observations to [-1, 1] ─────────────────────────────────
    float pitch_norm      = pitch_rad       / OBS_PITCH_LIMIT;
    float pitch_rate_norm = pitch_rate_rads / OBS_PITCH_RATE_LIMIT;
    float lw_norm         = lw_speed_rads   / OBS_WHEEL_SPD_LIMIT;
    float rw_norm         = rw_speed_rads   / OBS_WHEEL_SPD_LIMIT;

    // Clip to valid range
    pitch_norm      = constrain(pitch_norm,      -1.0f, 1.0f);
    pitch_rate_norm = constrain(pitch_rate_norm, -1.0f, 1.0f);
    lw_norm         = constrain(lw_norm,         -1.0f, 1.0f);
    rw_norm         = constrain(rw_norm,         -1.0f, 1.0f);

    // ── 5. Run policy inference ──────────────────────────────────────────────
    float action = run_inference(pitch_norm, pitch_rate_norm, lw_norm, rw_norm);

    // ── 6. Command motors ────────────────────────────────────────────────────
    motor_set_action(action);

    // ── 7. Serial telemetry (every 5 loops = 50 ms to avoid UART bottleneck) ─
    static uint8_t telem_div = 0;
    if (++telem_div >= 5) {
        telem_div = 0;
        uint32_t loop_ms = millis() - loop_start;
        Serial.printf("%.3f,%.3f,%.3f,%.3f,%.4f,%u\n",
                      pitch_rad * 57.2958f,  // convert to degrees for readability
                      pitch_rate_rads,
                      lw_speed_rads,
                      rw_speed_rads,
                      action,
                      loop_ms);
    }

    // ── 8. Maintain 100 Hz rate ──────────────────────────────────────────────
    uint32_t elapsed = millis() - loop_start;
    if (elapsed < CTRL_PERIOD_MS) {
        delay(CTRL_PERIOD_MS - elapsed);
    } else {
        // Loop overran budget — log warning (at most every 100 loops)
        static uint16_t overrun_count = 0;
        if (++overrun_count % 100 == 1) {
            Serial.printf("[WARN] Loop overrun: %u ms (budget %u ms)\n",
                          elapsed, CTRL_PERIOD_MS);
        }
    }
}
