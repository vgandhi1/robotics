/**
 * imu_driver.h — MPU-6050 IMU driver for ESP32
 *
 * Provides:
 *  - I2C initialization of the MPU-6050 at 200 Hz output rate
 *  - Raw register reads for accelerometer and gyroscope
 *  - Complementary filter for pitch angle estimation
 *    (fuses accelerometer gravity vector with gyroscope integration)
 *
 * Complementary filter:
 *   pitch = alpha * (pitch + gyro_rate * dt) + (1 - alpha) * accel_pitch
 *   alpha = 0.98 (trust gyro more at high frequency, accel for DC correction)
 *
 * Coordinate convention:
 *   - Positive pitch  = robot leans forward
 *   - Zero pitch      = perfectly upright
 *
 * Wiring:
 *   MPU-6050 SDA → ESP32 GPIO21
 *   MPU-6050 SCL → ESP32 GPIO22
 *   MPU-6050 VCC → 3.3 V
 *   MPU-6050 GND → GND
 *   MPU-6050 AD0 → GND  (I2C address = 0x68)
 */

#pragma once
#include <Wire.h>

// ─── MPU-6050 register map ──────────────────────────────────────────────────
#define MPU6050_ADDR         0x68
#define MPU6050_PWR_MGMT_1   0x6B
#define MPU6050_SMPLRT_DIV   0x19
#define MPU6050_CONFIG       0x1A
#define MPU6050_GYRO_CONFIG  0x1B
#define MPU6050_ACCEL_CONFIG 0x1C
#define MPU6050_ACCEL_XOUT_H 0x3B
#define MPU6050_GYRO_XOUT_H  0x43

// ─── Sensor scale factors ───────────────────────────────────────────────────
// Accelerometer: ±2g range → 16384 LSB/g
#define ACCEL_SCALE  (1.0f / 16384.0f)
// Gyroscope: ±250 deg/s range → 131 LSB/(deg/s)
#define GYRO_SCALE   (1.0f / 131.0f)
#define DEG_TO_RAD   (3.14159265f / 180.0f)

// ─── Complementary filter coefficient ──────────────────────────────────────
#define COMP_ALPHA   0.98f
#define CTRL_DT      0.01f   // 10 ms control period

// ─── Gyro bias estimate (calibrated at startup) ─────────────────────────────
static float _gyro_bias_rads = 0.0f;
static float _pitch_rad      = 0.0f;

// ─── Helpers ────────────────────────────────────────────────────────────────
static inline void _mpu_write(uint8_t reg, uint8_t val) {
    Wire.beginTransmission(MPU6050_ADDR);
    Wire.write(reg);
    Wire.write(val);
    Wire.endTransmission();
}

static inline int16_t _read_int16(uint8_t reg) {
    Wire.beginTransmission(MPU6050_ADDR);
    Wire.write(reg);
    Wire.endTransmission(false);
    Wire.requestFrom((uint8_t)MPU6050_ADDR, (uint8_t)2, (uint8_t)true);
    return (int16_t)((Wire.read() << 8) | Wire.read());
}

// ─── Public API ─────────────────────────────────────────────────────────────

/**
 * Initialize the MPU-6050 at 200 Hz sample rate.
 * Must be called after Wire.begin().
 *
 * Returns true on success, false if the device does not ACK.
 */
bool imu_init() {
    // Wake up MPU-6050 (clear sleep bit), use gyro X as clock source
    _mpu_write(MPU6050_PWR_MGMT_1, 0x01);
    delay(50);

    // Sample rate divider: SMPLRT_DIV = 4 → 1000/(1+4) = 200 Hz
    _mpu_write(MPU6050_SMPLRT_DIV, 0x04);

    // DLPF = 3 → 44 Hz bandwidth (reduces vibration noise, adds ~4.9 ms lag)
    _mpu_write(MPU6050_CONFIG, 0x03);

    // Gyro full-scale: ±250 deg/s
    _mpu_write(MPU6050_GYRO_CONFIG, 0x00);

    // Accel full-scale: ±2g
    _mpu_write(MPU6050_ACCEL_CONFIG, 0x00);

    // Verify device responded
    Wire.beginTransmission(MPU6050_ADDR);
    uint8_t err = Wire.endTransmission();
    if (err != 0) {
        Serial.println("[IMU] ERROR: MPU-6050 not found on I2C bus");
        return false;
    }
    Serial.println("[IMU] MPU-6050 initialized at 200 Hz");
    return true;
}

/**
 * Calibrate gyroscope bias by averaging 500 samples at rest.
 * Call once at startup while the robot is stationary.
 */
void imu_calibrate_gyro() {
    Serial.print("[IMU] Calibrating gyro bias");
    float sum = 0.0f;
    const int N = 500;
    for (int i = 0; i < N; i++) {
        int16_t raw = _read_int16(MPU6050_GYRO_XOUT_H);
        sum += raw * GYRO_SCALE * DEG_TO_RAD;
        delay(2);
        if (i % 100 == 0) Serial.print(".");
    }
    _gyro_bias_rads = sum / (float)N;
    Serial.printf("\n[IMU] Gyro bias: %.6f rad/s\n", _gyro_bias_rads);
}

/**
 * Read IMU and update the complementary filter estimate.
 *
 * @param[out] pitch_rad       Filtered pitch angle (rad), positive = forward lean
 * @param[out] pitch_rate_rads Bias-corrected gyro pitch rate (rad/s)
 */
void imu_update(float* pitch_rad_out, float* pitch_rate_rads_out) {
    // ── Read raw accel (X, Y, Z) ─────────────────────────────────────────────
    Wire.beginTransmission(MPU6050_ADDR);
    Wire.write(MPU6050_ACCEL_XOUT_H);
    Wire.endTransmission(false);
    Wire.requestFrom((uint8_t)MPU6050_ADDR, (uint8_t)6, (uint8_t)true);

    int16_t ax_raw = (int16_t)((Wire.read() << 8) | Wire.read());
    int16_t ay_raw = (int16_t)((Wire.read() << 8) | Wire.read());
    int16_t az_raw = (int16_t)((Wire.read() << 8) | Wire.read());

    float ax = ax_raw * ACCEL_SCALE;
    float ay = ay_raw * ACCEL_SCALE;
    float az = az_raw * ACCEL_SCALE;

    // ── Compute pitch from accelerometer (atan2 of gravity projection) ───────
    float accel_pitch = atan2f(ax, sqrtf(ay * ay + az * az));

    // ── Read gyro pitch rate (rotation about Y axis) ─────────────────────────
    int16_t gy_raw = _read_int16(MPU6050_GYRO_XOUT_H + 2);  // GYRO_YOUT_H
    float gyro_rate = gy_raw * GYRO_SCALE * DEG_TO_RAD - _gyro_bias_rads;

    // ── Complementary filter ─────────────────────────────────────────────────
    _pitch_rad = COMP_ALPHA * (_pitch_rad + gyro_rate * CTRL_DT)
               + (1.0f - COMP_ALPHA) * accel_pitch;

    *pitch_rad_out      = _pitch_rad;
    *pitch_rate_rads_out = gyro_rate;
}
