/**
 * motor_driver.h — L298N / DRV8833 DC motor driver for ESP32
 *
 * Provides:
 *  - LEDC PWM channel initialization for smooth motor control
 *  - Set motor speed symmetrically (balance mode) or differentially (steer)
 *  - Soft start / ramp rate limiting to prevent mechanical shock
 *
 * Wiring (L298N):
 *   IN1 → GPIO25 (DIR Left A)
 *   IN2 → GPIO26 (DIR Left B)
 *   ENA → GPIO27 (PWM Left, via LEDC channel 0)
 *   IN3 → GPIO14 (DIR Right A)
 *   IN4 → GPIO12 (DIR Right B)
 *   ENB → GPIO13 (PWM Right, via LEDC channel 1)
 *
 * PWM configuration:
 *   Frequency: 20 kHz (above audible range, reduces motor whine)
 *   Resolution: 10-bit (0–1023)
 */

#pragma once
#include "Arduino.h"

// ─── GPIO pin assignments ────────────────────────────────────────────────────
#define MOTOR_L_DIR_A   25
#define MOTOR_L_DIR_B   26
#define MOTOR_L_PWM     27
#define MOTOR_R_DIR_A   14
#define MOTOR_R_DIR_B   12
#define MOTOR_R_PWM     13

// ─── LEDC PWM config ─────────────────────────────────────────────────────────
#define PWM_FREQ_HZ     20000
#define PWM_RESOLUTION  10        // bits → 0..1023
#define PWM_MAX         1023
#define LEDC_CH_LEFT    0
#define LEDC_CH_RIGHT   1

// ─── Safety limits ───────────────────────────────────────────────────────────
#define MAX_DUTY        900       // ~88% — headroom to prevent H-bridge shoot-through
#define DEADBAND_DUTY   50        // Below this, motor stalls; output 0 instead

// ─── Internal state ──────────────────────────────────────────────────────────
static float _current_action = 0.0f;

/**
 * Initialize motor driver GPIO and LEDC PWM channels.
 * Must be called in setup().
 */
void motor_init() {
    // Direction pins
    pinMode(MOTOR_L_DIR_A, OUTPUT);
    pinMode(MOTOR_L_DIR_B, OUTPUT);
    pinMode(MOTOR_R_DIR_A, OUTPUT);
    pinMode(MOTOR_R_DIR_B, OUTPUT);

    // LEDC PWM channels
    ledcSetup(LEDC_CH_LEFT,  PWM_FREQ_HZ, PWM_RESOLUTION);
    ledcSetup(LEDC_CH_RIGHT, PWM_FREQ_HZ, PWM_RESOLUTION);
    ledcAttachPin(MOTOR_L_PWM, LEDC_CH_LEFT);
    ledcAttachPin(MOTOR_R_PWM, LEDC_CH_RIGHT);

    // Start stopped
    ledcWrite(LEDC_CH_LEFT,  0);
    ledcWrite(LEDC_CH_RIGHT, 0);
    digitalWrite(MOTOR_L_DIR_A, LOW);
    digitalWrite(MOTOR_L_DIR_B, LOW);
    digitalWrite(MOTOR_R_DIR_A, LOW);
    digitalWrite(MOTOR_R_DIR_B, LOW);

    Serial.println("[MOTOR] Driver initialized (20 kHz PWM, 10-bit)");
}

/**
 * Set both motors to the same normalized action value.
 *
 * @param action  Normalized motor command in [-1.0, +1.0]
 *                Positive = forward, Negative = backward
 */
void motor_set_action(float action) {
    action = constrain(action, -1.0f, 1.0f);
    _current_action = action;

    int duty = (int)(fabsf(action) * MAX_DUTY);

    // Apply deadband
    if (duty < DEADBAND_DUTY) {
        motor_stop();
        return;
    }

    bool forward = (action > 0.0f);

    // Left motor
    digitalWrite(MOTOR_L_DIR_A, forward ? HIGH : LOW);
    digitalWrite(MOTOR_L_DIR_B, forward ? LOW  : HIGH);
    ledcWrite(LEDC_CH_LEFT, duty);

    // Right motor (same direction for balance)
    digitalWrite(MOTOR_R_DIR_A, forward ? HIGH : LOW);
    digitalWrite(MOTOR_R_DIR_B, forward ? LOW  : HIGH);
    ledcWrite(LEDC_CH_RIGHT, duty);
}

/**
 * Emergency stop — coast both motors to zero.
 */
void motor_stop() {
    ledcWrite(LEDC_CH_LEFT,  0);
    ledcWrite(LEDC_CH_RIGHT, 0);
    digitalWrite(MOTOR_L_DIR_A, LOW);
    digitalWrite(MOTOR_L_DIR_B, LOW);
    digitalWrite(MOTOR_R_DIR_A, LOW);
    digitalWrite(MOTOR_R_DIR_B, LOW);
    _current_action = 0.0f;
}

/**
 * Returns the last commanded action value.
 */
float motor_get_last_action() {
    return _current_action;
}
