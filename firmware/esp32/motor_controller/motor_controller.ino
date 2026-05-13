/**
 * motor_controller.ino — ESP32 motor controller for Semantic SLAM Rover
 *
 * Hardware connections:
 *   Motor Driver (L298N or DRV8833):
 *     Left motor:   ENA=GPIO14, IN1=GPIO27, IN2=GPIO26
 *     Right motor:  ENB=GPIO13, IN3=GPIO25, IN4=GPIO33
 *
 *   Wheel encoders (Hall effect, 2-channel quadrature):
 *     Left A=GPIO34, Left B=GPIO35 (input-only, 3.3V tolerant)
 *     Right A=GPIO36, Right B=GPIO39
 *
 *   IMU (MPU-6050 I2C):
 *     SDA=GPIO21, SCL=GPIO22
 *
 * Serial protocol with Jetson (115200 baud, newline-terminated ASCII):
 *   Receive:  "CMD,<v_linear>,<v_angular>\n"   (m/s, rad/s)
 *   Transmit: "ODO,<left_ticks>,<right_ticks>,<dt_ms>\n"  (10 Hz)
 *             "IMU,<ax>,<ay>,<az>,<gx>,<gy>,<gz>\n"       (10 Hz, if present)
 *             "ERR,<message>\n"                            (on error)
 *
 * Safety:
 *   If no CMD received within CMD_TIMEOUT_MS, motors are stopped.
 *
 * Tuning constants at top of file — adjust for your specific chassis.
 */

#include <Arduino.h>
#include <Wire.h>

// ── Physical constants ─────────────────────────────────────────────────────────
static constexpr float WHEEL_BASE_M     = 0.22f;    // meters, center-to-center
static constexpr float WHEEL_RADIUS_M   = 0.033f;   // meters
static constexpr int   ENCODER_TICKS_REV = 1120;    // 28 pulses × 40:1 gear ratio

// ── Motor driver pins ──────────────────────────────────────────────────────────
static constexpr int PIN_ENA = 14;
static constexpr int PIN_IN1 = 27;
static constexpr int PIN_IN2 = 26;
static constexpr int PIN_ENB = 13;
static constexpr int PIN_IN3 = 25;
static constexpr int PIN_IN4 = 33;

static constexpr int PWM_CHANNEL_L = 0;
static constexpr int PWM_CHANNEL_R = 1;
static constexpr int PWM_FREQ      = 20000;   // 20 kHz (above audible range)
static constexpr int PWM_BITS      = 8;       // 0–255 duty cycle
static constexpr int PWM_MAX       = 255;

// ── Encoder pins ───────────────────────────────────────────────────────────────
static constexpr int PIN_ENC_LA = 34;
static constexpr int PIN_ENC_LB = 35;
static constexpr int PIN_ENC_RA = 36;
static constexpr int PIN_ENC_RB = 39;

// ── Timing ─────────────────────────────────────────────────────────────────────
static constexpr unsigned long ODO_PUBLISH_MS  = 100;   // 10 Hz odometry
static constexpr unsigned long IMU_PUBLISH_MS  = 100;   // 10 Hz IMU
static constexpr unsigned long CMD_TIMEOUT_MS  = 500;   // stop if no CMD received

// ── IMU ────────────────────────────────────────────────────────────────────────
static constexpr int MPU6050_ADDR   = 0x68;
static constexpr int MPU6050_PWR    = 0x6B;
static constexpr int MPU6050_ACCEL  = 0x3B;
static constexpr float ACCEL_SCALE  = 2.0f / 32768.0f * 9.81f;  // ±2g → m/s²
static constexpr float GYRO_SCALE   = 250.0f / 32768.0f * (3.14159f / 180.0f); // ±250°/s → rad/s

// ── State ──────────────────────────────────────────────────────────────────────
volatile long  g_enc_left  = 0;
volatile long  g_enc_right = 0;
volatile int   g_last_la   = LOW;
volatile int   g_last_ra   = LOW;

float  g_cmd_v_lin  = 0.0f;
float  g_cmd_v_ang  = 0.0f;
unsigned long g_last_cmd_ms = 0;

unsigned long g_last_odo_ms = 0;
unsigned long g_last_imu_ms = 0;

bool g_imu_present = false;

// ── Encoder ISRs ───────────────────────────────────────────────────────────────

void IRAM_ATTR enc_left_isr() {
  int a = digitalRead(PIN_ENC_LA);
  int b = digitalRead(PIN_ENC_LB);
  if (a != g_last_la) {
    g_enc_left += (a == b) ? 1 : -1;
    g_last_la = a;
  }
}

void IRAM_ATTR enc_right_isr() {
  int a = digitalRead(PIN_ENC_RA);
  int b = digitalRead(PIN_ENC_RB);
  if (a != g_last_ra) {
    g_enc_right += (a == b) ? -1 : 1;  // right encoder is mirrored
    g_last_ra = a;
  }
}

// ── Motor control ──────────────────────────────────────────────────────────────

void set_motor_left(float speed) {
  // speed: -1.0 to +1.0
  int pwm = (int)(constrain(fabsf(speed), 0.0f, 1.0f) * PWM_MAX);
  if (speed >= 0) {
    digitalWrite(PIN_IN1, HIGH);
    digitalWrite(PIN_IN2, LOW);
  } else {
    digitalWrite(PIN_IN1, LOW);
    digitalWrite(PIN_IN2, HIGH);
  }
  ledcWrite(PWM_CHANNEL_L, pwm);
}

void set_motor_right(float speed) {
  int pwm = (int)(constrain(fabsf(speed), 0.0f, 1.0f) * PWM_MAX);
  if (speed >= 0) {
    digitalWrite(PIN_IN3, HIGH);
    digitalWrite(PIN_IN4, LOW);
  } else {
    digitalWrite(PIN_IN3, LOW);
    digitalWrite(PIN_IN4, HIGH);
  }
  ledcWrite(PWM_CHANNEL_R, pwm);
}

void apply_cmd_vel(float v_lin, float v_ang) {
  // Differential drive inverse kinematics
  float v_left  = (v_lin - v_ang * WHEEL_BASE_M / 2.0f) / WHEEL_RADIUS_M;
  float v_right = (v_lin + v_ang * WHEEL_BASE_M / 2.0f) / WHEEL_RADIUS_M;

  // Normalise to [–1, 1] using max wheel speed
  float max_v = max(fabsf(v_left), fabsf(v_right));
  float max_wheel_rad_s = (float)PWM_MAX / PWM_MAX;  // normalised: full speed = 1 m/s equiv
  if (max_v > 1.0f) {
    v_left  /= max_v;
    v_right /= max_v;
  } else {
    // Scale to physical max speed (assume 0.3 m/s at full PWM for this gearbox)
    static constexpr float MAX_WHEEL_SPEED = 0.3f / WHEEL_RADIUS_M;  // rad/s
    v_left  = constrain(v_left  / MAX_WHEEL_SPEED, -1.0f, 1.0f);
    v_right = constrain(v_right / MAX_WHEEL_SPEED, -1.0f, 1.0f);
  }

  set_motor_left(v_left);
  set_motor_right(v_right);
}

void stop_motors() {
  ledcWrite(PWM_CHANNEL_L, 0);
  ledcWrite(PWM_CHANNEL_R, 0);
  digitalWrite(PIN_IN1, LOW); digitalWrite(PIN_IN2, LOW);
  digitalWrite(PIN_IN3, LOW); digitalWrite(PIN_IN4, LOW);
}

// ── IMU ────────────────────────────────────────────────────────────────────────

bool init_mpu6050() {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(MPU6050_PWR);
  Wire.write(0x00);  // wake up
  return Wire.endTransmission() == 0;
}

bool read_mpu6050(float &ax, float &ay, float &az, float &gx, float &gy, float &gz) {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(MPU6050_ACCEL);
  if (Wire.endTransmission(false) != 0) return false;
  Wire.requestFrom(MPU6050_ADDR, 14);
  if (Wire.available() < 14) return false;

  int16_t raw_ax = (Wire.read() << 8) | Wire.read();
  int16_t raw_ay = (Wire.read() << 8) | Wire.read();
  int16_t raw_az = (Wire.read() << 8) | Wire.read();
  Wire.read(); Wire.read();  // temperature (skip)
  int16_t raw_gx = (Wire.read() << 8) | Wire.read();
  int16_t raw_gy = (Wire.read() << 8) | Wire.read();
  int16_t raw_gz = (Wire.read() << 8) | Wire.read();

  ax = raw_ax * ACCEL_SCALE;
  ay = raw_ay * ACCEL_SCALE;
  az = raw_az * ACCEL_SCALE;
  gx = raw_gx * GYRO_SCALE;
  gy = raw_gy * GYRO_SCALE;
  gz = raw_gz * GYRO_SCALE;
  return true;
}

// ── Serial command parsing ─────────────────────────────────────────────────────

void parse_serial_line(const String &line) {
  if (!line.startsWith("CMD,")) return;

  int comma1 = line.indexOf(',', 4);
  if (comma1 < 0) {
    Serial.println("ERR,bad CMD format");
    return;
  }
  float v_lin = line.substring(4, comma1).toFloat();
  float v_ang = line.substring(comma1 + 1).toFloat();

  g_cmd_v_lin = v_lin;
  g_cmd_v_ang = v_ang;
  g_last_cmd_ms = millis();
}

// ── Setup ──────────────────────────────────────────────────────────────────────

void setup() {
  Serial.begin(115200);

  // Motor direction pins
  pinMode(PIN_IN1, OUTPUT); pinMode(PIN_IN2, OUTPUT);
  pinMode(PIN_IN3, OUTPUT); pinMode(PIN_IN4, OUTPUT);
  stop_motors();

  // PWM channels
  ledcSetup(PWM_CHANNEL_L, PWM_FREQ, PWM_BITS);
  ledcSetup(PWM_CHANNEL_R, PWM_FREQ, PWM_BITS);
  ledcAttachPin(PIN_ENA, PWM_CHANNEL_L);
  ledcAttachPin(PIN_ENB, PWM_CHANNEL_R);

  // Encoder inputs (no pull-up, Hall sensors provide voltage)
  pinMode(PIN_ENC_LA, INPUT);
  pinMode(PIN_ENC_LB, INPUT);
  pinMode(PIN_ENC_RA, INPUT);
  pinMode(PIN_ENC_RB, INPUT);
  attachInterrupt(digitalPinToInterrupt(PIN_ENC_LA), enc_left_isr,  CHANGE);
  attachInterrupt(digitalPinToInterrupt(PIN_ENC_RA), enc_right_isr, CHANGE);

  // IMU
  Wire.begin(21, 22);
  g_imu_present = init_mpu6050();
  if (g_imu_present) {
    Serial.println("ERR,IMU_OK");  // using ERR channel for info (not an error)
  }

  g_last_cmd_ms = millis();
}

// ── Main loop ──────────────────────────────────────────────────────────────────

void loop() {
  unsigned long now = millis();

  // ── Read incoming serial commands ──────────────────────────────────────────
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() > 0) {
      parse_serial_line(line);
    }
  }

  // ── Safety watchdog ────────────────────────────────────────────────────────
  if ((now - g_last_cmd_ms) > CMD_TIMEOUT_MS) {
    g_cmd_v_lin = 0.0f;
    g_cmd_v_ang = 0.0f;
    stop_motors();
  } else {
    apply_cmd_vel(g_cmd_v_lin, g_cmd_v_ang);
  }

  // ── Publish odometry ───────────────────────────────────────────────────────
  if ((now - g_last_odo_ms) >= ODO_PUBLISH_MS) {
    unsigned long dt_ms = now - g_last_odo_ms;
    g_last_odo_ms = now;

    noInterrupts();
    long left_ticks  = g_enc_left;
    long right_ticks = g_enc_right;
    g_enc_left  = 0;
    g_enc_right = 0;
    interrupts();

    Serial.print("ODO,");
    Serial.print(left_ticks);
    Serial.print(",");
    Serial.print(right_ticks);
    Serial.print(",");
    Serial.println(dt_ms);
  }

  // ── Publish IMU ────────────────────────────────────────────────────────────
  if (g_imu_present && (now - g_last_imu_ms) >= IMU_PUBLISH_MS) {
    g_last_imu_ms = now;
    float ax, ay, az, gx, gy, gz;
    if (read_mpu6050(ax, ay, az, gx, gy, gz)) {
      Serial.print("IMU,");
      Serial.print(ax, 4); Serial.print(",");
      Serial.print(ay, 4); Serial.print(",");
      Serial.print(az, 4); Serial.print(",");
      Serial.print(gx, 4); Serial.print(",");
      Serial.print(gy, 4); Serial.print(",");
      Serial.println(gz, 4);
    }
  }
}
