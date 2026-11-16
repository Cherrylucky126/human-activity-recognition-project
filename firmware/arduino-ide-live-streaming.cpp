#include "BluetoothSerial.h"
#include <Wire.h>
#include <MAX30105.h> // SparkFun library supports MAX30102

#if !defined(CONFIG_BT_ENABLED) || !defined(CONFIG_BLUEDROID_ENABLED)
#error Bluetooth is not enabled!
#endif

BluetoothSerial ESP_BT;
MAX30105 particleSensor;

const int MPU_addr = 0x68; 
unsigned long lastSampleTime = 0;
const unsigned long intervalMs = 10; // 10ms = 100 Hz loop
int sampleCounter = 0;

void setupMPU() {
  Wire.beginTransmission(MPU_addr);
  Wire.write(0x6B); 
  Wire.write(0); // Wake up MPU6050
  Wire.endTransmission(true);
}

void setup() {
  setCpuFrequencyMhz(80); // Keep CPU clock at 80 MHz to stabilize power

  Serial.begin(115200);
  delay(1000);

  // 1. Initialize Bluetooth Classic FIRST so power levels settle before I2C starts
  ESP_BT.begin("ESP32_HAR_Classic"); 
  Serial.println("Bluetooth Classic started. Settling power...");
  delay(500); // 500ms delay prevents power-transient lockups on the sensor

  // 2. Start I2C bus at Standard Speed (100 kHz prevents RF noise glitches)
  Wire.begin(21, 22);
  Wire.setClock(100000);

  // 3. Setup MPU6050
  setupMPU();

  // 4. Setup MAX30102 with Hardware Force-Wake
  if (!particleSensor.begin(Wire, I2C_SPEED_STANDARD)) {
    Serial.println("MAX30102 not found. Check wiring/power!");
  } else {
    // Issue internal soft-reset to unfreeze registers without needing to unplug USB
    particleSensor.writeRegister8(0x57, 0x09, 0x40);
    delay(100);

    particleSensor.setup(
      /*powerLevel=*/ 0x3F,    // High LED power (~15mA)
      /*sampleAverage=*/ 4,    // 400Hz / 4 avg = 100Hz output
      /*ledMode=*/ 2,          // Dual Mode (Red + IR)
      /*sampleRate=*/ 400,     // 400 Hz hardware rate
      /*pulseWidth=*/ 411,     // 411us pulse width
      /*adcRange=*/ 4096       // 4096nA range
    );

    // Force LED current registers directly
    particleSensor.writeRegister8(0x57, 0x0C, 0x3F); // Red LED max current
    particleSensor.writeRegister8(0x57, 0x0D, 0x3F); // IR LED max current

    Serial.println("MAX30102 initialized post-Bluetooth.");
  }
}

void loop() {
  if (ESP_BT.connected()) {
    // Keep-alive handshake check
    if (ESP_BT.available()) {
      char incoming = ESP_BT.read();
      if (incoming == '3') {
        ESP_BT.println("ACK_3");
      }
    }

    // 100 Hz Loop (10ms)
    unsigned long currentMillis = millis();
    if (currentMillis - lastSampleTime >= intervalMs) {
      lastSampleTime = currentMillis;

      // --- Read MPU6050 ---
      Wire.beginTransmission(MPU_addr);
      Wire.write(0x3B);
      Wire.endTransmission(false);
      Wire.requestFrom(MPU_addr, 14, true);

      int16_t raw_ax = Wire.read() << 8 | Wire.read();
      int16_t raw_ay = Wire.read() << 8 | Wire.read();
      int16_t raw_az = Wire.read() << 8 | Wire.read();
      int16_t temp   = Wire.read() << 8 | Wire.read();
      int16_t raw_gx = Wire.read() << 8 | Wire.read();
      int16_t raw_gy = Wire.read() << 8 | Wire.read();
      int16_t raw_gz = Wire.read() << 8 | Wire.read();

      float ax = raw_ax / 16384.0;
      float ay = raw_ay / 16384.0;
      float az = raw_az / 16384.0;
      float gx = raw_gx / 131.0;
      float gy = raw_gy / 131.0;
      float gz = raw_gz / 131.0;

      // --- Direct Polling Read of IR Channel ---
      uint32_t irValue = particleSensor.getIR(); 

      // --- Stream CSV ---
      ESP_BT.printf("%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%lu\n", 
                    sampleCounter++, ax, ay, az, gx, gy, gz, irValue);
    }
  } else {
    delay(100);
  }
}
