#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

Adafruit_MPU6050 mpu;

// Target frequency: 100 Hz = 10,000 microseconds interval
const unsigned long TARGET_INTERVAL_US = 10000; 
unsigned long lastExecutionTime = 0;
unsigned long sampleCount = 0;

void setup() {
  // Use a fast baud rate to prevent serial buffer bottlenecks
  Serial.begin(921600);
  while (!Serial) {
    delay(10);
  }

  // Initialize I2C and sensor
  Wire.begin();
  if (!mpu.begin()) {
    Serial.println("ERROR: MPU6050 not found!");
    while (1) { delay(10); }
  }

  // Configure sensor settings for calibration/sampling
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  // Small delay before starting stream
  delay(1000);
  lastExecutionTime = micros();
}

void loop() {
  unsigned long currentTime = micros();

  // Check if it's time for the next 100 Hz sample
  if (currentTime - lastExecutionTime >= TARGET_INTERVAL_US) {
    // Adjust lastExecutionTime precisely to prevent drift
    lastExecutionTime += TARGET_INTERVAL_US;
    sampleCount++;

    // Read sensor data
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    // Print CSV format: SampleIndex, Timestamp(us), AccX, AccY, AccZ, GyroX, GyroY, GyroZ
    Serial.print(sampleCount);
    Serial.print(",");
    Serial.print(currentTime);
    Serial.print(",");
    Serial.print(a.acceleration.x, 4);
    Serial.print(",");
    Serial.print(a.acceleration.y, 4);
    Serial.print(",");
    Serial.print(a.acceleration.z, 4);
    Serial.print(",");
    Serial.print(g.gyro.x, 4);
    Serial.print(",");
    Serial.print(g.gyro.y, 4);
    Serial.print(",");
    Serial.println(g.gyro.z, 4);
  }
}
