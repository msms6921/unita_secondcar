#include <Arduino.h>

constexpr uint8_t STEERING_POT_PIN = A0;
constexpr unsigned long SAMPLE_INTERVAL_MS = 50;

unsigned long lastSampleMs = 0;

void setup() {
  Serial.begin(115200);
  pinMode(STEERING_POT_PIN, INPUT);
}

void loop() {
  const unsigned long now = millis();
  if (now - lastSampleMs < SAMPLE_INTERVAL_MS) {
    return;
  }

  lastSampleMs = now;
  Serial.println(analogRead(STEERING_POT_PIN));
}
