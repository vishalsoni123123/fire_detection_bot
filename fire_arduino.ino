#include <Servo.h>

Servo servoX;
Servo servoY;

const int servoPinX = 9;
const int servoPinY = 10;
const int dataPin = 6;
const int noDataPin = 7;
const int resetPin = 5;

unsigned long lastDataTime = 0;
const unsigned long dataTimeout = 2000; // Timeout period in milliseconds

void setup() {
  Serial.begin(9600); // Initialize Serial communication
  servoX.attach(servoPinX);
  servoY.attach(servoPinY);

  pinMode(dataPin, OUTPUT);
  pinMode(noDataPin, OUTPUT);
  pinMode(resetPin, OUTPUT);

  resetServos();
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim(); // Remove any extra whitespace

    int commaIndex = input.indexOf(',');
    if (commaIndex > 0) {
      int servoXPos = input.substring(0, commaIndex).toInt();
      int servoYPos = input.substring(commaIndex + 1).toInt();

      // Ensure the servo positions are within the valid range (0-180)
      servoXPos = constrain(servoXPos, 0, 180);
      servoYPos = constrain(servoYPos, 0, 180);

      // Write the positions to the servos
      servoX.write(servoXPos);
      servoY.write(servoYPos);

      // Indicate data is available with PWM at 3V
      analogWrite(dataPin, 153);
      digitalWrite(noDataPin, LOW);
      digitalWrite(resetPin, HIGH);

      // Update the last data received time
      lastDataTime = millis();
    }
  } else if (millis() - lastDataTime > dataTimeout) {
    // No data received within the timeout period: reset servos and set pins
    resetServos();
    digitalWrite(dataPin, LOW);
    digitalWrite(noDataPin, HIGH);
    digitalWrite(resetPin, LOW);
  }
}

void resetServos() {
  servoX.write(90); // Reset X servo to 90 degrees
  servoY.write(90); // Reset Y servo to 90 degrees
}
