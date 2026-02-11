#include <SPI.h>
#include <Servo.h>

Servo servo1;
Servo servo2;

volatile byte lastReceived = 0;   // SPI incoming byte
volatile bool newData = false;    // Flag for main loop

void setup() {
  // --- Servo Setup ---
  servo1.attach(5);   // 
  servo2.attach(6);
  servo1.write(90);   // initialize position 
  servo2.write(90);

  // --- SPI Slave Setup ---
  pinMode(MISO, OUTPUT);     // 

  SPCR |= _BV(SPE);          // Enable SPI in slave mode
  SPI.attachInterrupt();     


}

void loop() {
  // When a byte comes in via SPI
  if (newData) {
    newData = false; 
    //do something with servos here
  }
}

// Interrupt when SPI receives 8 bits
ISR(SPI_STC_vect) {
  lastReceived = SPDR;       // Read incoming byte
  SPDR = lastReceived + 1;   // response to master
  newData = true;
}
