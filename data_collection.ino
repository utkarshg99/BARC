double milli_time;

void setup() {
  //pinMode(A0, INPUT);
  Serial.begin(250000);
  Serial.println("CLEARDATA");
  Serial.println("time(milli),GSR,Volt");
}

void loop() {
  milli_time = micros()/1000.0;
  //Serial.print("DATA,TIME,");
  Serial.print(milli_time);
  Serial.print(",");
  Serial.print(analogRead(A0));
  Serial.print(",");
  Serial.println(analogRead(A6));
}
