unsigned long int milli_time;

void setup() {
  pinMode(A0, INPUT);
  Serial.begin(9600);
  Serial.println("CLEARDATA");
  Serial.println("LABEL,Computer Time,Time (Milli Sec.),Volt");
}

void loop() {
  milli_time = millis();
  Serial.print("DATA,TIME,");
  Serial.print(milli_time);
  Serial.print(",");
  Serial.println(analogRead(A0));
  delay(1);
}
