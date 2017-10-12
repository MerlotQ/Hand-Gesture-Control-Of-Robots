#include <L298N.h>

#define EN1 9
#define IN1_1 8
#define IN2_1 7

#define EN2 10
#define IN1_2 11
#define IN2_2 12
char receivedChar;
L298N motor_1(EN1, IN1_1, IN2_1);

L298N motor_2(EN2, IN1_2, IN2_2);

char readString;
void setup() {
  // put your setup code here, to run once:
Serial.begin(9600);  // initialize serial communications at 9600 bps
pinMode(13, OUTPUT);
  motor_1.setSpeed(80);  // an integer 0 - 255
  motor_2.setSpeed(80);

}
void rotate_clockwise(int speed){
  motor_1.forward();
  motor_2.forward();
  
  motor_1.setSpeed(speed);
  motor_2.setSpeed(speed);
  
  //delay(5000);
  
  }

void rotate_anticlockwise(int speed){
  motor_1.backward();
  motor_2.backward();
  
  motor_1.setSpeed(speed);
  motor_2.setSpeed(speed);
  
  //delay(5000);
  
  }
void move_forward(int speed){
  motor_1.forward();
  motor_2.backward();
  
  motor_1.setSpeed(speed);
  motor_2.setSpeed(speed);
  
  //delay(5000);
  
  }
 void stop_motor(){
  motor_1.stop();
  motor_2.stop();
  //delay(5000);
  
  }

void loop()
{
  // serial read section
  //Serial.println(Serial.available());
  while (Serial.available()) // this will be skipped if no data present, leading to
                             // the code sitting in the delay function below
  {
    delay(30);  //delay to allow buffer to fill 
    if (Serial.available() >0)
    {
      char c = Serial.read();  //gets one byte from serial buffer
      //readString += c; //makes the string readString
      //Serial.println("serial available"+c);
      //if(c=='6'){
      //  Serial.println("c is 6");
      //  digitalWrite(13, HIGH);
           
      //}
      //else{
      //  digitalWrite(13, LOW);
       // }
if (c == '1'){
  rotate_clockwise(255);
  }
if(c == '2'){
  rotate_anticlockwise(255);
  }
if (c == '3'){
  stop_motor();
  }
    }
  }
}
