import serial
import time

#sets up the serial connection
arduino_port = 'COM3' #changing arduino port to com3
baud_rate = 9600 #Same as set in arduino port

#Initialize serial connection
ser = serial.Serial(arduino_port, baud_rate, timeout=1)

#Give some time to establish connection
time.sleep(2)

#Send data to Arduino

ser.write(b"G1 X1 Y2") #Sends data

#Read data from arduino
while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').rstrip()
            print(f"Received: {line}")

#Close the serial connection when done(Optional based on your needs)
#ser.close()