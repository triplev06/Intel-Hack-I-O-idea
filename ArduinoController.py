import serial
import time

#sets up the serial connection


#Initialize serial connection


#Give some time to establish connection


#Send data to Arduino
def move():
    arduino_port = 'COM3' #changing arduino port to com3
    baud_rate = 9600 #Same as set in arduino port

    ser = serial.Serial(arduino_port, baud_rate, timeout=1)
    time.sleep(2)

    ser.write(b"G1 X4 Y2") #Sends data
    print("moved")

    time.sleep(5)
    ser.close()


#Close the serial connection when done(Optional based on your needs)
