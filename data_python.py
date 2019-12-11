import serial
import csv

ser = serial.Serial()
ser.baudrate=115200
ser.port="/dev/ttyUSB0"
ser.open()
# for i in range (100):
#     val =  ser.readline().decode("utf-8")
# print(val)
with open("output1.csv", "a") as fp:
    wr = csv.writer(fp, dialect='excel')
    while(True):
        val =  ser.readline().decode("utf-8").split(',')
        val[2] = val[2].rstrip('\r\n')
        print((val))
        wr.writerow(val)