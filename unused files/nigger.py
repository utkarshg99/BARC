import serial

ser = serial.Serial()
ser.baudrate=19200
ser.port="COM6"
ser.open()
while(True):
    # val = int(ser.readline().decode('ascii')[0])
    # val = int(ser.readline().decode('ascii')[0:3])
    val = ser.readline().decode('ascii').split(',')
    # print(val)
    tme=int(val[2])
    value=5.0*int(val[3][0:3])
    print(value)
