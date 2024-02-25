
import serial
import time

from serial import Serial

try:
	ser = Serial(
	   port='/dev/ttyUSB0',
	   baudrate=9600,
	   parity=serial.PARITY_NONE,
	   stopbits=serial.STOPBITS_ONE,
	   bytesize=serial.EIGHTBITS
	)
except:
	print("Can't open serial")


i = 0

while(True):
	ser.write(i.to_bytes(1, "big"))
	i += 1
	
	if(i == 255):
		i = 0
		
	print(i)
	time.sleep(1)
	
	
