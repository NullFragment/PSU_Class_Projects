import RPi.GPIO as GPIO
import sys

print "Setup GPIO"
GPIO.setmode(GPIO.BCM)
GPIO.setup(23, GPIO.OUT)
GPIO.setup(18, GPIO.OUT)
ans = "y"

x = float(raw_input("Enter Pulse Frequency and press enter: "))
y = float(raw_input("Enter duty cycle and press enter: "))
print ("Frequency set to " + str(x) + "Hz with a " + str(y) + "% duty cycle.")

while ans == "y":
    z = raw_input("Press enter to start")
    print"Turning counter-clockwise"
    GPIO.output(23, GPIO.LOW)
    pulse = GPIO.PWM(18, x)
    pulse.start(y)
    z = raw_input("Press enter to stop")
    pulse.stop()

    z = raw_input("Press enter to unwind")
    print"Turning clockwise"
    GPIO.output(23, GPIO.HIGH)
    pulse = GPIO.PWM(18, 2*x)
    pulse.start(y)
    z = raw_input("Press enter to stop")
    pulse.stop()
    ans = raw_input("Run again? (y/n)")
    if ans == "y":
        print ("Frequency set to " + str(x) + "Hz with a " + str(y) + "% duty cycle.")
    
GPIO.cleanup()
