#from picamera.array import PiRGBArray
#from picamera import PiCamera
import numpy as np
import cv2
import time
from threading import Thread
#import RPi.GPIO as GPIO 
import time
import serial

#GPIO.setmode(GPIO.BCM)
#GPIO.setup(18, GPIO.OUT)
#pwm = GPIO.PWM(18, 100)
#pwm.start(50)

face_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_eye.xml')

px = 0
py = 0
height = None
width = None
angle = 90.

camera = cv2.VideoCapture(0) # usar webcam
camera.set(cv2.cv.CV_CAP_PROP_FPS, 5)
camera.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 320); 
camera.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 240);

#camera = PiCamera()
#camera.resolution = (320, 240)
#camera.framerate = 4
#capture = PiRGBArray(camera, size=(320, 240))

#time.sleep(0.1)
#duty = float(angle) / 10.0 + 2.5
#pwm.ChangeDutyCycle(duty)

while True:        
#for frame in camera.capture_continuous(capture, format="bgr", use_video_port=True):
    ret, image = camera.read()
    #image = frame.array
    #frame.truncate(0)
    
    if(width == None):
        height, width = image.shape[:2] 
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        px, py = x+w/2, y+h/2 # centro do objeto
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.circle(image,(px,py), 5, (0,0,255), -1)
        err = width/2 - px # 160        
        break
        
    cv2.imshow('img', image)
    if cv2.waitKey(1) == 27: 
        break

#camera.release()
cv2.destroyAllWindows()
