from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import threading
import time
import RPi.GPIO as GPIO 
import serial

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.OUT)
pwm = GPIO.PWM(18, 100)
pwm.start(50)

# sudo modprobe uvcvideo nodrop=1 timeout=6000
p_det = False
image_det = None

f = open("mscoco_label_map.pbtxt","r") 
data = f.read()
r1=data.split('item')
r1.pop(0)

map_label = {}
for i in range(len(r1)):
    id = r1[i].split('id:')[1].split('\n')[0].strip()
    name = r1[i].split('display_name:')[1].split('\n')[0].strip().replace('"','')
    map_label[int(id)] = name

print map_label

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile('frozen_inference_graph.pb', 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

def load_image_into_numpy_array(image):
    im_height, im_width = image.shape[:2] 
    aux = np.array(image).reshape((im_height, im_width, 3)).astype(np.uint8)
    return aux 

def th_camera():
    global image_det, p_det
    camera = cv2.VideoCapture(0) # usar webcam
    camera.set(cv2.cv.CV_CAP_PROP_FPS, 5)
    camera.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 320); 
    camera.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 240);
    #camera = PiCamera()
    #camera.resolution = (320, 240)
    #camera.framerate = 4
    #capture = PiRGBArray(camera, size=(320, 240))
    #for frame in camera.capture_continuous(capture, format="bgr", use_video_port=True):
    while (True):    
        ret, image = camera.read()
        cv2.imshow('odet', image)
        #image = frame.array
        #frame.truncate(0)
        
        if(p_det == False):
            image_det = image
            p_det = True
        if cv2.waitKey(1) == 27: 
            break

def th_detect(): 
    global image_det, p_det 
    err = 0
    err0 = 0
    px = 0
    py = 0
    pm = 0   
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
    
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            
            while True:
                          
                if(p_det == True):
                    print '...'
                    image_np = load_image_into_numpy_array(image_det)
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
                      feed_dict={image_tensor: image_np_expanded})
                 
                                            
                    result = [[map_label[int(item)],box] for item, per, box in zip(classes[0],scores[0],boxes[0]) if per >= 0.65]
                    if(len(result) > 0):
                        height, width = image_det.shape[:2] 
                        for obj in result:
                            xmin = int(obj[1][1]*width) # posicao do objeto
                            ymin = int(obj[1][0]*height)
                            xmax = int(obj[1][3]*width)
                            ymax = int(obj[1][2]*height)
                            px, py = xmin+(xmax-xmin)/2, ymin+(ymax-ymin)/2 # centro do objeto
                            cv2.rectangle(image_det, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                            cv2.circle(image_det,(px,py), 5, (0,0,255), -1)
                            #print xmin,ymin,xmax,ymax
                            err = width/2 - px
                            print obj[0]
                            print px, py
                            break
                        cv2.imshow('odet', image_det)
                    p_det = False
                    
threading.Thread(target=th_camera).start()
threading.Thread(target=th_detect).start()


