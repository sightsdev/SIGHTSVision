
'''
Tutorial:
https://youtu.be/GGeF_3QOHGE

This file can't be used for training. This OpenCV code will use a .cfg and .weights file
as input (which we get from darknet), and use them to make detections.
'''

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# only one class to make training process much easier
classes = ['sign']

while True:
    success, img = cap.read()

    cv2.imshow('Frame', img)
    cv2.waitKey(1)
