
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# only one class to make training process much easier
classes = ['sign']

while True:
    success, img = cap.read()

    cv2.imshow('Frame', img)
    cv2.waitKey(1)