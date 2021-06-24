
'''
Tutorial:
https://youtu.be/GGeF_3QOHGE

This file can't be used for training. This OpenCV code will use a .cfg and .weights file
as input (which we get from darknet), and use them to make detections.
'''

import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-mode", "--mode", default="hazmat", help="coco mode versus hazmat mode")
args = vars(ap.parse_args())

cap = cv2.VideoCapture(0)
whT = 320
classesFile = 'hazmat/coco.names'
testing = args['mode'] == "coco"
confThreshold = 0.5

if testing:
    modelConfiguration = 'hazmat/coco.cfg'
    modelWeights = 'hazmat/coco.weights'
    with open(classesFile,'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
else:
    modelConfiguration = 'hazmat/custom-yolov4-tiny-detector.cfg'
    modelWeights = 'hazmat/custom-yolov4-tiny-detector_best.weights'
    classes = ['sign'] # only one sign

net = cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeights)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

# figure out how to change this to GPU
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def findObjects(outs,img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        # det is details
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w,h = int(det[2]*wT) , int(det[3]*hT)
                x,y = int((det[0]*wT) - w/2) , int((det[1]*hT) - h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
    
    # only use the bbox
    return bbox


def drawBox(bbox,img):
    x1,y1 = bbox[0],bbox[1]
    x2,y2 = x1+bbox[2],y1+bbox[3]
    cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
    return img


def drawBoxes(bboxes,img):
    for box in bboxes:
        img = drawBox(box,img)
    return img



while True:
    success, img = cap.read()

    blob = cv2.dnn.blobFromImage(img,1/255,(whT,whT),[0,0,0],1,crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    # print(outputNames)
    # print(net.getUnconnectedOutLayers())

    outputs = net.forward(outputNames)
    
    detections = findObjects(outputs,img)
    img = drawBoxes(detections,img)

    cv2.imshow('Frame', img)
    cv2.waitKey(1)
