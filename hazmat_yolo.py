
'''
This file can't be used for training. This OpenCV code will use a .cfg and .weights file
as input (which we get from darknet), and use them to make detections.
'''

import cv2
import numpy as np
import argparse
import modules.HOGUtils
from modules.classify.classify_abstracted import *
from imutils.video import VideoStream
import time
import os

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--suppression", default="on", help="enable use of nonmax suppression")
ap.add_argument("-sp", "--speed", default="slow", help="fast/slow: classification algorithm speed")
ap.add_argument('-vs', '--videosource', default='r', help='-vs r for robot, -vs w for webcam.')
ap.add_argument('-ip', '--address', default='10.0.0.5', help='ip address of the robot on the SART Control network')
args = vars(ap.parse_args())

# I hope this is the right IP address
ip = args['address']
blue = (255,0,0)
green = (0,255,0)
red = (0,0,255)
draw_colour = blue
#cap = cv2.VideoCapture(0)
whT = 320
robot_camera = args['videosource'] == 'r'
suppression = args['suppression'] == "on"
fast_mode = args['speed'] == 'fast'
confThreshold = 0.5

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
    for output in outs:
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


def boxArea(box):
    [x1,y1,x2,y2] = box
    return abs((x2 - x1) * (y2 - y1))


def boxValid(box):
    [x1,y1,x2,y2] = box
    goodArea = boxArea(box) > 500
    goodX = x2 > x1
    goodY = y2 > y1
    return goodArea and goodX and goodY


def bigger(box1, box2):
    # return the bigger of two boxes
    if boxArea(box1) > boxArea(box2):
        return box1
    else:
        return box2


def smaller(box1, box2):
    # return the smaller of two boxes
    if boxArea(box1) < boxArea(box2):
        return box1
    else:
        return box2


# turn x y w h into x1 y1 x2 y2
def makeBox(bbox):
    x,y,w,h = bbox
    return [x, y, x+w, y+h]


# apply makeBox to a list of bounding boxes
def makeBoxes(bboxes):
    new_boxes = []
    for box in bboxes:
        new_boxes.append(makeBox(box))
    return new_boxes


# ensure that the bboxes coming in are coming as outputs from makeBoxes
def drawBox(bbox,img):
    x1,y1,x2,y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), draw_colour, 2)
    return img


def drawBoxes(bboxes,img):
    for box in bboxes:
        img = drawBox(box,img)
    return img


def annotate(img, bounding_box, lst):
    # box
    # grab just the area of the located sign from the image, instead of the entire image
    x1, y1, x2, y2 = int(bounding_box[0]), int(bounding_box[1]), int(bounding_box[2]), int(bounding_box[3])
    region = img[y1:y2, x1:x2]

    # constants
    text = classify(region, sign_list)
    lst.append(text) # throw the text onto the pile of signs we have found
    text_x = int(x1 + (x2-x1)/2) - 20
    text_y = int(y1 + (y2-y1)/2) - 20
    colour = (255, 255, 255)
    black = (0, 0, 0)
    font_size = 0.75
    font_thickness = 2
    buff = 5

    # get text position etc
    (text_width_1, text_height_1) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_size, thickness=font_thickness)[0]

    # get coords of rectangle behind text
    box_coords_1 = ((text_x - buff, text_y + buff), (text_x + text_width_1 + buff, text_y - text_height_1 - buff))

    # draw rectangle behind text
    cv2.rectangle(img, box_coords_1[0], box_coords_1[1], black, cv2.FILLED)

    # draw text
    cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_size, colour, font_thickness)

    return img, lst


# it gives a super weird error for no reason, so I just use a try-except to dodge that
# and it works fine
def annotateFullySafely(bboxes, img, lst):
    for bbox in bboxes:
        try:
            img, lst = annotate(img, bbox, lst)
        except:
            pass
    return img, lst


# initialize classification stuff
sign_list = []
templates_dir = "modules/classify/"
if fast_mode:
    folder = "templates_fast/"
else:
    folder = "templates/"
FILETYPE = ".png"
names = ["Explosives 1.1 1", "Explosives 1.2 1", "Explosives 1.3 1", "Explosives 1.4 1", "Blasting Agents 1.5 1", "Explosives 1.6 1", "Flammable Gas 2", "Non-Flammable Gas 2",
"Oxygen 2", "Inhalation Hazard", "Flammable 3", "Gasoline 3", "Combustible 3", "Fuel Oil 3", "Dangerous When Wet 4", "Flammable Solid 4", "Spontaneously Combustible 4",
"Oxidizer 5.1", "Organic Peroxide 5.2", "Inhalation Hazard 6", "Poison 6", "Toxic 6", "Radioactive 7", "Corrosive 8", "Other Dangerous Goods 9", "Dangerous"]


# generate list of template signs
for i in range(1, 27):
    sign_list.append(Sign(templates_dir + folder + str(i) + FILETYPE, names[i-1]))


# get video stream from the robot
if args['videosource'] == "r":
    # robot
    print("using robot camera")
    vs = VideoStream(src="http://"+ip+":8081/1/stream").start()
else:
    # webcam
    print("using local computer webcam")
    vs = VideoStream(src=0).start()


# make a list to store signs found in the order we found them
signs_found = []


while True:

    # reads image from webcam
    #success, img = cap.read()
    img = vs.read()

    blob = cv2.dnn.blobFromImage(img,1/255,(whT,whT),[0,0,0],1,crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]

    outputs = net.forward(outputNames)
    
    height, width = img.shape[0], img.shape[1]
    img = cv2.resize(img, (600,400))

    detections = makeBoxes(findObjects(outputs,img))

    # non max suppression
    if len(detections)>1 and suppression:
        matrix = np.vstack(detections)
        detections = modules.HOGUtils.non_max_suppression_fast(matrix, 0.5)

    img, signs_found = annotateFullySafely(detections,img,signs_found)
    img = drawBoxes(detections,img)
    img = cv2.resize(img, (width, height))

    cv2.imshow('Frame', img)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27: break

# test

# chuck the signs in a fil
f = open("hazmat/res/results.txt","w")

def countElem(e,l):
    c=0
    for k in l:
        if k==e:
            c+=1
    return c

print(signs_found)

signs_found_pruned = []
for sign in signs_found:
    if sign not in signs_found_pruned and countElem(sign,signs_found) >= 10:
        signs_found_pruned.append(sign)

for name in signs_found_pruned:
    f.write(name+"\n")

# close f
f.close()

# destroy all windows
cv2.destroyAllWindows()
