
# imports
import cv2
import imutils
import argparse
import numpy as np
import time
from math import sqrt

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size") # guessing this is a distance from the previous frame's ball
# that the new frame's ball must be within such that it can be considered the same ball
args = vars(ap.parse_args())

# constants
MAX_DIST = int(args['buffer'])
THRESH_BORDER = 127
# i will store the path of points that each object has travelled in in its own array.
# there may be more than one object, so i will store each array in the following array.
ARRAY_OF_ARRAYS = []
GREEN = (0, 255, 0)

# distance between 2 points
def distance(p1, p2):
    # p1 and p2 are (x,y) tuple pairs
    xdiff = abs(p1[0] - p2[0])
    ydiff = abs(p1[1] - p2[1])
    dist = sqrt(xdiff*xdiff + ydiff*ydiff)
    return dist

# video
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# loop
while True:
    # read frame
    ret, frame = cap.read()

    # handle the frame from VideoCapture or VideoStream
    frame = frame[1] if args.get("video", False) else frame

    # be super safe
    if frame is None: break
    
    # resize blur et cetera
    # resize the frame, blur it, and convert it to binary
    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, THRESH_BORDER, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    # find the centre of the new contours
    new_centre_points = []
    for contour in contours:
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        new_centre_points.append((cX, cY))
    
    # look at each of the centres and see if they are close to the last point in any of our arrays
    # which are pre-existing paths. if it isnt, make a new array and append the point to it.
    for centre in new_centre_points:
        if len(ARRAY_OF_ARRAYS) > 0:
            for pre_existing_path in ARRAY_OF_ARRAYS:
                if distance(centre, pre_existing_path[-1]) < MAX_DIST:
                    pre_existing_path.append(centre) # add it to preexisting one
                else:
                    ARRAY_OF_ARRAYS.append([centre]) # make a new one
        else:
            ARRAY_OF_ARRAYS.append([centre]) # make a new one

    # draw a dot on all the pre-existing points
    for pre_existing_path in ARRAY_OF_ARRAYS:
        for point in pre_existing_path:
            cv2.circle(frame, point, 2, GREEN, -1)
    
    cv2.imshow("Window", frame)

    # close window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()