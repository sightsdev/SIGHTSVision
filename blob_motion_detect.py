
# using adrian's article
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
from math import sqrt

ap = argparse.ArgumentParser()
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(ap.parse_args())

# compute distance between two tuple (x,y) points
def distance(p1, p2):
    x_diff = abs(p1[0] - p2[0])
    y_diff = abs(p1[1] - p2[1])
    return sqrt(x_diff*x_diff + y_diff*y_diff)

# make our colour black
colour_lower = (0, 0, 0)
colour_upper = (20, 20, 20)
pts = deque(maxlen=args["buffer"])

vs = VideoStream(src=0).start()

time.sleep(2.0)

while True:
    frame = vs.read()

    print(frame.shape)

    #frame = frame[1]

    if frame is None: break

    frame = imutils.resize(frame, width=600)

    blurred = cv2.GaussianBlur(frame, (11,11), 0)

    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # construct a mask for the colour we specified
    mask = cv2.inRange(hsv, colour_lower, colour_upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # contours etc
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    # only proceed if at least 1 contour was found
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea) # get the contour w the largest area
        ((x,y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))

        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0,255,255), 2)
            cv2.circle(frame, center, 5, (0,0,255), -1)
    
    # update points queue
    pts.appendleft(center)

    # loop over points found
    for i in range(1, len(pts)):

        # if any of the tracked points are none, ignore them
        if pts[i-1] is None or pts[i] is None:
            continue # sends back to top of loop w/out completing it
        
        # otherwise compute thickness of line and draw it
        #thickness = int(np.sqrt(args["buffer"] / float(i+1)) * 2.5)
        thickness = 2
        if distance(pts[i-1], pts[i]) < 100:
            cv2.line(frame, pts[i-1], pts[i], (0,0,255), thickness)
    
    # display image
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    key = cv2.waitKey(1) & 0xFF

    # if the key q is pressed then quit
    if key == ord('q'):
        break

# stop the stream
vs.stop()
cv2.destroyAllWindows()
