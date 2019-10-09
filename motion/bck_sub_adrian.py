
# import the necessary packages
from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2
from modules.colorlabeler import ColorLabeler

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--max-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())

# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
	vs = VideoStream(src="http://10.0.0.4:8081").start()
	time.sleep(2.0)
# otherwise, we are reading from a video file
else:
	vs = cv2.VideoCapture(args["video"])

# initialize the first frame in the video stream
firstFrame = None

# distance function
def distance(p1, p2):
    x_diff = abs(p1[0]-p2[0])
    y_diff = abs(p1[1]-p2[1])
    return (x_diff*x_diff + y_diff*y_diff)

# class for objects
class Object:
    # stores a path of pixels points
    def __init__(self, point): # init the object at a starting point
        self.path = [point]
    def add_to_path(self, point, maxdist):
        if distance(self.path[-1], point) < maxdist:
            self.path.append(point)
            return True
        else:
            return False

# store all the objects
objects = []

# draw things in a colour
color = (0, 0, 255)

# loop over the frames of the video
while True:
	# grab the current frame and initialize the occupied/unoccupied
	# text
	frame = vs.read()
	frame = frame if args.get("video", None) is None else frame[1]
 
	# if the frame could not be grabbed, then we have reached the end
	# of the video
	if frame is None:
		break
 
	# resize the frame, convert it to grayscale, and blur it
	frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)
 
	# if the first frame is None, initialize it
	if firstFrame is None:
		firstFrame = gray
		continue

	# compute the absolute difference between the current frame and
	# first frame
	frameDelta = cv2.absdiff(firstFrame, gray)
	thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
 
	# dilate the thresholded image to fill in holes, then find contours
	# on thresholded image
	thresh = cv2.dilate(thresh, None, iterations=2)
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
 
	# loop over the contours
	for c in cnts:
		# if the contour is too small, ignore it
		cb = ColorLabeler()
		colour = cb.label(frame, c)
		is_white = colour == "white"
		is_invalid = (cv2.contourArea(c) > args["max_area"]) or not is_white
		if is_invalid:
			continue
 
		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
        
		# chuck the centre in the centres thing
		M = cv2.moments(c)
		cen = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
		added = []
		for o in objects:
			added.append(o.add_to_path(cen, 100))
		if True not in added:
			objects.append(Object(cen))

	# draw lines between the centres
	for o in objects:
		for i in range(1, len(o.path)):
			cv2.line(frame, o.path[i-1], o.path[i], color, 1)

	# show the frame and record if the user presses a key
	cv2.imshow("Camera", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key is pressed, break from the lop
	if key == ord("q"):
		break
 
# cleanup the camera and close any open windows
cv2.waitKey(0)
vs.stop() if args.get("video", None) is None else vs.release()
cv2.destroyAllWindows()
