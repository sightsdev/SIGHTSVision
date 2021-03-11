
# this is the main file of the hazmat detector
# to run it, enter "hazmat.py -vs w" in console.
# this will run the hazmat detector on your computer camera.
# just "hazmat.py -vs r" will run the detector on your robot camera, if you've entered its IP into the variable below.


# imports
import modules.HOGUtils as HU
import imutils
import argparse
import cv2
import numpy as np
from math import pi, degrees
from modules.classify.classify_abstracted import *
from imutils.video import VideoStream
import time

# ip
ip = "10.0.0.3"

ap = argparse.ArgumentParser()
ap.add_argument("-vs", "--videosource", help="-vs r for robot camera, -vs w for webcam.")
ap.add_argument("-t","--threshVal", default=127, type = int, help="Threshold val for edge detection")
ap.add_argument("-m", "--minimum", default=200, type=int, help="The minimum number of pixels to be inside a contour to render it valid")
ap.add_argument("-v", "--video", default=None, help="The path to the input video")
args = vars(ap.parse_args())

# Init templates
sign_list = []
templates_dir = "modules/classify/"
FILETYPE = ".png"

# these names are in the same order as the saved template images at modules/classify/templates. do not change the order of this array.
names = ["Explosives 1.1 1", "Explosives 1.2 1", "Explosives 1.3 1", "Explosives 1.4 1", "Blasting Agents 1.5 1", "Explosives 1.6 1", "Flammable Gas 2", "Non-Flammable Gas 2",
"Oxygen 2", "Inhalation Hazard", "Flammable 3", "Gasoline 3", "Combustible 3", "Fuel Oil 3", "Dangerous When Wet 4", "Flammable Solid 4", "Spontaneously Combustible 4",
"Oxidizer 5.1", "Organic Peroxide 5.2", "Inhalation Hazard 6", "Poison 6", "Toxic 6", "Radioactive 7", "Corrosive 8", "Other Dangerous Goods 9", "Dangerous"]

for i in range(1, 27):
    sign_list.append(Sign(templates_dir+"templates/"+str(i) + FILETYPE, names[i-1]))


if args['videosource'] == "r":
    # robot
    print("using robot camera")
    vs = VideoStream(src="http://"+ip+":8081").start()
else:
    # webcam
    if args["video"] == None:
        print("using local computer webcam")
        vs = VideoStream(src=0).start()
        time.sleep(2.0) #comically large spoon
    # video file
    else:
        print("using an imported video file")
        vs = cv2.VideoCapture(args["video"])



# loop over the frames of the video
while True:
    frame = vs.read()
    frame = frame if args["video"] == None else frame[1]

    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if frame is None:
        break

    # operate
    frame = imutils.resize(frame, width=500) # the frame
    descs = HU.colorShape(frame, colors=False, thresh=int(args['threshVal'])) # info about the frame - gets contours etc
    mask = np.zeros(frame.shape[:2], dtype="uint8") # a new black image the same size as the frame
    sqrs = np.zeros([0,5]) # a new array to store the bounding boxes
    for (contour, centre, desc) in descs: # for each contour in the info about the frame
        # if it is a square and is big enough
        if "square" in desc.lower() and cv2.contourArea(contour) > int(args['minimum']):
            sqrs = np.append(sqrs, HU.boundingBox(contour),0) # append it to the bounding boxes
            cv2.drawContours(mask, [contour], -1, 255, -1) # draw it onto the black image in a white colour

    # this combines them into one image using the and operation on each pixel
    masked = cv2.bitwise_and(frame, frame, mask=mask)
    # choosing an image to be drawn to the screen
    draw_image = frame

    # calculate
    for bounding_box in sqrs:

        # box
        # grab just the area of the located sign from the image, instead of the entire image
        x1, y1, x2, y2 = int(bounding_box[0]), int(bounding_box[1]), int(bounding_box[2]), int(bounding_box[3])
        region = masked[y1:y2, x1:x2]

        # constants
        text = classify(region, sign_list)
        text_x = int(x1 + (x2-x1)/2)
        text_y = int(y1 + (y2-y1)/2)
        colour = (255, 255, 255)
        black = (0, 0, 0)
        font_size = 0.5
        font_thickness = 1
        buff = 0

        # get text position etc
        (text_width_1, text_height_1) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_size, thickness=font_thickness)[0]

        # get coords of rectangle behind text
        box_coords_1 = ((text_x + buff, text_y + buff), (text_x + text_width_1 - 2*buff, text_y - text_height_1 - 2*buff))

        # draw rectangle behind text
        cv2.rectangle(draw_image, box_coords_1[0], box_coords_1[1], black, cv2.FILLED)

        # draw text
        cv2.putText(draw_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_size, colour, font_thickness)

    cv2.imshow("Camera Feed", draw_image) # images[0] is a combination of the mask and the main image side by side

    # if the "q" key is pressed, break from the loop
    key = cv2.waitKey(500) & 0xFF
    if key == ord("q"):
        break

# cleanup the camera and close any open windows
vs.stop() if args["video"] == None else vs.release()
cv2.destroyAllWindows()
