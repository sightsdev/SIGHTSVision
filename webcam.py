
import modules.HOGUtils as HU
import imutils
import argparse
import cv2
import numpy as np
from math import pi, degrees
from modules.classify.classify_abstracted import *
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-sup", "--suppression", required=True, type=float, help="overlap area before suppression")
#ap.add_argument("-s", "--scale", type=float, default=1.5, help="scale factor size")
#ap.add_argument("-w", "--winSize", default=(128,128), help="sliding window size")
#ap.add_argument("-it","--invertedThresh", default="false",help="Uses BINARY_THRESH_INV")
ap.add_argument("-t","--threshVal", default=127, type = int, help="Threshold val for edge detection")
ap.add_argument("-m", "--minimum", default=200, type=int, help="The minimum number of pixels to be inside a contour to render it valid")
args = vars(ap.parse_args())

# start doing video feed
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Init templates
sign_list = []
templates_dir = "C:/Users/GAMBALAJ/Documents/Files/Development/SART/alex/modules/classify/"
sign_list.append(Sign(templates_dir+"templates/1" + FILETYPE, "Explosives 1.1 1"))
sign_list.append(Sign(templates_dir+"templates/2" + FILETYPE, "Explosives 1.2 1"))
sign_list.append(Sign(templates_dir+"templates/3" + FILETYPE, "Explosives 1.3 1"))
sign_list.append(Sign(templates_dir+"templates/4" + FILETYPE, "Explosives 1.4 1"))
sign_list.append(Sign(templates_dir+"templates/5" + FILETYPE, "Blasting Agents 1.5 1"))
sign_list.append(Sign(templates_dir+"templates/6" + FILETYPE, "Explosives 1.6 1"))
sign_list.append(Sign(templates_dir+"templates/7" + FILETYPE, "Flammable Gas 2"))
sign_list.append(Sign(templates_dir+"templates/8" + FILETYPE, "Non-Flammable Gas 2"))
sign_list.append(Sign(templates_dir+"templates/9" + FILETYPE, "Oxygen 2"))
sign_list.append(Sign(templates_dir+"templates/10" + FILETYPE, "Inhalation Hazard"))
sign_list.append(Sign(templates_dir+"templates/11" + FILETYPE, "Flammable 3"))
sign_list.append(Sign(templates_dir+"templates/12" + FILETYPE, "Gasoline 3"))
sign_list.append(Sign(templates_dir+"templates/13" + FILETYPE, "Combustible 3"))
sign_list.append(Sign(templates_dir+"templates/14" + FILETYPE, "Fuel Oil 3"))
sign_list.append(Sign(templates_dir+"templates/15" + FILETYPE, "Dangerous When Wet 4"))
sign_list.append(Sign(templates_dir+"templates/16" + FILETYPE, "Flammable Solid 4"))
sign_list.append(Sign(templates_dir+"templates/17" + FILETYPE, "Spontaneously Combustible 4"))
sign_list.append(Sign(templates_dir+"templates/18" + FILETYPE, "Oxidizer 5.1"))
sign_list.append(Sign(templates_dir+"templates/19" + FILETYPE, "Organic Peroxide 5.2"))
sign_list.append(Sign(templates_dir+"templates/20" + FILETYPE, "Inhalation Hazard 6"))
sign_list.append(Sign(templates_dir+"templates/21" + FILETYPE, "Poison 6"))
sign_list.append(Sign(templates_dir+"templates/22" + FILETYPE, "Toxic 6"))
sign_list.append(Sign(templates_dir+"templates/23" + FILETYPE, "Radioactive 7"))
sign_list.append(Sign(templates_dir+"templates/24" + FILETYPE, "Corrosive 8"))
sign_list.append(Sign(templates_dir+"templates/25" + FILETYPE, "Oher Dangerous Goods 9"))
sign_list.append(Sign(templates_dir+"templates/26" + FILETYPE, "Dangerous"))

enumerate(sign_list)
print("there might be an error before this")

while True:

    # get frame
    ret, frame = cap.read()

    # operate
    frame = imutils.resize(frame, width=500) # the frame
    descs = HU.colorShape(frame, colors=False, thresh=int(args['threshVal'])) # info about the frame
    mask = np.zeros(frame.shape[:2], dtype="uint8") # a new black image the same size as the frame
    sqrs = np.zeros([0,5]) # a new array to store the bounding boxes
    for (contour, centre, desc) in descs: # for each contour in the info about the frame
        # if it is a square and is big enough
        if "square" in desc.lower() and cv2.contourArea(contour) > int(args['minimum']):
            sqrs = np.append(sqrs, HU.boundingBox(contour),0) # append it to the bounding boxes
            cv2.drawContours(mask, [contour], -1, 255, -1) # draw it onto the black image

    # this combines them into one image
    masked = cv2.bitwise_and(frame, frame, mask=mask)
    images = imutils.build_montages([frame, masked], (frame.shape[1], frame.shape[0]), (2,1))

    # calculate
    for square in sqrs:
        x1, y1, x2, y2 = int(square[0]), int(square[1]), int(square[2]), int(square[3])
        region = masked[y1:y2, x1:x2]
        text = classify(region, sign_list)
        text_x = int(x1 + (x2-x1)/2)
        text_y = int(y1 + (y2-y1)/2)
        cv2.putText(images[0], text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(images[0], text, (text_x+masked.shape[0], text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.waitKey(0)
        os.system("cls")

    cv2.imshow("Camera Feed", images[0]) # images[0] is a combination of the mask and the main image side by side

    # close window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
