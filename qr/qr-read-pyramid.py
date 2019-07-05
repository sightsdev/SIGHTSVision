
# imports
import cv2
from pyzbar.pyzbar import decode
import imutils
from imutils.video import VideoStream
import os
import argparse
from pyramid import *
from modules.HOGUtils import pyramid, sliding_window

# argparse
ap = argparse.ArgumentParser()
ap.add_argument('-vs', '--videosource', default='r', help='-vs r for robot, -vs w for webcam.')
ap.add_argument('-s', '--scale', default=2, help='image pyramid scale')
args = vars(ap.parse_args())

# ip
ip = "10.0.0.4"

# funcs
def getNumber(line, prop):
    numbers = []
    count = 0
    found_equal = False
    if prop in line:
        for character in line:
            if found_equal == True:
                numbers.append(line[count])
            if character == 'b':
                found_equal = True
            if character == "'" and line[count + 1] == ",":
                return numbers
            count = count + 1
            return numbers
def dataArrayToString(line, prop):
    number_string = ""
    number = getNumber(line, prop)
    if number == None:
        return ""
    for num in number:
        number_string += num
    return number_string

# grab stream
if args['videosource'] == 'w':
    source = VideoStream(src=0).start()
elif args['videosource'] == 'r':
    source = VideoStream(src="http://"+ip+":8081").start()

# save codes
codes = {}

# loop on video stream
while (True):
    # read front camera
    im = source.read()
    # convert to grayscale
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # increase contrast
    contrast_image = cv2.equalizeHist(im)

    # pyramid, ratio
    for current_level, ratio in pyramid(im, float(args["scale"])):
        dimensions = (int(args['winsize']), int(args['winsize']))
        for x, y, window in sliding_window(current_level, int(args['stepsize']), dimensions):
            pass

    '''

    # decode QR code
    decoded = decode(contrast_image)
    print(decoded)
    decodedString = dataArrayToString(str(decoded), "data=")
        
    # if theres something in the output of decoded()   
    if len(decoded) == 1:

        # do the region
        [x, y, w, h] = decoded[0].rect
        [x1, y1, x2, y2] = [x, y, x+w, y+h]
        region = contrast_image[y1:y2, x1:x2]

        # print to console, if there is not nothing (= if there is something)
        if (decodedString != ""):
            print(decodedString)

        # add the string and the region to the codes dict
        codes[decodedString] = region

        # draw keypoints window
        #cv2.imshow("Keypoints", im)
        # draw histogram window
    cv2.imshow("Camera", contrast_image)
    # exit condition
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    '''

# chuck the codes in a fil
f = open("res/results.txt","w")

for key in codes:
    f.write(key+"\n")
    filename = "".join(x for x in key if x.isalnum())
    cv2.imwrite("res/"+filename+".png", codes[key])

# close f
f.close()

# destroy all windows
cv2.destroyAllWindows()
