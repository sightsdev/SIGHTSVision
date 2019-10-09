
# imports
import cv2
from pyzbar.pyzbar import decode
import imutils
from imutils.video import VideoStream
import os
import argparse
from modules.HOGUtils import pyramid, sliding_window

# argparse
ap = argparse.ArgumentParser()
ap.add_argument('-vs', '--videosource', default='w', help='-vs r for robot, -vs w for webcam.')
ap.add_argument('-s', '--scale', default=1.5, help='image pyramid scale')
ap.add_argument('-ss', '--stepsize', default=32, help='size of step')
ap.add_argument('-ws', '--winsize', default=64, help='side length of sliding window')
args = vars(ap.parse_args())

# ip
ip = "10.0.0.4"

# funcs
def genDimensions(start_length, scale, w, h):
    # w and h are the width and height of the original window
    max_length = max(w, h)
    dimensions = [start_length]
    while dimensions[-1] < max_length:
        dimensions.append(dimensions[-1]*scale)
    dimensions[-1] = max_length
    print(dimensions)
    return dimensions

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
    for side_length in genDimensions(int(args['winsize']), float(args['scale']), im.shape[1], im.shape[0]):
        for x, y, window in sliding_window(contrast_image, int(args['stepsize']), (int(side_length), int(side_length))):

            # decode the sliding window contents
            decoded = decode(window)
            decodedString = dataArrayToString(str(decoded), "data=")

            # if anything was read
            if len(decoded) == 1:
                print('qr code detected.')
                if (decodedString != ""):
                    print(decodedString)

                # get the region
                [x_r, y_r, w_r, h_r] = decoded[0].rect # get x/y/w/h
                # get x, y of top-left and bottom-right
                [x1_r, y1_r, x2_r, y2_r] = [x_r, y_r, x_r+w_r, y_r+h_r]
                # make x,y of top-left and bottom-right relative to the main image
                [x1_r, y1_r, x2_r, y2_r] = [x+x1_r, y+y1_r, x+x2_r, y+y2_r]
                # draw it on the frame
                cv2.rectangle(im, (x1_r, y1_r), (x2_r, y2_r), (0, 255, 0), 2)

                # save the region and the decoded string
                codes[decodedString] = im[int(y1_r):int(y2_r), int(x1_r):int(x2_r)]

    cv2.imshow("Frame", im)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

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
