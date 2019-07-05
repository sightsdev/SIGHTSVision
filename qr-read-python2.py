import cv2
from pyzbar.pyzbar import decode
import imutils
from imutils.video import VideoStream
import os

# ip
ip = "10.0.0.3"

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

def readCode(image):
    # read from camera source
    im = image

    # convert to grayscale
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        
    # increase contrast
    contrast_image = cv2.equalizeHist(im)

    # decode QR code
    decoded = decode(contrast_image)
    decodedString = dataArrayToString(str(decoded), "data=")
        
    # if theres something in the output of decoded()   
    if len(decoded) == 1:

        # do the region
        [x, y, w, h] = decoded[0].rect
        [x1, y1, x2, y2] = [x, y, x+w, y+h]
        region = contrast_image[y1:y2, x1:x2]

        # return
        return decodedString, region
    else:
        return None, None

# Uncomment to use web stream from robot
front_camera = VideoStream(src="http://"+ip+":8081").start()
side_1_camera = VideoStream(src="http://"+ip+":8082").start() # left/right could be the other way round
side_2_camera = VideoStream(src="http://"+ip+":8083").start()
#vs = VideoStream(src=0).start()
#times = [] # to tell how fast it was
codes = {} # to be saved

while (True):
        # arrange data
        images = [front_camera.read(), side_1_camera.read(), side_2_camera.read()]
        decodedStrings, regions = [], []

        # collect decoded strings and regions
        for image in images:
            decodedString, region = readCode(image)
            decodedStrings.append(decodedString)
            regions.append(readCode(region))

        # print to console, if there is not nothing (= if there is something)
        if (decodedStrings != [[""],[""],[""]]):
            print decodedStrings

        # add the string and the region to the codes dict
        for i in range(2):
            codes[decodedSrings[i]] = regions[i]

        # draw keypoints window
        #cv2.imshow("Keypoints", im)
        # draw histogram window
        cv2.imshow("Front", images[0])
        cv2.imshow("Side 1", images[1])
        cv2.imshow("Side 2", images[2])

        # exit condition
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
                break

# chuck the codes in a file
f = open("results.txt","w")

for key in codes:
    f.write(key+"\n")
    filename = "".join(x for x in key if x.isalnum())
    cv2.imwrite(filename+".png", codes[key])

# close f
f.close()

# destroy all windows
cv2.destroyAllWindows()
