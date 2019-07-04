import cv2
from pyzbar.pyzbar import decode
import imutils
from imutils.video import VideoStream
import os

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

# Uncomment to use web stream from robot
#vs = VideoStream(src="http://10.0.2.4:8081").start()
vs = VideoStream(src=0).start()
#times = [] # to tell how fast it was
codes = {} # to be saved

while (True):
        # read from camera source
        im = vs.read()

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
            codes[decodedString] = region

            # create text
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(contrast_image, decodedString, (50 ,60) ,font , 1, (200,255,155), 2, cv2.LINE_AA)

        # print to console
        #if (not decoded == ""):
        #        print(decoded)
        if (decodedString != ""):
            print(decodedString)

        # draw keypoints window
        #cv2.imshow("Keypoints", im)
        # draw histogram window
        cv2.imshow("Contrasted (Equalize Histogram)", contrast_image)

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