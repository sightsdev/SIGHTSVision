import cv2
from pyzbar.pyzbar import decode
import imutils
from imutils.video import VideoStream
from time import time
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
times = [] # to tell how fast it was
codes = [] # to be saved

while (True):
        # read from camera source
        im = vs.read()
        time_at_start = time()

        # convert to grayscale
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        
        # increase contrast
        contrast_image = cv2.equalizeHist(im)

        # decode QR code
        decoded = str(decode(contrast_image))
        decodedString = dataArrayToString(decoded, "data=")

        # create text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(contrast_image, decodedString, (50 ,60) ,font , 1, (200,255,155), 2, cv2.LINE_AA)
        if decodedString not in codes:
            codes.append(decodedString)

        # print to console
        #if (not decoded == ""):
        #        print(decoded)
        if (decodedString != ""):
                    print(decodedString)

        # draw keypoints window
        #cv2.imshow("Keypoints", im)
        # draw histogram window
        cv2.imshow("Contrasted (Equalize Histogram)", contrast_image)

        # get the time it took and chuck it in times
        time_at_end = time()
        time_taken = time_at_end - time_at_start
        times.append(time_taken)

        # exit condition
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
                break

# get the average time taken and print it
total = 0
for time in times:
        total += time
avg = total/len(times)
print("Each frame took on average "+str(avg)+" seconds to compute.")
print("The average speed was "+str(1/avg)+" frames per second.")

# chuck the codes in a file
f = open("results.txt","w")
for code in codes:
    f.write(code+"\n")
f.close()
input()
cv2.destroyAllWindows()

