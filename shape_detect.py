'''
# imports
import modules.HOGUtils as HU
import argparse
import cv2
import numpy as np

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required=True, help="path to the input image")
ap.add_argument("-sup", "--suppression", required=True, type=float, help="overlap area before suppression")
ap.add_argument("-s", "--scale", type=float, default=1.5, help="scale factor size")
ap.add_argument("-w", "--winSize", default=(128,128), help="sliding window size")
ap.add_argument("-it","--invertedThresh", default="false",help="Uses BINARY_THRESH_INV")
ap.add_argument("-t","--threshVal", default=127, type = int, help="Threshold val for edge detection")
args = vars(ap.parse_args())


# start doing video feed
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')

while True:

    # get frame
    ret, frame = cap.read()
    image = frame
    copy = image.copy()
    suppressed = HU.suppressed_AOI(image, args["suppression"], winSize = args["winSize"],
            scale = args["scale"], invertedThresh = HU.boolInput(args["invertedThresh"]), thresh = args["threshVal"])     

    for row in suppressed:
            # draw
            cv2.rectangle(copy, (row[0], row[1]), (row[2], row[3]), (0,255,0))

            # region and x, y, w, h
            temp = image[row[1]:row[3], row[0]:row[2]]
            (x, y, w, h) = (row[0], row[1], row[2], row[3])

            # if the region is not nothing (is something)
            if np.size(np.ndarray.flatten(temp)) > 0:
                descs = HU.colorShape(temp, thresh = args["threshVal"])
                # for (cnt, centre, desc) in descs:
                #     if cv2.contourArea(cnt) > 300:
                #         cv2.drawContours(temp, cnt, -1, (0,255,0))
                #         pos = (x+centre[0], y+centre[1])
                #         cv2.putText(copy, desc, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                biggest_desc_index = 0
                for i in range(len(descs)):
                    if cv2.contourArea(descs[i][0]) > cv2.contourArea(descs[biggest_desc_index][0]):
                        biggest_desc_index = i
                cv2.drawContours(temp, descs[biggest_desc_index][0], -1, (0,255,0))
                pos = (x+descs[biggest_desc_index][1][0], y+descs[biggest_desc_index][1][1])
                cv2.putText(copy, descs[biggest_desc_index][2], pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Image", copy)
    cv2.waitKey(0)

    # close window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
'''

