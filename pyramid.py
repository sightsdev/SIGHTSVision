# import the necessary packages
from modules.HOGUtils import pyramid
import argparse
import cv2
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-s", "--scale", type=float, default=1.5, help="scale factor size")
args = vars(ap.parse_args())
 
# load the image
image = cv2.imread(args["image"])
 
# METHOD #1: No smooth, just scaling.
# loop over the image pyramid
for (i, (resized, _)) in enumerate(pyramid(image, scale=args["scale"])):
	# show the resized image
	cv2.imshow("Layer {}".format(i + 1), resized)
	cv2.waitKey(0)
 
# close all windows
cv2.destroyAllWindows()
