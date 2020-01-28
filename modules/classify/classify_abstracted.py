
# imports
import numpy as np
import argparse
import glob
import cv2

# Brute Force Feature matching
MIN_MATCH_RATING = 0.7

# Colour histogram comparisons
BINS = 4
COL_COMP_METHOD = cv2.HISTCMP_INTERSECT # Choose from: cv2.HISTCMP_CORREL, cv2.HISTCMP_CHISQR, cv2.HISTCMP_INTERSECT, cv2.HISTCMP_BHATTACHARYYA


# Create a class to store sign image, name and match data
class Sign(object):
	
	def __init__(self, image, name):
		self.name = name
		self.image = image
		self.col = 0
		self.bff = 0
		self.bff_data = []


# Helper function to convert an image from BGR to RGB
def rgb(bgr_image):
	return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)


# Code sourced (but highly simplified) from https://www.pyimagesearch.com/2014/07/14/3-ways-compare-histograms-using-opencv-python/
# Simply convert to a histogram
def get_hist(image):	
	hist = cv2.calcHist([image], [0, 1, 2], None, [BINS, BINS, BINS],
		[0, 256, 0, 256, 0, 256])
	return cv2.normalize(hist,hist).flatten()


# Code sourced (but highly simplified) from https://www.pyimagesearch.com/2014/07/14/3-ways-compare-histograms-using-opencv-python/
def color_match(image, template):	
	return cv2.compareHist(get_hist(image), get_hist(template), COL_COMP_METHOD)


# Code sourced from http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
def bff_match(image, template):
	# Initiate SIFT detector
	sift = cv2.xfeatures2d.SIFT_create()

	# find the keypoints and descriptors with SIFT
    # keypoints and descriptors of the image we are interested in
	kp_detection, des_detection = sift.detectAndCompute(image, None) # apparently this only works with 8-bit images
    # keypoints and descriptors of the template image
	kp_template, des_template = sift.detectAndCompute(template, None)

    # make a "flann based matcher" object which has these parameters.
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks = 50) # how many times to check something, higher number is more checks which is more accurate but slower
	flann = cv2.FlannBasedMatcher(index_params, search_params)

    # this looks through the descriptors of each image and sees which points match?
	matches = flann.knnMatch(des_detection, des_template, k=2)

	# store all the good matches as per Lowe's ratio test.
    # what this does is, only chooses the features detected that are close together. If they aren't close together, they aren't "good"
    # isn't this what flann.knnMatch is already doing?
    # yes, but this filters through those matches to find only the MOST accurate ones
	good = []
	for m,n in matches:
		if m.distance < MIN_MATCH_RATING*n.distance:
			good.append(m)

	# return the number of matches (the tutorial describes how to draw the features if interested)
	return good


# classify the input image
def classify(image, sign_list):
    
    # debug
    print("function called")

    # Loop through signs to store bff and color matches
    for i, sign in enumerate(sign_list):
        template = cv2.imread(sign.image)
        sign.bff_data = bff_match(image, template) # maybe this is a list of all the features
        sign.bff = len(sign.bff_data) # maybe this is the number of features that were detected as matches?
        sign.col = color_match(image, template)
        
    # Sort the results using the bff match attribute in reverse order
    # this only uses the bff property of the sign objects, not colour.
    sign_list.sort(key=lambda x: x.bff, reverse=True)

    best = sign_list[0]
    best_image = cv2.imread(best.image)

    return best.name
