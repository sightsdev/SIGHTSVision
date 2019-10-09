
# imports
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
import cv2

# Brute Force Feature matching 
MIN_MATCH_COUNT = 10
MIN_MATCH_RATING = 0.7

# Colour histogram comparisons
BINS = 4
COL_COMP_METHOD = cv2.HISTCMP_INTERSECT # Choose from: cv2.HISTCMP_CORREL, cv2.HISTCMP_CHISQR, cv2.HISTCMP_INTERSECT, cv2.HISTCMP_BHATTACHARYYA
FILETYPE=".png"

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
	kp1, des1 = sift.detectAndCompute(image, None) # apparently this only works with 8-bit images
	kp2, des2 = sift.detectAndCompute(template, None)

	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks = 50)

	flann = cv2.FlannBasedMatcher(index_params, search_params)

	matches = flann.knnMatch(des1, des2, k=2)

	# store all the good matches as per Lowe's ratio test.
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
        sign.bff_data = bff_match(image, template)
        sign.bff = len(sign.bff_data)
        sign.col = color_match(image, template)
        
    # Sort the results using the bff match attribute in reverse order
    sign_list.sort(key=lambda x: x.bff, reverse=True)

    # Loop through signs to display sorted images, and match data
    for i, sign in enumerate(sign_list):
        #ax = fig.add_subplot(3, 6, i+7)
        template = cv2.imread(sign.image)
        #ax.imshow(rgb(template))
        #ax.set_title("B: %.0f, C: %.3f" % (sign.bff, sign.col))
        #plt.axis("off")

    best = sign_list[0]
    best_image = cv2.imread(best.image)

    return best.name
