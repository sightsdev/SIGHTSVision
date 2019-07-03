
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
	
	def __init__(self, image, name, short):
		self.name = name
		self.short = short
		self.image = image
		self.col = 0
		self.bff = 0
		self.bff_data = []

# Helper function to convert an image from BGR to RGB
def rgb(bgr_image):
	return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

# Helper function to plot a histogram on a graph, modified to draw on a subplot
# Code sourced from https://lmcaraig.com/image-histograms-histograms-equalization-and-histograms-comparison/
#def draw_image_histogram(image, channels, rcn, color='k' ):
#    hist = cv2.calcHist([image], channels, None, [BINS], [0, 256])
#    plt.subplot(rcn),plt.plot(hist, color=color)
#    plt.subplot(rcn),plt.xlim([0, BINS])

# Draw the color histogram on BGR channels, sourced from https://lmcaraig.com/image-histograms-histograms-equalization-and-histograms-comparison/
#def show_color_histogram(image, rcn):
#    for i, col in enumerate(['b', 'g', 'r']):
#        draw_image_histogram(image, [i], rcn, color=col)

# Draw BFF matches (sourced and modified from http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html)
# def show_bff_matches(image, title, template, good, rcn):
# 	# Initiate SIFT detector
# 	sift = cv2.xfeatures2d.SIFT_create()
	
# 	img1 = image
# 	img2 = template

# 	# find the keypoints and descriptors with SIFT
# 	kp1, des1 = sift.detectAndCompute(img1,None)
# 	kp2, des2 = sift.detectAndCompute(img2,None)
	
	
# 	src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
# 	dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
	
# 	M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
# 	matchesMask = mask.ravel().tolist()
    
# 	draw_params = dict(matchColor = (0,255,0), # draw matches in green color
#                    singlePointColor = None,
#                    matchesMask = matchesMask, # draw only inliers
#                    flags = 2)

# 	img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

# 	plt.subplot(rcn),plt.imshow(rgb(img3)),plt.title(title)

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

sign_list = []

# Templates downloaded from the google group discussion @ https://groups.google.com/forum/#!topic/oarkit/T2ACYFXZ0XA
templates_dir = "C:/Users/GAMBALAJ/Documents/Files/Development/SART/alex/modules/classify/"
sign_list.append(Sign(templates_dir+"templates/1" + FILETYPE, "Explosives 1.1 1", ""))
sign_list.append(Sign(templates_dir+"templates/2" + FILETYPE, "Explosives 1.2 1", ""))
sign_list.append(Sign(templates_dir+"templates/3" + FILETYPE, "Explosives 1.3 1", ""))
sign_list.append(Sign(templates_dir+"templates/4" + FILETYPE, "Explosives 1.4 1", ""))
sign_list.append(Sign(templates_dir+"templates/5" + FILETYPE, "Blasting Agents 1.5 1", ""))
sign_list.append(Sign(templates_dir+"templates/6" + FILETYPE, "Explosives 1.6 1", ""))
sign_list.append(Sign(templates_dir+"templates/7" + FILETYPE, "Flammable Gas 2", ""))
sign_list.append(Sign(templates_dir+"templates/8" + FILETYPE, "Non-Flammable Gas 2", ""))
sign_list.append(Sign(templates_dir+"templates/9" + FILETYPE, "Oxygen 2", ""))
sign_list.append(Sign(templates_dir+"templates/10" + FILETYPE, "Inhalation Hazard", ""))
sign_list.append(Sign(templates_dir+"templates/11" + FILETYPE, "Flammable 3", ""))
sign_list.append(Sign(templates_dir+"templates/12" + FILETYPE, "Gasoline 3", ""))
sign_list.append(Sign(templates_dir+"templates/13" + FILETYPE, "Combustible 3", ""))
sign_list.append(Sign(templates_dir+"templates/14" + FILETYPE, "Fuel Oil 3", ""))
sign_list.append(Sign(templates_dir+"templates/15" + FILETYPE, "Dangerous When Wet 4", ""))
sign_list.append(Sign(templates_dir+"templates/16" + FILETYPE, "Flammable Solid 4", ""))
sign_list.append(Sign(templates_dir+"templates/17" + FILETYPE, "Spontaneously Combustible 4", ""))
sign_list.append(Sign(templates_dir+"templates/18" + FILETYPE, "Oxidizer 5.1", ""))
sign_list.append(Sign(templates_dir+"templates/19" + FILETYPE, "Organic Peroxide 5.2", ""))
sign_list.append(Sign(templates_dir+"templates/20" + FILETYPE, "Inhalation Hazard 6", ""))
sign_list.append(Sign(templates_dir+"templates/21" + FILETYPE, "Poison 6", ""))
sign_list.append(Sign(templates_dir+"templates/22" + FILETYPE, "Toxic 6", ""))
sign_list.append(Sign(templates_dir+"templates/23" + FILETYPE, "Radioactive 7", ""))
sign_list.append(Sign(templates_dir+"templates/24" + FILETYPE, "Corrosive 8", ""))
sign_list.append(Sign(templates_dir+"templates/25" + FILETYPE, "Oher Dangerous Goods 9", ""))
sign_list.append(Sign(templates_dir+"templates/26" + FILETYPE, "Dangerous", ""))


# Create a figure and resize it (I think units of measurement are inches)
#fig = plt.figure("Query", figsize=(12,8))

image = cv2.imread("C:/Users/GAMBALAJ/Documents/Files/Development/SART/alex/modules/classify/test.png")

# ax = fig.add_subplot(3, 6, 1)
# ax.imshow(rgb(image))
# ax.set_title("Square detected:")
# plt.axis("off")
# show_color_histogram(image, 363)

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
'''
ax = fig.add_subplot(3, 6, 4)
ax.imshow(rgb(best_image))
ax.set_title("Best match")
plt.axis("off")
'''

# show_bff_matches(image, "Best match: \n" + best.name, best_image, best.bff_data, 364)
# show_color_histogram(best_image, 366)
# plt.show()

print(best.name)
