import imutils
import cv2
import argparse
import numpy as np
import math
from modules.colorlabeler import ColorLabeler
from modules.shapedetector import ShapeDetector


# turns common string representations of booleans into true/false
def boolInput(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
                return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
                return False
        else:
                raise argparse.ArgumentTypeError('Boolean value expected.')


# generates an image pyramid
def pyramid(image, scale=1.5, minSize=(30, 30)):
	ODim = image.shape[1]
	# yield the original image
	yield (image, 1)
 
	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)
		ratio = ODim/float(image.shape[1])
		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break
 
		# yield the next image in the pyramid
		yield (image, ratio)


# generates a sliding window (really just a list of image regions. 'sliding' is just a visualisation)
def sliding_window(image, stepSize, windowSize = (128, 128)):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


# gets contours and return them in bounding box form
def findStuff(image, inverted, threshval = 127, ratio = 1):
##        # load the image and resize it to a smaller factor so that
##        # the shapes can be approximated better
##        resized = imutils.resize(image, width=300)
##        ratio = image.shape[0] / float(resized.shape[0])

        # blur the resized image slightly, then convert it to both
        # grayscale and the L*a*b* color spaces
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        # i dont think this is used
        #lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        if inverted:
                thresh = cv2.threshold(gray, threshval, 255, cv2.THRESH_BINARY_INV)[1]
        else:
                thresh = cv2.threshold(gray, threshval, 255, cv2.THRESH_BINARY)[1]
        
        # find contours in the thresholded image
        # the middle argument used to be cv2.RETR_EXTERNAL but i see RETR_TREE more often so i went with that
        # just in case
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ## ANTHONY COMMENT
        ## the following line might be an issue, because in the bounding box code, there were some
        ## problems with the array not being the right size.
        # i just decided to comment it out even though it might do something i need, but i haven't seen it in
        # any tutorials i did for opencv4 so yeah
        #cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        cnts = imutils.grab_contours(cnts)

        # loop over the contours
        boxes = np.zeros([0,5]) # an empty array
        for c in cnts: # iterate over all contours
                #print(np.shape(bbox))
                boxes = np.append(boxes, boundingBox(c),0) # add bounding box to self
                #cv2.rectangle(image, (extLeft,extTop),(extRight,extBot), (0,255,0))
                #cv2.drawContours(image, [c], -1, (0, 0, 255))
        #print(boxes)
        # show the output image
        #cv2.imshow("Image", image)
        #cv2.waitKey(0)
        return boxes # return all the bounding boxes


# turn a single contour into a single bounding box
def boundingBox(c):
        ## ANTHONY COMMENT:
        ## print(c) shows an array looking like [[a b c d]], which is probably why it doesn't
        ## work super well with the code below. the contour should be a series of points
        extLeft = c[c[:, :, 0].argmin()][0][0]
        extRight = c[c[:, :, 0].argmax()][0][0]
        extTop = c[c[:, :, 1].argmin()][0][1]
        extBot = c[c[:, :, 1].argmax()][0][1]
        Left = c[c[:, :, 0].argmin()][0]
        Bottom = c[c[:, :, 1].argmax()][0]
        
        # s2 = math.sqrt(2*math.sqrt(math.pow(Left[0]-Bottom[0],2)+math.pow(Left[1]-Bottom[1],2)))
        # A = cv2.moments(c)["m00"]
        # #        ((width)*(height)-momentArea)/
        # inner = ((extRight-extLeft)*(extBot-extTop)-A)/math.pow(s2/2,2)
        # inner2 = (4*math.pow(s2/2,2)-inner)/(4*math.pow(s2/2,2))
        # #angle = math.degrees(math.acos(inner2))/2
        angle  = 0
        return np.array([[extLeft,extTop,extRight,extBot,angle]])


# maybe does something for the nonmax suppression
def suppressed_AOI(image, suppression, winSize = (128, 128), scale = 1.5, invertedThresh = False, thresh = 127):
        (winW, winH) = winSize
        boxes = np.zeros([0,5])
        # loop over the image pyramid
        for (resized, ratio) in pyramid(image, scale=scale):
                # loop over the sliding window for each layer of the pyramid
                for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
                        # if the window does not meet our desired window size, ignore it
                        if window.shape[0] != winH or window.shape[1] != winW:
                                continue
                        newb = findStuff(window.copy(), invertedThresh, threshval = thresh)
                        if np.shape(newb)[0] != 0:
                                #print(newb) 
                                newb = np.multiply(np.add(newb,np.array([x,y,x,y, 0])),ratio)
                                #print(newb)
                                #print(np.shape(newb))
                                #print(np.shape(boxes))
                                boxes = np.append(boxes,newb,0)
                        #print(boxes)
        return non_max_suppression_fast(boxes, suppression)



# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
 
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
 
	# initialize the list of picked indexes	
	pick = []
 
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
 
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
 
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
 
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
 
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
 
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
 
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")


# returns a description of the object
def colorShape(image, shapes = True, colors = True, thresh = 127):
        # blur the resized image slightly, then convert it to both
        # grayscale and the L*a*b* color spaces
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)

        # changed the system so that the contours are the inverted and non inverted ones both concatenated.
        # this is cause sometimes the inverted one did better and sometimes the non inverted one did better
        #if inverted:
        #        thresholded = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)[1]
        #else:
        #        thresholded = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]
        thresholded = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]
        inverted = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)[1]

        # find contours in the thresholded image
        _, cnts_a, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        _, cnts_b, _ = cv2.findContours(inverted.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # commented out the following line because i havent seen this before, and it might not really be relevant
        # in opencv 4. but i dont know, if theres an error here try uncommenting this
        #cnts = cnts[0] if imutils.is_cv2() else cnts[1]
         
        # initialize the shape detector and color labeler
        sd = ShapeDetector()
        cl = ColorLabeler()
        desc = []
        # loop over the contours
        for c in cnts_a:
                # compute the center of the contour
                M = cv2.moments(c)
                if M["m00"] != 0:
                        cX = int((M["m10"] / M["m00"]))
                        cY = int((M["m01"] / M["m00"]))
                else:
                        cX = int((M["m10"] / (M["m00"]+1)))
                        cY = int((M["m01"] / (M["m00"]+1)))
         
                # detect the shape of the contour and label the color
                if shapes:
                        shape = sd.detect(c)
                else:
                        shape = ""
                if colors:
                        color = cl.label(lab, c)
                else:
                        color = ""
         
                # multiply the contour (x, y)-coordinates by the resize ratio,
                # then draw the contours and the name of the shape and labeled
                # color on the image
                text = "{} {}".format(color, shape)
                #cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
                #cv2.putText(image, text, (cX, cY),
                #	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                #desc.append(tuple([c,(cX, cY), text]))
                desc.append((c, (cX,cY), text))
        # loop over more contours
        for c in cnts_b:
                # compute the center of the contour
                M = cv2.moments(c)
                if M["m00"] != 0:
                        cX = int((M["m10"] / M["m00"]))
                        cY = int((M["m01"] / M["m00"]))
                else:
                        cX = int((M["m10"] / (M["m00"]+1)))
                        cY = int((M["m01"] / (M["m00"]+1)))
         
                # detect the shape of the contour and label the color
                if shapes:
                        shape = sd.detect(c)
                else:
                        shape = ""
                if colors:
                        color = cl.label(lab, c)
                else:
                        color = ""
         
                # multiply the contour (x, y)-coordinates by the resize ratio,
                # then draw the contours and the name of the shape and labeled
                # color on the image
                text = "{} {}".format(color, shape)
                #cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
                #cv2.putText(image, text, (cX, cY),
                #	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                #desc.append(tuple([c,(cX, cY), text]))
                desc.append((c, (cX,cY), text))
        return desc


'''
# the old colorshape function that only did either inverted or not inverted, not both at once
def colorShapeOld(image, inverted = False, shapes = True, colors = True, thresh = 127):
        # blur the resized image slightly, then convert it to both
        # grayscale and the L*a*b* color spaces
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)

        if inverted:
                thresholded = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)[1]
        else:
                thresholded = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]

        # find contours in the thresholded image
        cnts, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
         
        # initialize the shape detector and color labeler
        sd = ShapeDetector()
        cl = ColorLabeler()
        desc = []
        # loop over the contours
        for c in cnts:
                # compute the center of the contour
                M = cv2.moments(c)
                if M["m00"] != 0:
                        cX = int((M["m10"] / M["m00"]))
                        cY = int((M["m01"] / M["m00"]))
                else:
                        cX = int((M["m10"] / (M["m00"]+1)))
                        cY = int((M["m01"] / (M["m00"]+1)))
         
                # detect the shape of the contour and label the color
                if shapes:
                        shape = sd.detect(c)
                else:
                        shape = ""
                if colors:
                        color = cl.label(lab, c)
                else:
                        color = ""
         
                # multiply the contour (x, y)-coordinates by the resize ratio,
                # then draw the contours and the name of the shape and labeled
                # color on the image
                text = "{} {}".format(color, shape)
                #cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
                #cv2.putText(image, text, (cX, cY),
                #	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                #desc.append(tuple([c,(cX, cY), text]))
                desc.append((c, (cX,cY), text))
        return desc
'''
