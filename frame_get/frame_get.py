
# save frames from a video as separate images
# to gather training data for a model

import os
import cv2

out_name = "output"

if not os.path.exists(out_name):
    os.makedirs(out_name)

vidcap = cv2.VideoCapture('sample.mp4')
success,image = vidcap.read()
frame_count = 0
step = 20
while success:
    success,image = vidcap.read()
    if frame_count % step == 0:
        cv2.imwrite(f"{out_name}/frame%d.jpg" % (frame_count//step), image)     # save frame as JPEG file      
        print('Read a new frame: ', success)
    frame_count += 1
