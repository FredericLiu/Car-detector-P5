import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pipelineClass import pipeline


'''
read in the distortion coeffects from CamerCali.py
'''
dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]


'''
read in the trained scv and relative parameters from TrainSVC.py
'''
carDetection_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
svc = carDetection_pickle["svc"]
X_scaler = carDetection_pickle["scaler"]
orient = carDetection_pickle["orient"]
pix_per_cell = carDetection_pickle["pix_per_cell"]
cell_per_block = carDetection_pickle["cell_per_block"]
spatial_size = carDetection_pickle["spatial_size"]
hist_bins = carDetection_pickle["hist_bins"]

#perform the pipeline on the single image

image = mpimg.imread('test_images/test5.jpg')
ld = pipeline(mtx,dist, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
#result = ld.laneDetection(image)
result = ld.carDetection(image)
plt.imshow(result)
plt.show()


'''
fig = plt.figure()
plt.subplot(121)
plt.imshow(result)
plt.subplot(122)
plt.imshow(heat_img)
plt.show()
'''
'''
#perform the pipeline on the video
from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

#ffmpeg_extract_subclip('project_video.mp4', 22, 27, targetname=project_video_sample_path)
detector = pipeline(mtx,dist, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
clip1 = VideoFileClip('project_video.mp4')
project_video_clip = clip1.fl_image(detector.carDetection) #NOTE: this function expects color images!!
project_video_clip.write_videofile('output_images/lanes_project_video.mp4', audio=False)
'''










