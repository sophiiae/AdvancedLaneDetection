import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import color_warp

im = mpimg.imread('practice/images/test5.jpg')

mask = [ [683, 447], [1037, 666], [270, 666], [596, 447]]
dst = [[900, 0], [900, 666], [200, 666], [200,0]]

# region = color_warp.region(im, mask)

warped = color_warp.warp(im, mask, dst)
# [color_binary, combined_binary] = color_warp.pipeline(im)
[h, l, s] = color_warp.hls(warped)
r = warped[:, :, 0]
g = warped[:, :, 1]
b = warped[:, :, 2]
sobel = color_warp.sobel_thresh(l,'x', 14, 160)
color = color_warp.color_th(l, (90, 150))
grad = color_warp.gradient_th(l, (100, 155))

result = np.zeros_like(s)
result[(sobel == 1) & (color == 1)] = 1
color_warp.plot(h, 'h channel', 'gray', 
                l, 'l channel', 'gray', 
                s, 's channel', 'gray',
                sobel, 'sobel', 'gray', 
                color, 'color threshold', 'gray', 
                grad, 'gradient threshold', 'gray'
                )

# Assuming you have created a warped binary image called "sobel"
# Take a histogram of the bottom half of the image
histogram = np.sum(sobel[sobel.shape[0]//2:,:], axis=0)

# Create an output image to draw on and visualize the result
out_img = np.dstack((sobel, sobel, sobel))*255
# Find the peak of the left and right halves of the histogram
# These will be the starting point for the left and right lines
midpoint = np.int(histogram.shape[0]//2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint

# HYPERPARAMETERS
# Choose the number of sliding windows
nwindows = 9
# Set the width of the windows +/- margin
margin = 100
# Set minimum number of pixels found to recenter window
minpix = 50

# Set height of windows - based on nwindows above and image shape
window_height = np.int(sobel.shape[0]//nwindows)
# Identify the x and y positions of all nonzero (i.e. activated) pixels in the image
nonzero = sobel.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
# Current positions to be updated later for each window in nwindows
leftx_current = leftx_base
rightx_current = rightx_base

# Create empty lists to receive left and right lane pixel indices
left_lane_inds = []
right_lane_inds = []

# for win in windows:

plt.plot(histogram)
plt.show()
