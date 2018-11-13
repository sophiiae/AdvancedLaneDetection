import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

im = mpimg.imread('images/test3.jpg')
gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)  #convert to grayscale

# calculate derivative in the x & y directions
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

# calculate absolute value of the derivatives
abs_sobelx = np.absolute(sobelx)
abs_sobely = np.absolute(sobely)

# convert the absolute value image to 8-bit, useful when apply threshold and work with different scale of images. 
scaled_sobelx = np.uint8(255 * abs_sobelx/np.max(abs_sobelx))
scaled_sobely = np.uint8(255 * abs_sobely/np.max(abs_sobely))

# create threshold
thresh_min = 20
thresh_max = 100

sxbinary = np.zeros_like(scaled_sobelx)
sxbinary[(scaled_sobelx >= thresh_min) & (scaled_sobelx <= thresh_max)] = 1

sybinary = np.zeros_like(scaled_sobely)
sybinary[(scaled_sobely >= thresh_min) & (scaled_sobely <= thresh_max)] = 1

# plot images
plt.figure(figsize=(6, 12))

plt.subplot(311)
plt.imshow(gray, cmap='gray')
plt.title("gray image")

plt.subplot(312)
plt.imshow(sxbinary, cmap='gray')
plt.title("sobel x")

plt.subplot(313)
plt.imshow(sybinary, cmap='gray')
plt.title("sobel y")
plt.show()