import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

im = mpimg.imread('project/images/test/ROAD2_1004.png')
thresh = (0, 255)
gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
binary = np.zeros_like(gray)
binary[(gray > thresh[0]) & (gray <= thresh[1])] = 1

red = im[:,:,0]
green = im[:,:,1]
blue = im[:,:,2]

# thresh = (0, 255)
# binary = np.zeros_like(red)
# binary[(red > thresh[0]) & (red <= thresh[1])] = 1

hls = cv2.cvtColor(im, cv2.COLOR_RGB2HLS)
h = hls[:,:,0]
l = hls[:,:,1]
s = hls[:,:,2]

thresh = (90, 255)
hh = np.zeros_like(h)
hh[(h > thresh[0]) & (h <= thresh[1])] = 1

hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
lab = cv2.cvtColor(im, cv2.COLOR_RGB2LAB)

h = hsv[:,:,0]
s = hsv[:,:,1]
v = hsv[:,:,2]

l = lab[:,:,0]
a = lab[:,:,1]
b = lab[:,:,2]

th = (60, 80)
hi = np.zeros_like(l)
hi[(l > th[0]) & (l <= th[1])] = 1

# plot images
plt.figure(figsize=(10, 6))

plt.subplot(221)
plt.imshow(l, cmap='gray')
plt.title("l")

plt.subplot(222)
plt.imshow(a, cmap='gray')
plt.title("a")

plt.subplot(223)
plt.imshow(b, cmap='gray')
plt.title("b")

plt.subplot(224)
plt.imshow(hi, cmap='gray')
plt.title("l")
plt.show()