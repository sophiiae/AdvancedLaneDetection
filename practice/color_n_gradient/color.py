import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

im = mpimg.imread('practice/images/test6.jpg')
# thresh = (180, 255)
gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
# binary = np.zeros_like(gray)
# binary[(gray > thresh[0]) & (gray <= thresh[1])] = 1

red = im[:,:,0]
green = im[:,:,1]
blue = im[:,:,2]

thresh = (200, 255)
binary = np.zeros_like(red)
binary[(red > thresh[0]) & (red <= thresh[1])] = 1

hls = cv2.cvtColor(im, cv2.COLOR_RGB2HLS)
h = hls[:,:,0]
l = hls[:,:,1]
s = hls[:,:,2]

thresh = (90, 255)
binary = np.zeros_like(s)
binary[(s > thresh[0]) & (s <= thresh[1])] = 1

th = (15, 100)
hi = np.zeros_like(h)
hi[(h > th[0]) & (h <= th[1])] = 1

# plot images
plt.figure(figsize=(10, 6))

plt.subplot(221)
plt.imshow(s, cmap='gray')
plt.title("s")

plt.subplot(222)
plt.imshow(binary, cmap='gray')
plt.title("b")

plt.subplot(223)
plt.imshow(l, cmap='gray')
plt.title("l")

plt.subplot(224)
plt.imshow(h, cmap='gray')
plt.title("h")
plt.show()