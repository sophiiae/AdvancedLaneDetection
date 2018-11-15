import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

im = mpimg.imread('images/test6.jpg')

def hls_select(im, thresh=(0, 255)): 
    hls = cv2.cvtColor(im, cv2.COLOR_RGB2HLS)
    # 1) Convert to HLS color space
    # 2) Apply a threshold to the S channel
    # 3) Return a binary image of threshold result
    s = hls[:,:,2]
    binary = np.zeros_like(s)
    binary[(s > thresh[0]) & (s <= thresh[1])] = 1
    return binary

binary = hls_select(im, (90, 255))
# plot images ============================================
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(im)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(binary, cmap='gray')
ax2.set_title('Thresholded S', fontsize=30)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()