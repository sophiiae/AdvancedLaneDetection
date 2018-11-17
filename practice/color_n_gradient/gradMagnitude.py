import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# apply sobel filter to the image
def mag_thresh(img, kernel_size, thresh_min, thresh_max):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # calculate derivative in different directions
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
    sobel = np.sqrt(sobelx ** 2 + sobely ** 2)

    # calculate absolute value and convert to 8-bit
    abs_v = np.absolute(sobel)
    scaled = np.uint8(255 * abs_v / np.max(abs_v))

    # generate output
    binary_output = np.zeros_like(scaled)
    binary_output[(scaled >= thresh_min) & (scaled <= thresh_max)] = 1
    return binary_output

# define the inputs for the function
im = mpimg.imread('practice/images/signs.png')
kernel_size = 9
thresh_min = 30
thresh_max = 100
output = mag_thresh(im, kernel_size, thresh_min, thresh_max)

# plot the result
fig = plt.figure
plt.imshow(output, cmap='gray')
plt.show()