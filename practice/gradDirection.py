import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


# apply sobel filter to the image
def dir_threshold(img, kernel_size, thresh):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # calculate derivative in different directions
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)

    # calculate absolute value and gradient direction 
    abs_x = np.absolute(sobelx)
    abs_y = np.absolute(sobely)
    dir = np.arctan2(abs_y, abs_x) 

    # generate output
    binary_output = np.zeros_like(dir)
    binary_output[(dir >= thresh[0]) & (dir <= thresh[1])] = 1
    return binary_output

# define the inputs for the function
im = mpimg.imread('images/signs.png')
kernel = 15
thresh = (0.7, 1.3)
output = dir_threshold(im, kernel, thresh)

# plot the result
fig = plt.figure
plt.imshow(output, cmap='gray')
plt.show()