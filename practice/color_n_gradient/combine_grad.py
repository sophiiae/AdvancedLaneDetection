import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

''' apply sobel filter to the image, calculate directional gradient'''
def abs_sobel_thresh(img, orient='x', kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # calculate derivative in different directions
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel)

    # assgin derivate with parameter
    if (orient == 'x'):
        sobel = sobelx
    else:
        sobel = sobely

    # calculate absolute value and convert to 8-bit
    abs_v = np.absolute(sobel)
    scaled = np.uint8(255 * abs_v / np.max(abs_v))

    # generate output
    grad_binary = np.zeros_like(scaled)
    grad_binary[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
    return grad_binary

''' calculate magnitude of the gradient '''
def mag_thresh(img, kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # calculate derivative in different directions
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel)
    sobel = np.sqrt(sobelx ** 2 + sobely ** 2)

    # calculate absolute value and convert to 8-bit
    abs_v = np.absolute(sobel)
    scaled = np.uint8(255 * abs_v / np.max(abs_v))

    # generate output
    mag_binary = np.zeros_like(scaled)
    mag_binary[(scaled >= mag_thresh[0]) & (scaled <= mag_thresh[1])] = 1
    return mag_binary

''' calculate direction of the gradient '''
def dir_threshold(img, kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # calculate derivative in different directions
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel)

    # calculate absolute value and gradient direction 
    abs_x = np.absolute(sobelx)
    abs_y = np.absolute(sobely)
    dir = np.arctan2(abs_y, abs_x) 

    # generate output
    dir_binary = np.zeros_like(dir)
    dir_binary[(dir >= thresh[0]) & (dir <= thresh[1])] = 1
    return dir_binary

image = mpimg.imread('practice/images/signs.png')

# Choose a Sobel kernel size
ksize = 3 # Choose a larger odd number to smooth gradient measurements

# Apply each of the thresholding functions
gradx = abs_sobel_thresh(image, orient='x', kernel=ksize, thresh=(20, 100))
grady = abs_sobel_thresh(image, orient='y', kernel=ksize, thresh=(20, 100))
mag_binary = mag_thresh(image, kernel=ksize, mag_thresh=(30, 100))
dir_binary = dir_threshold(image, kernel=ksize, thresh=(0.7, 1.3))

combined = np.zeros_like(dir_binary)
combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

# plot the result
fig = plt.figure
plt.imshow(combined, cmap='gray')
plt.show()