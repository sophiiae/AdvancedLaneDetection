import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# return each channel of HLS color space
def hls(img):
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    return [h_channel, l_channel, s_channel]

def sobel(img):
    # Sobel x
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    return scaled_sobel

def sobel_thresh(img, orient, thresh_min, thresh_max):

    # calculate derivative in different directions
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)

    # assgin derivate with parameter
    if (orient == 'x'):
        sobel = sobelx
    else:
        sobel = sobely

    # calculate absolute value and convert to 8-bit
    abs_v = np.absolute(sobel)
    scaled = np.uint8(255 * abs_v / np.max(abs_v))

    # generate output
    binary_output = np.zeros_like(scaled)
    binary_output[(scaled >= thresh_min) & (scaled <= thresh_max)] = 1
    return binary_output

def gradient_th(img, th=(0, 255)):
    # Threshold x gradient
    g_binary = np.zeros_like(img)
    g_binary[(img >= th[0]) & (img <= th[1])] = 1
    return g_binary

def color_th(img, th=(0, 255)):
    # Threshold color channel
    s_binary = np.zeros_like(img)
    s_binary[(img >= th[0]) & (img <= th[1])] = 1
    return s_binary

def pipeline(img, s_thresh=(170, 255), g_thresh=(20, 100)):
    img = np.copy(img)

    [h, l, s] = hls(img)   # get different color channels
    scaled_sobel = sobel(l) # apply sobel to channel image
    
    g_binary = gradient_th(scaled_sobel, g_thresh) # apply gradient threshold
    s_binary = color_th(s, s_thresh) # apply color threshold

    # Stack each channel
    color_binary = np.dstack((np.zeros_like(g_binary), g_binary, s_binary)) * 255
    combined_binary = np.zeros_like(g_binary)
    combined_binary[(s_binary == 1) | (g_binary == 1)] = 1
    
    return [color_binary, combined_binary]

# draw ROI on image, mask = [[topright], [bottomright], [bottomleft], [topleft]]
def region(im, mask):

    pts = np.array([mask], np.int32)
    pts = pts.reshape((-1,1,2))
    
    # draw the region of interest insize four points
    result = cv2.polylines(im, [pts], True, (255, 0, 0), 4)

    return result

def warp(im, mask, dst):
    size = (im.shape[1], im.shape[0])

    # points coordinates on original image
    src = np.float32(mask)

    # expected points coordinates
    dst = np.float32(dst)

    # compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)

    # create warped image - use linear interpolation
    warped = cv2.warpPerspective(im, M, size, flags=cv2.INTER_LINEAR)

    return warped

def plot(im1, name_im1, cmap_im1, im2, name_im2, cmap_im2, im3, name_im3, cmap_im3, im4, name_im4, cmap_im4, im5, name_im5, cmap_im5, im6, name_im6, cmap_im6):
    
    plt.figure(figsize=(14, 8))

    plt.subplot(231)
    plt.imshow(im1, cmap=cmap_im1)
    plt.title(name_im1)

    plt.subplot(232)
    plt.imshow(im2, cmap=cmap_im2)
    plt.title(name_im2)

    plt.subplot(233)
    plt.imshow(im3, cmap=cmap_im3)
    plt.title(name_im3)

    plt.subplot(234)
    plt.imshow(im4, cmap=cmap_im4)
    plt.title(name_im4)

    plt.subplot(235)
    plt.imshow(im5, cmap=cmap_im5)
    plt.title(name_im5)

    plt.subplot(236)
    plt.imshow(im6, cmap=cmap_im6)
    plt.title(name_im6)

    plt.show()