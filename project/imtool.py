import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

def lab(im):
    # Convert to HLS color space and separate the V channel
    lab = cv2.cvtColor(im, cv2.COLOR_RGB2LAB)

    l = lab[:,:,0]
    a = lab[:,:,1]
    b = lab[:,:,2]
    return [l, a, b]

def ROI(img):
    h = img.shape[0]
    w = img.shape[1]
    pts = np.array([[w/2, h/2+50], [0, h], [w, h]], dtype=np.int32)
    mask = np.zeros_like(img)
    cv2.fillConvexPoly(mask, pts, 1)
    return cv2.bitwise_and(img, mask)

def perspective_transform(src, dst):
    """compute perspective transform matrix and inverse matrix """
    M = cv2.getPerspectiveTransform(src, dst)
    # M_inv = cv2.getPerspectiveTransform(dst, src)
    return M

def warp(im, src, dst):
    out_img = np.zeros_like(im) # for binary image
    # out_img = np.dstack((im, im, im))*255 # for RGB image
    size = (im.shape[1], im.shape[0])

    # compute the perspective transform matrix
    M = perspective_transform(np.float32(src), np.float32(dst))
    # create warped image - use linear interpolation
    cv2.polylines(out_img, [src], True, (68, 17, 84), 2)
    warped = cv2.warpPerspective(im, M, size, flags=cv2.INTER_LINEAR)
    return warped, out_img, M
    
def color_th(img, th=(0, 255)):
    # Threshold color channel
    binary = np.zeros_like(img)
    binary[(img > th[0]) & (img <= th[1])] = 1
    return binary

def sobel_thresh(img, orient, th=10, kernel=15):
    """ apply Sobel filter to image with orientation and threshold """
    # calculate derivative in different directions
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel)

    # assgin derivate with parameter
    if (orient == 'x'):
        sobel = sobelx
    elif (orient == 'y'):
        sobel = sobely
    else: 
        sobel = np.sqrt(sobelx ** 2 + sobely ** 2)

    # calculate absolute value and convert to 8-bit
    abs_v = np.absolute(sobel)
    scaled = np.uint8(255 * abs_v / np.max(abs_v))

    # generate output
    binary = np.zeros_like(scaled)
    binary[scaled >= th] = 1
    return binary

def gradient(gray, th=1.5, kernel=15):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel)

    # calculate absolute value and gradient direction 
    abs_x = np.absolute(sobelx)
    abs_y = np.absolute(sobely)
    dir = np.arctan2(abs_y, abs_x)

    # generate output
    binary_output = np.zeros_like(dir)
    # binary_output[(dir >= thresh[0]) & (dir <= thresh[1])] = 1
    binary_output[dir >= 1.5] = 1
    return binary_output

def combine_thresh(color_c, kernel=15):
    th = (68, 100)
    cc = color_th(color_c, th)
    sx = sobel_thresh(color_c,'x', 15)
    sy = sobel_thresh(color_c,'y', 15)
    sxy = sobel_thresh(color_c, 'xy')
    combined = np.zeros_like(color_c)
    # combine sobel filters result
    combined[(sx == 1) & ((sy == 1) & (sxy == 1))] = 1
    # get best fit result from sobel and color filters
    combined[(cc == 0) & (combined == 1)] = 0
    combined[(cc == 1) & (combined == 0)] = 1
    return ROI(combined)

def argmax(h):
    start = np.argmax(h)
    end = len(h) - np.argmax(h[::-1])
    return int((start + end)/2)

def hist_peak(img):
    [h, w] = img.shape[:2]
    mid = int(w/2)
    half = img[int(h/2):,:]
  
    hist = np.sum(half, axis=0)

    leftb = argmax(hist[:mid])
    rightb = argmax(hist[mid:]) + mid

    return leftb, rightb