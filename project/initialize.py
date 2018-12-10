import matplotlib.image as mpimg
import numpy as np
import imtool as tool
import matplotlib.pyplot as plt
import cv2

src = np.array([[1049, 629], [1525, 1066], [620, 1066], [988, 629]])
dst = np.array([[1400, 0], [1400, 1070], [700, 1070], [700, 0]])

def find_lane_set(input):
    # read input images
    images = []
    for file in input: 
        img = mpimg.imread(file)
        images.append(img)

    # apply threshold to l channel
    binary_img = []
    binary_roi_img = []
    for img in images: 
        [l, a, b]  = tool.lab(img)
        binary_roi, binary = tool.combine_thresh(l)
        binary_roi_img.append(binary_roi)
        binary_img.append(binary)

    # warp images
    warped_images = []
    for img in binary_img: 
        warped, out_img, M = tool.warp(img, src, dst)
        warped_images.append(warped)

    # find peaks on histogram of warped image
    peaks = []
    for img in warped_images:
        lb, rb = tool.hist_peak(img)
        peaks.append([lb, rb])

    return images, binary_roi_img, binary_img, warped_images, peaks, M

def find_lane(img):
    # apply threshold to l channel
    [l, a, b]  = tool.lab(img)
    binary = tool.combine_thresh(l)

    # warp images
    warped, out_img, M = tool.warp(binary, src, dst)

    # find peaks on histogram of warped image
    lp, rp = tool.hist_peak(warped)
    peak = [lp, rp]

    return warped, peak, M

def plot(a, m=None, t=''):
    plt.imshow(a, cmap=m)
    plt.title(t)
    plt.show()