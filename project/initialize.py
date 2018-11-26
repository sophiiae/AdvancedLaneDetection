import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import imtool as tool

src = np.array([[1049, 629], [1525, 1066], [620, 1066], [988, 629]])
dst = np.array([[1400, 0], [1400, 1070], [700, 1070], [700, 0]])

def find_lane(input):
    images = []
    for file in input: 
        img = mpimg.imread(file)
        images.append(img)

    binary_img = []
    for img in images: 
        [l, a, b]  = tool.lab(img)
        binary = tool.combine_thresh(l)
        binary_img.append(binary)

    warped_images = []
    roi = []
    for img in binary_img: 
        warped, out_img, M = tool.warp(img, src, dst)
        warped_images.append(warped)
        roi.append(out_img)

    peaks = []
    for img in warped_images:
        lb, rb = tool.hist_peak(img)
        peaks.append([lb, rb])

    return images, binary_img, warped_images, peaks, M