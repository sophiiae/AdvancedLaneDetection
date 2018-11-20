import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def region(im, mask):

    pts = np.array([mask], np.int32)
    pts = pts.reshape((-1,1,2))
    
    # draw the region of interest insize four points
    result = cv2.polylines(im, [pts], True, (255, 0, 0), 2)

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
