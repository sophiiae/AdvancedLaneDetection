import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

im = mpimg.imread('practice/images/straight_lines1.jpg')
# plt.imshow(im)

# trapezoid points [ topright, botright, botleft, topleft]
mask = [ [683, 447], [1037, 666], [270, 666], [596, 447]]
dst = [[900, 0], [900, 666], [200, 666], [200,0]]

def region(im, mask):
    # result = np.copy(im)
    pts = np.array([mask], np.int32)
    pts = pts.reshape((-1,1,2))
    result = cv2.polylines(im, [pts], True, (255, 0, 0), 4)
    # cv2.fillPoly(result, pts, 255)
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

def plot(im, name_im, result, name_result):
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 20))

    ax1.imshow(im)
    ax1.set_title(name_im)

    ax2.imshow(result)
    ax2.set_title(name_result)
    plt.show()

result = warp(im, mask, dst)
region_select = region(im, mask)
warped_region = region(result, dst)
plot(region_select,"image", warped_region,"warped image")


