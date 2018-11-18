import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

dir = 'practice/checkerboard_images/'
images = []
for x in os.listdir(dir):
    images.append(x)

objpoints = []   # 3D points in real world space
imgpoints = []   # 2D points in image plane

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ...., (7,5,0)
objp = np.zeros((6 * 9, 3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)        

for fname in images: 
    path = dir + fname                                          
    im = mpimg.imread(path)   
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) 

    # find chessboard corners for (9 x 6 board)
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    # if corners are found
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)

        # draw detected corners on an image
        im = cv2.drawChessboardCorners(im, (9, 6), corners, ret)

# camera calibration, get all parameters
size = gray.shape[::-1]  # image size in pixel. alternative: colorIMG.shape[1::-1]
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, size, None, None)

img = mpimg.imread('practice/checkerboard_images/calibration8.jpg')
# undistort a test image
dst = cv2.undistort(img, mtx, dist, None, mtx)

plt.imshow(dst)
plt.show()