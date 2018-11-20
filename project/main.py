import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import lineExtraction
import plot
import warp

im = mpimg.imread('project/images/cordova1/f00200.png')
gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
mask = [[327, 185], [476, 370], [161, 370], [306, 185]]
dst = [[500, 0], [500, 460], [160, 460], [160, 0]]

region = warp.region(im, mask)
warped = warp.warp(im, mask, dst)
[h, l, s] = lineExtraction.hls(warped)
sobel_l = lineExtraction.sobel_thresh(l, 'x', (10, 50))

# sobel_s = lineExtraction.sobel_thresh(s, 'x', 30, 100)

# ---  plot images ---------------------------
# plot.plot1(im, 'im', None)
# plot.plot2([[im, 'image', None], [region, 'region', None]])
plot.plot4([[l, 'l', 'gray'],
            [sobel_l, 'sobel l', 'gray'],
            [warped, 'w', 'gray'], 
            [region, 'r', None]]
            )   

# plot.plot4([[l, 'l', 'gray'],
#             [h, 'h', 'gray'],
#             [s, 's', 'gray'], 
#             [sobel_s, 'sobel s', 'gray']]
#             )

