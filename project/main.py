import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from modules import warp
from modules import line
from modules import plot

im = mpimg.imread('project/images/road2/ROAD2_0001.png')
gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
mask = [[1049, 629], [1525, 1066], [620, 1066], [988, 629]]
dst = [[1400, 0], [1400, 1070], [700, 1070], [700, 0]]
# region = warp.region(im, mask)
warped = warp.warp(im, mask, dst)
[h, l, s] = line.hls(warped)
sobel_l = line.sobel_thresh(l, 'x', (10, 30))
# hist = line.hist(sobel_l)

leftx, lefty, rightx, righty, out_img = line.slidewins(sobel_l)

# ---  plot images ---------------------------
plot.plot1(out_img,"out", None)
# plot.plot(hist)
# plot.plot2([[im, 'image'], [warped, 'w']])
# plot.plot4([[l, 'l'],
#             [sobel_l, 'sobel l'],
#             [warped, 'w'], 
#             [region, 'r']]
#             )   

# plot.plot4([[l, 'l'],
#             [s, 's'],
#             [sobel_l, 'sl'], 
#             [h, 'b']]
#             )

