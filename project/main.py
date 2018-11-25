import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import initialize as init
import detection as dt

input = glob.glob('project/images/test/*.png')
images, binary_img, warped_images, peaks = init.find_lane(input)

for i in range(len(warped_images)):
    im = warped_images[i]
    lp, rp = peaks[i]
    out_img = np.dstack((im, im, im))*255
    x = dt.curvature(im, [], [])

'''======================================================='''
# plot images
# fig, (ax1, ax2, ax3, ax4) = plt.subplots(2, 1, figsize=(7, 10))

# plt.subplot(221)
# plt.imshow(images[0], cmap='gray')
# plt.title("Image 1")

# plt.subplot(222)
# plt.imshow(images[4], cmap='gray')
# plt.title("Image 2")

# plt.subplot(223)
# plt.imshow(images[5], cmap='gray')
# plt.title("Image 3")

# plt.subplot(224)
# plt.imshow(images[2], cmap='gray')
# plt.title("Image 4")
# plt.show()
