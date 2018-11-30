import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import os
import initialize as init
import detection as dt
import curvature as cur
import draw

# dir = 'project/images/test_out/'
dir_in = 'project/images/test/'
input = glob.glob(dir_in + '*.png')
images, binary_img, warped_images, peaks, M = init.find_lane(input)
dir_out = dir_in + "out"
if not os.path.isdir(dir_out): os.makedirs(dir_out)
print("*-> Input directory:   " + dir_in)
print("*-> Output directory:  " + dir_out)
print("-")

prev_left = []
prev_right = []
left_c = []
right_c = []
left_fitxx =[]
right_fitxx = []
area_out = []
region_out = []
lane_out = []

def processIMG(im, warped, peak, prev_left, prev_right):
    left_fit, right_fit, left_fitx, right_fitx, ploty, left_pts, right_pts,leftw, rightw = dt.find_lane(warped, peak, prev_left, prev_right)

    lc, rc = dt.compute_curvature(left_fit, right_fit, left_fitx, right_fitx, ploty)

    lane = draw.draw_lane(warped, left_fitx, right_fitx, leftw, rightw)
    region = draw.draw_region(warped, left_fitx, right_fitx)
    area = draw.draw_area(im, left_fitx, right_fitx, M)
    left_fitxx.append(left_fitx)
    right_fitxx.append(right_fitx)
    prev_left = left_fit
    prev_right = right_fit

    #change color of detected lane pixels
    lane[left_pts] = [255, 0, 0]
    lane[right_pts] = [0, 0, 255]

    # add parameters to corresponding list
    left_c.append(lc)
    right_c.append(rc)

    return area, lane, region

for i in range(len(images)):
    area, lane, region = processIMG(images[i], warped_images[i], peaks[i], prev_left, prev_right)
    # output.append(out)

    # title = "Area of Lanes on Image " + str(i+1)
    # plt.imshow(out)
    # plt.title(title)
    # plt.show()

    out_name = dir_out + "/" + str(i+1) + "_area.png"
    plt.imsave(out_name,area)
    out_name = dir_out + "/" + str(i+1) + "_lane.png"
    plt.imsave(out_name,lane)
    out_name = dir_out + "/" + str(i+1) + "_region.png"
    plt.imsave(out_name,region)
    print(" *** Images Processed: " + str(i+1) + " *** ")

print("---- done ----")

# title = "Area of Lanes on Image " + str(i+1)
# plt.imshow(area)
# plt.title(title)
# plt.show()

'''======================================================='''
# plot images

    # out_img = np.dstack((warped, warped, warped))
    # out_img[lefty, leftx] = [255, 0, 0]
    # out_img[righty, rightx] = [0, 0, 255]
    # output
    #.append(out_img)

# fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(7, 10))

# plt.subplot(221)
# plt.imshow(output[0], cmap='gray')
# plt.title("Image 1")

# plt.subplot(222)
# plt.imshow(output[4], cmap='gray')
# plt.title("Image 2")

# plt.subplot(223)
# plt.imshow(output[5], cmap='gray')
# plt.title("Image 3")

# plt.subplot(224)
# plt.imshow(output[2], cmap='gray')
# plt.title("Image 4")

# plt.show()
