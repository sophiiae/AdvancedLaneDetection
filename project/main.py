import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import initialize as init
import detection as dt
import curvature as cur
import draw

dir = 'project/images/test_out/'
input = glob.glob('project/images/test/*.png')
images, binary_img, warped_images, peaks, M = init.find_lane(input)

im = images[0]
bi = binary_img[0]
warped = warped_images[4]
pk = peaks[0]

prev_left = []
prev_right = []
result = []
left_c = []
right_c = []
left_fitxx =[]
right_fitxx = []
left = []
right = []
left_win = []
right_win = []

for i in range(len(warped_images)):
    warped = warped_images[i]
    im = images[i]
    if i == 0: 
        left_fit, right_fit, left_fitx, right_fitx, leftx, lefty, rightx, righty, ploty, leftw, rightw = dt.find_lane(warped, peaks[i], [], [])
    else:
        left_fit, right_fit, left_fitx, right_fitx, leftx, lefty, rightx, righty, ploty, leftw, rightw = dt.find_lane(warped, peaks[i], prev_left[i-1], prev_right[i-1])

    lc, rc = dt.compute_curvature(left_fit, right_fit, left_fitx, right_fitx, ploty)
    
    out_img = draw.draw_lane(warped, left_fitx, right_fitx, leftw, rightw)
    # region_out = draw.draw_region(warped, left_fitx, right_fitx)
    area = draw.draw_area(im, left_fitx, right_fitx, M)

    left.append([leftx, lefty])
    right.append([rightx, righty])
    left_fitxx.append(left_fitx)
    right_fitxx.append(right_fitx)

    prev_left.append(left_fit)
    prev_right.append(right_fit)

    left_win.append(leftw)
    right_win.append(rightw)

    left_c.append(lc)
    right_c.append(rc)
    # print('lc: ', lc, '|', 'rc: ', rc)

    title = "Area of Lanes on Image " + str(i+1)
    plt.imshow(area)
    plt.title(title)
    plt.show()

'''======================================================='''
# plot images

    # out_img = np.dstack((warped, warped, warped))
    # out_img[lefty, leftx] = [255, 0, 0]
    # out_img[righty, rightx] = [0, 0, 255]
    # result.append(out_img)

# fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(7, 10))

# plt.subplot(221)
# plt.imshow(binary_img[0], cmap='gray')
# plt.title("Image 1")

# plt.subplot(222)
# plt.imshow(binary_img[4], cmap='gray')
# plt.title("Image 2")

# plt.subplot(223)
# plt.imshow(binary_img[5], cmap='gray')
# plt.title("Image 3")

# plt.subplot(224)
# plt.imshow(binary_img[2], cmap='gray')
# plt.title("Image 4")

# plt.show()
