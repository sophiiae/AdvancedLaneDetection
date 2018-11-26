import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import initialize as init
import detection as dt
import curvature as cur

dir = 'project/images/test_out/'
input = glob.glob('project/images/test/*.png')
images, binary_img, warped_images, peaks = init.find_lane(input)

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

for i in range(len(warped_images)):
    im = warped_images[i]
    
    if i == 0: 
        left_fit, right_fit, left_fitx, right_fitx, leftx, lefty, rightx, righty, ploty = dt.find_lane(im, peaks[i], [], [])
    else:
        left_fit, right_fit, left_fitx, right_fitx, leftx, lefty, rightx, righty, ploty = dt.find_lane(im, peaks[i], prev_left[i-1], prev_right[i-1])

    out_img = np.dstack((im, im, im))
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    result.append(out_img)

    left.append([leftx, lefty])
    right.append([rightx, righty])
    left_fitxx.append(left_fitx)
    right_fitxx.append(right_fitx)
    # left_fit, right_fit, ploty, out_img = cur.fit_polynomial(im)
    # lc, rc = cur.measure_curvature_pixels(ploty, left_fit, right_fit)
    prev_left.append(left_fit)
    prev_right.append(right_fit)

    # left_c.append(lc)
    # right_c.append(rc)
    # print('lc: ', lc, '|', 'rc: ', rc)

    # title = "Lane of Image " + str(i+1)
    # plt.imshow(out_img)
    # plt.title(title)
    # plt.show()


'''======================================================='''
# plot images
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
