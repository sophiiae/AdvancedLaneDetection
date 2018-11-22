import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

def hls(img):
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    return [h_channel, l_channel, s_channel]

def sobel(img):
    # Sobel x
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    return scaled_sobel

def sobel_thresh(img, orient, th=(0, 255)):
    # calculate derivative in different directions
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)

    # assgin derivate with parameter
    if (orient == 'x'):
        sobel = sobelx
    else:
        sobel = sobely

    # calculate absolute value and convert to 8-bit
    abs_v = np.absolute(sobel)
    scaled = np.uint8(255 * abs_v / np.max(abs_v))

    # generate output
    binary_output = np.zeros_like(scaled)
    binary_output[(scaled >= th[0]) & (scaled <= th[1])] = 1
    return binary_output

def gradient_th(img, th=(0, 255)):
    # Threshold x gradient
    g_binary = np.zeros_like(img)
    g_binary[(img >= th[0]) & (img <= th[1])] = 1
    return g_binary

def color_th(img, th=(0, 255)):
    # Threshold color channel
    s_binary = np.zeros_like(img)
    s_binary[(img >= th[0]) & (img <= th[1])] = 1
    return s_binary

def pipeline(img, s_thresh=(170, 255), g_thresh=(20, 100)):
    img = np.copy(img)

    [h, l, s] = hls(img)   # get different color channels
    scaled_sobel = sobel(l) # apply sobel to channel image
    
    g_binary = gradient_th(scaled_sobel, g_thresh) # apply gradient threshold
    s_binary = color_th(s, s_thresh) # apply color threshold

    # Stack each channel
    color_binary = np.dstack((np.zeros_like(g_binary), g_binary, s_binary)) * 255
    combined_binary = np.zeros_like(g_binary)
    combined_binary[(s_binary == 1) | (g_binary == 1)] = 1
    
    return [color_binary, combined_binary]

def hist(im):
    # Take a histogram of the bottom half of the image
    hist = np.sum(im[im.shape[0]//2:,:], axis=0)
    return hist

# params = (binary image, # of wins, width of wins, min # of pixels found to recenter windows)
def slidewins(binary, nwindows=9, margin=200, minpix=100):
    hist = np.sum(binary[binary.shape[0]//2:,:], axis=0)
    out_img = np.dstack((binary, binary, binary))*255
    # Find the peak of the left and right halves of the histogram and set start point of windows
    midpoint = np.int(hist.shape[0]//2)
    leftx_base = np.argmax(hist[:midpoint])
    rightx_base = np.argmax(hist[midpoint:]) + midpoint

    window_height = np.int(binary.shape[0]//nwindows)
    # x and y positions of all nonzero (activated) px in the image
    nonzero = binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # initialize empty array for left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows): 
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary.shape[0] - (window+1) * window_height
        win_y_high = binary.shape[0] - window * window_height

        win_xleft_low = leftx_current - margin  
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(255,0,0), 4) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,0,255), 4) 

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        print("not recentered the window properly")

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = slidewins(binary_warped)

    ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
    left_fit = np.polyfit(lefty,leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')

    return out_img
