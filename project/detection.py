import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import imtool as tool

def find_lane(binary_warped, peak, prev_left=[], prev_right=[]):
    nwindows = 12
    margin = 140
    minpix = 70

    window_height = np.int(binary_warped.shape[0]//nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    total_nonzeros = len(nonzeroy)

    # Current positions to be updated for each window
    leftx_current = peak[0]
    rightx_current = peak[1] 

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    windows = []
    ratio = 0.0

    if (len(prev_left) > 0) & (len(prev_right) > 0):
        left_lane_inds = ((nonzerox > (prev_left[0]*(nonzeroy**2) + prev_left[1]*nonzeroy +  prev_left[2] - margin)) & (nonzerox < (prev_left[0]*(nonzeroy**2) + prev_left[1]*nonzeroy + prev_left[2] + margin)))

        right_lane_inds = ((nonzerox > (prev_right[0]*(nonzeroy**2) + prev_right[1]*nonzeroy +  prev_right[2] - margin)) & (nonzerox < (prev_right[0]*(nonzeroy**2) + prev_right[1]*nonzeroy + prev_right[2] + margin)))

        ratio = (np.sum(left_lane_inds) + np.sum(right_lane_inds)) / total_nonzeros
    
    if ratio < 0.8: 
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height

            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            # Draw the windows on the visualization image
            windows.append([(win_xleft_low,win_y_low),
            (win_xleft_high,win_y_high)]) 
            windows.append([(win_xright_low,win_y_low),
            (win_xright_high,win_y_high)])
            
            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    return left_fit, right_fit, left_fitx, right_fitx, leftx, lefty, rightx, righty, ploty

def compute_curvature(left_fit, right_fit, left_fitx, right_fitx, ploty):
    y_eval = np.max(ploty)

    leftx = left_fitx
    rightx = right_fitx

    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    return left_curverad, right_curverad
