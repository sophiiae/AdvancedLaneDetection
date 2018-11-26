import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2

def compute_lane_lines(self, warped_img):
    # Take a histogram of the bottom half of the image, summing pixel values column wise 
    histogram = np.sum(warped_img[warped_img.shape[0]//2:,:], axis=0)
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines 
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint # don't forget to offset by midpoint!

    # Set height of windows
    window_height = np.int(warped_img.shape[0]//self.sliding_windows_per_line)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    total_non_zeros = len(nonzeroy)
    non_zero_found_pct = 0.0
    
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base    

    # Set the width of the windows +/- margin
    margin = 140
    # Set minimum number of pixels found to recenter window
    minpix = 70
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Our lane line objects we store the result of this computation
    left_line = LaneLine()
    right_line = LaneLine()
                    
    if self.previous_left_lane_line is not None and self.previous_right_lane_line is not None:
        # We have already computed the lane lines polynomials from a previous image
        left_lane_inds = ((nonzerox > (prev_left[0] * (nonzeroy**2) + prev_left[1] * nonzeroy + prev_left[2] - margin)) & (nonzerox < (prev_left[0] * (nonzeroy**2)  + prev_left[1] * nonzeroy + prev_left[2] + margin))) 

        right_lane_inds = ((nonzerox > (prev_right[0] * (nonzeroy**2) + prev_right[1] * nonzeroy + prev_right[2] - margin)) & (nonzerox < (prev_right[0] * (nonzeroy**2) + prev_right[1] * nonzeroy + prev_right[2] + margin))) 
        
        non_zero_found_left = np.sum(left_lane_inds)
        non_zero_found_right = np.sum(right_lane_inds)
        non_zero_found_pct = (non_zero_found_left + non_zero_found_right) / total_non_zeros
        
        print("[Previous lane] Found pct={0}".format(non_zero_found_pct))
        #print(left_lane_inds)
    
    if non_zero_found_pct < 0.85:
        print("Non zeros found below thresholds, begining sliding window - pct={0}".format(non_zero_found_pct))
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(self.sliding_windows_per_line):
            # Identify window boundaries in x and y (and right and left)
            # We are moving our windows from the bottom to the top of the screen (highest to lowest y value)
            win_y_low = warped_img.shape[0] - (window + 1)* window_height
            win_y_high = warped_img.shape[0] - window * window_height

            # Defining our window's coverage in the horizontal (i.e. x) direction 
            # Notice that the window's width is twice the margin
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            left_line.windows.append([(win_xleft_low,win_y_low),(win_xleft_high,win_y_high)])
            right_line.windows.append([(win_xright_low,win_y_low),(win_xright_high,win_y_high)])

            # Super crytic and hard to understand...
            # Basically nonzerox and nonzeroy have the same size and any nonzero pixel is identified by
            # (nonzeroy[i],nonzerox[i]), therefore we just return the i indices within the window that are nonzero
            # and can then index into nonzeroy and nonzerox to find the ACTUAL pixel coordinates that are not zero
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

        # Concatenate the arrays of indices since we now have a list of multiple arrays (e.g. ([1,3,6],[8,5,2]))
        # We want to create a single array with elements from all those lists (e.g. [1,3,6,8,5,2])
        # These are the indices that are non zero in our sliding windows
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        
        non_zero_found_left = np.sum(left_lane_inds)
        non_zero_found_right = np.sum(right_lane_inds)
        non_zero_found_pct = (non_zero_found_left + non_zero_found_right) / total_non_zeros
        
        print("[Sliding windows] Found pct={0}".format(non_zero_found_pct))
        
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    left_line.polynomial_coeff = left_fit
    right_line.polynomial_coeff = right_fit
    
    if not self.previous_left_lane_lines.append(left_line):
        left_fit = self.previous_left_lane_lines.get_smoothed_polynomial()
        left_line.polynomial_coeff = left_fit
        self.previous_left_lane_lines.append(left_line, force=True)
        print("**** REVISED Poly left {0}".format(left_fit))            

    if not self.previous_right_lane_lines.append(right_line):
        right_fit = self.previous_right_lane_lines.get_smoothed_polynomial()
        right_line.polynomial_coeff = right_fit
        self.previous_right_lane_lines.append(right_line, force=True)
        print("**** REVISED Poly right {0}".format(right_fit))

    # Generate x and y values for plotting
    ploty = np.linspace(0, warped_img.shape[0] - 1, warped_img.shape[0] )
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
    
    left_line.polynomial_coeff = left_fit
    left_line.line_fit_x = left_fitx
    left_line.non_zero_x = leftx  
    left_line.non_zero_y = lefty

    right_line.polynomial_coeff = right_fit
    right_line.line_fit_x = right_fitx
    right_line.non_zero_x = rightx
    right_line.non_zero_y = righty
    
    return (left_line, right_line)