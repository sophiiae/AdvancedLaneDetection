import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2

def normalize_samples(l_samples, r_samples):
    widths = r_samples[:,0] - l_samples[:,0]

    # lane width should not be less than 600
    mean_width = np.max([np.mean(widths), 600])
    
    diff = (widths - mean_width) / 2
    l_samples[:,0] = l_samples[:,0] + diff
    r_samples[:,0] = r_samples[:,0] - diff
    return l_samples, r_samples

# curve detection
def curvature_detection(img, old_l_samples=[], old_r_samples=[], img_c=None):
    threshold = 100
    stride_height = 50
    window_height = 100
    window_width = 50
    img_height = img.shape[0]
    img_width = img.shape[1]

    if (len(old_l_samples) == 0) | (len(old_r_samples) == 0):
        old_samples_provided = False
        print("detecting left and peak from current image")
        lp, rp = peak_detection(img)
        lane_widths = [rp - lp] * int((img_height / stride_height) + 1)
    else:
        old_samples_provided = True
        lp = old_l_samples[0, 0]
        rp = old_r_samples[0, 0]
        lane_widths = old_r_samples[:, 0] - old_l_samples[:, 0]
        
    # result
    l_samples = []
    r_samples = []

    # window sliding
    i = 0
    for window_y_end in range(img_height, stride_height, -stride_height):
        window_y_start = window_y_end - window_height
        
        # histogram for entire horizon of current window slide
        histogram = np.sum(img[window_y_start:window_y_end, int(window_width / 2) : -int(window_width / 2)], axis=0)

        # calculate convolution by sliding window accross the horizon
        window = np.ones(window_width)
        conv = np.convolve(window, histogram)
                
        # left window based on previous left peak
        window_l_start = int(lp - (window_width / 2))
        window_l_end = window_l_start + window_width

        # right window based on previous right peak
        window_r_start = int(rp - (window_width / 2))
        window_r_end = window_r_start + window_width

        # new left and right peaks
        l_found = sum(conv[window_l_start:window_l_end]) > threshold
        r_found = sum(conv[window_r_start:window_r_end]) > threshold

        if r_found & l_found:
            lp = argmax(conv[window_l_start:window_l_end]) + window_l_start
            rp = argmax(conv[window_r_start:window_r_end]) + window_r_start 
        else:
            if l_found:
                lp = argmax(conv[window_l_start:window_l_end]) + window_l_start
                rp = min(lp + lane_widths[i], img_width)    
            if r_found:
                rp = argmax(conv[window_r_start:window_r_end]) + window_r_start 
                lp = max(rp - lane_widths[i], 0)

        # draw left window on output image (green if found, red otherwise)
        if type(img_c) is np.ndarray:
            l_window_color = [0,255,0] if l_found else [255, 0, 0]
            cv2.rectangle(img_c, (window_l_start, window_y_start), (window_l_end, window_y_end), l_window_color, 2)
            r_window_color = [0,255,0] if l_found else [255, 0, 0]
            cv2.rectangle(img_c, (window_r_start, window_y_start), (window_r_end, window_y_end), r_window_color, 2)

        l_samples.append([lp, window_y_end])
        r_samples.append([rp, window_y_end])
        
        window_width = window_width + 10
        i = i + 1
        
    l_samples = np.array(l_samples)
    r_samples = np.array(r_samples)
    #normalize_samples(l_samples, r_samples)
    
    if old_samples_provided:
        all_l_samples = np.concatenate((l_samples, old_l_samples))
        all_r_samples = np.concatenate((r_samples, old_r_samples))
    else:
        all_l_samples = l_samples
        all_r_samples = r_samples
        
    l_fit = np.polyfit(all_l_samples[:,1], all_l_samples[:,0], 2)
    r_fit = np.polyfit(all_r_samples[:,1], all_r_samples[:,0], 2)
    
    # Define conversions in x and y from pixels space to meters
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    ym_per_pix = 30/720 # meters per pixel in y dimension

    # Fit new polynomials to x,y in world space
    l_fit_r = np.polyfit(all_l_samples[:,1]*ym_per_pix, all_l_samples[:,0]*xm_per_pix, 2)
    r_fit_r = np.polyfit(all_r_samples[:,1]*ym_per_pix, all_r_samples[:,0]*xm_per_pix, 2)
    
    return l_samples, r_samples, l_fit, r_fit, l_fit_r, r_fit_r


















def add_curvature_overlay(img, l_fit, r_fit, l_fit_r, r_fit_r, M):
    img_height = img.shape[0]
    img_width = img.shape[1]
    
    # Create an image to draw the lines on
    color_warp = np.zeros_like(img).astype(np.uint8)
    y_plot = np.linspace(0, img_height-1, img_height)
    l_plot = l_fit[0] * y_plot ** 2 + l_fit[1] * y_plot + l_fit[2]
    r_plot = r_fit[0] * y_plot ** 2 + r_fit[1] * y_plot + r_fit[2]
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([l_plot, y_plot]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([r_plot, y_plot])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, M, (img_width, img_height), flags=cv2.WARP_INVERSE_MAP) 
    
    # Calculate the new radii of curvature
    y_eval = np.max(y_plot)
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    ym_per_pix = 30/720 # meters per pixel in y dimension
    
    l_c = ((1 + (2*l_fit_r[0]*y_eval*ym_per_pix + l_fit_r[1])**2)**1.5) / np.absolute(2*l_fit_r[0])
    r_c = ((1 + (2*r_fit_r[0]*y_eval*ym_per_pix + r_fit_r[1])**2)**1.5) / np.absolute(2*r_fit_r[0])    

    cv2.putText(img, 'L={:.4f}y^2{:+.4f}y{:+.4f}'.format(l_fit[0], l_fit[1], l_fit[2]), (10,220), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(img, 'R={:.4f}y^2{:+.4f}y{:+.4f}'.format(r_fit[0], r_fit[1], r_fit[2]), (10,260), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(img, 'Left: {:.2f}m, Right: {:.2f}m'.format(l_c, r_c), (10,300), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(img, 'Offset: {:+.2f}m'.format(((r_plot[-1] + l_plot[-1]) / 2 - (img_width / 2)) * xm_per_pix), (10,340), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)

    # Combine the result with the original image
    return cv2.addWeighted(img, 1, newwarp, 0.3, 0)

def process_image(image, mtx, dist):    
    global l_sample_history, r_sample_history
    
    img = distortion_correction(image, mtx, dist)
    img = get_threshold_binary(img)
    img, M = perspective_transform(img) 
    img_c = np.dstack([img, img, img]) * 255
    
    l_sh = np.array(l_sample_history).reshape(-1, 2)
    r_sh = np.array(r_sample_history).reshape(-1, 2)
    
    new_l_samples, new_r_samples, l_fit, r_fit, l_fit_c, r_fit_c = curvature_detection(img, l_sh, r_sh, img_c)
    
    l_sample_history.append([new_l_samples])
    r_sample_history.append([new_r_samples])
    
    img_c = cv2.resize(img_c, (int(img_c.shape[1] / 4), int(img_c.shape[0] / 4))) 
    result = add_curvature_overlay(image, l_fit, r_fit, l_fit_c, r_fit_c, M)
    result[10:img_c.shape[0] + 10, 10:img_c.shape[1] + 10] = img_c
    
    return result

