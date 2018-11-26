import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

def draw_lane(im, left_fitx, right_fitx, leftw, rightw):
    """ draw lane and sliding windows on binary warped image """
    out_img = np.dstack((im, im, im))*255

    # draw the lines
    ploty = np.linspace(0, im.shape[0] - 1, im.shape[0])
    left_pts = np.dstack((left_fitx, ploty)).astype(np.int32)
    right_pts = np.dstack((right_fitx, ploty)).astype(np.int32)

    cv2.polylines(out_img, left_pts, False, (138,43,226), 4)
    cv2.polylines(out_img, right_pts, False, (138,43,226), 4)

    for low, high in leftw:
        cv2.rectangle(out_img, low, high, (255, 0, 0), 3)

    for low, high in rightw:            
        cv2.rectangle(out_img, low, high, (255, 0, 0), 3)   

    return out_img

def draw_region(im, left_fitx, right_fitx):
    """ draw lane area on binary warped image """
    margin = 140
    ploty = np.linspace(0, im.shape[0] - 1, im.shape[0])
    
    left_w1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_w2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_pts = np.hstack((left_w1, left_w2))

    right_w1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_w2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_pts = np.hstack((right_w1, right_w2))

    # Create RGB image from binary warped image
    region_img = np.dstack((im, im, im)) * 255

    # Draw the lane onto the warped blank image
    cv2.fillPoly(region_img, np.int_([left_pts]), (0, 255, 0))
    cv2.fillPoly(region_img, np.int_([right_pts]), (0, 255, 0))

    return region_img

def draw_area(im, left_fitx, right_fitx, M):
    color = np.zeros_like(im)
    ploty = np.linspace(0, im.shape[0] - 1, im.shape[0])
    
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color, np.int_([pts]), (51,0,102))

    # newwarp = np.zeros_like(im)
    newwarp = cv2.warpPerspective(color, M, (im.shape[1], im.shape[0]), flags=cv2.WARP_INVERSE_MAP)

    result = cv2.addWeighted(im, 1, newwarp, 0.3, 0)
    return result

    return result
    