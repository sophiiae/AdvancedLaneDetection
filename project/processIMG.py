import initialize as init
import detection as dt
import draw

left_fitxx =[]
right_fitxx = []
left_c = []
right_c = []

def processIMG_set(im, warped, peak, prev_left, prev_right, M):
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

    return area, lane, region, left_fit, right_fit


def processIMG(im, prev_left=[], prev_right=[]):
    warped, peak, M = init.find_lane(im)
    left_fit, right_fit, left_fitx, right_fitx, ploty, left_pts, right_pts,leftw, rightw = dt.find_lane(warped, peak, prev_left, prev_right)

    lc, rc = dt.compute_curvature(left_fit, right_fit, left_fitx, right_fitx, ploty)

    lane = draw.draw_lane(warped, left_fitx, right_fitx, leftw, rightw)
    region = draw.draw_region(warped, left_fitx, right_fitx)
    area = draw.draw_area(im, left_fitx, right_fitx, M)

    #change color of detected lane pixels
    lane[left_pts] = [255, 0, 0]
    lane[right_pts] = [0, 0, 255]

    return area, lane, region, left_fit, right_fit