import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('project/images/test/ROAD2_0709.png')

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

def gradient_th(img, th=(0, 255)):
    # Threshold x gradient
    g_binary = np.zeros_like(img)
    g_binary[(img >= th[0]) & (img <= th[1])] = 1
    return g_binary

def color_th(img, th=(0, 255)):
    # Threshold color channel
    binary = np.zeros_like(img)
    binary[(img >= th[0]) & (img <= th[1])] = 1
    return binary

def pipeline(img, s_thresh=(170, 255), g_thresh=(20, 100)):
    img = np.copy(img)

    [h, l, s] = hls(img)   # get different color channels
    scaled_sobel = sobel(l) # apply sobel to channel image
    
    g_binary = gradient_th(scaled_sobel, g_thresh) # apply gradient threshold
    s_binary = color_th(l, s_thresh) # apply color threshold

    # Stack each channel
    color_binary = np.dstack((np.zeros_like(g_binary), g_binary, s_binary)) * 255
    combined_binary = np.zeros_like(g_binary)
    combined_binary[(s_binary == 1) | (g_binary == 1)] = 1
    
    # return [color_binary, combined_binary]
    return [g_binary, s_binary]

def plot(im, name_im, result, name_result):
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 20))

    ax1.imshow(im,cmap='gray')
    ax1.set_title(name_im)

    ax2.imshow(result,cmap='gray')
    ax2.set_title(name_result)
    plt.show()
    
[color, combined] = pipeline(image)
plot(combined,"image", color,"color image")
