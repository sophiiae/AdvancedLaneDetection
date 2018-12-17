import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import run
import processIMG as process
import time

start = time.time()

prev_left = []
prev_right = []

def process_image(image):
    global prev_left
    global prev_right
    area, lane, region, left_fit, right_fit = process.processIMG(image,prev_left, prev_right)
    prev_left = left_fit
    prev_right = right_fit
    return area

# run lane detection on image folder
dir = 'project/images/test/' 
run.run_set(dir)

end = time.time()
print('Run time: ', end - start)