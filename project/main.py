import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip
import run

# run lane detection on image folder
dir = 'project/images/test/'
run.run_set(dir)  

im = mpimg.imread(dir + 'ROAD2_0157.png')
# run.run_single(im)

vid_dir = 'project/images/'
vid_name = 'test1.mp4'