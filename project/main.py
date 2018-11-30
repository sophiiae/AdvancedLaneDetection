import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip
import run

# run lane detection on image folder
dir = 'project/images/r4/'
run.run_set(dir)  

im = mpimg.imread(dir + 'ROAD2_0157.png')
# run.run_single(im)
