import os
import cv2
import glob

dir = 'project/images/imgs/out/region/'
input = glob.glob(dir + '*.png')
input.sort()

def make_video(images, fps=20, size=None, is_color=True, format="XVID", outvid='image_video_region.avi'):
        fourcc = cv2.VideoWriter_fourcc(*format)
        vid = None
        for image in images:
            print(image)
            if not os.path.exists(image):
                raise FileNotFoundError(image)
            img = cv2.imread(image)
            if vid is None:
                if size is None:
                    size = img.shape[1], img.shape[0]
                vid = cv2.VideoWriter(outvid, fourcc, float(fps), size, is_color)
            if size[0] != img.shape[1] and size[1] != img.shape[0]:
                img = cv2.resize(img, size)
            vid.write(img)
        vid.release()
        return vid

vid = make_video(input)