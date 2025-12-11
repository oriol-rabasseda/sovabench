import numpy as np
from PIL import Image
import cv2

def video_to_frames(video_path, fps_sample, min_frames=2, nframes=None):
    video = cv2.VideoCapture(video_path)
    frame_count = round(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if not nframes:
        fps = round(video.get(cv2.CAP_PROP_FPS))
        nframes = round(frame_count/fps*fps_sample) + 1
        if nframes < min_frames:
            nframes = min_frames
    
    image_list_idx = np.rint(np.linspace(0, frame_count-1, nframes))

    i = 0
    j = 0
    frame_list = []
    while(video.isOpened()):
        ret, frame = video.read()
        if ret == False:
            break
        
        while i == int(image_list_idx[j]):
            frame_list.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

            j += 1
            if j >= len(image_list_idx):
                break
        i += 1
    video.release()

    return frame_list