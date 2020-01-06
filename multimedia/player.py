import cv2
import numpy as np
from PIL import Image

class VideoPlayer:

    def __init__(self, video_path, resolution=(1024, 768)):
        self.video_path = video_path
        self.resolution = resolution
        self.cap = cv2.VideoCapture(video_path)

    def __str__(self):
        content = "[%s]\n - Number of frames: %d\n - Video FPS: %d\n - Resolution: %s" % (self.video_path,
            int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(self.cap.get(cv2.CAP_PROP_FPS)),
            str(self.resolution))
        return content

    def __iter__(self):
        """Called by for loop syntax"""
        return self

    def __next__(self):
        """Raise StopIteration exception when all frames are yielded"""
        ret, frame = self.cap.read()

        # Finish playing the video
        if not ret:
            self.cap.release()
            self.cap = cv2.VideoCapture(self.video_path)
            raise StopIteration

        # In opencv, Frame shape: (height, width, depth)
        frame = cv2.resize(frame, self.resolution)
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        return frame, img
