import time
from queue import Queue
from threading import Thread

import cv2
import numpy as np
from PIL import Image

class VideoStream:

    def __init__(self, source="0", queue_size=256, resolution=(1024, 768)):
        self.source = source
        self.queue_size = queue_size
        self.resolution = resolution

        self.stream = cv2.VideoCapture(int(source) if source.isdecimal() else source)
        self.frame_queue = Queue(maxsize=queue_size)
        self.thread = Thread(target=self._update, args=())
        self.thread.daemon = True
        self.state = "pause"

        if not self.stream.isOpened():
            raise Exception("Fail to connect to video source %s" % source)

    def __str__(self):
        if self.source.isdecimal():
            content = "[Webcam %s] - Resolution: %s" % (self.source, str(self.resolution))
        else:
            content = "[%s]\n - Number of frames: %d\n - Video FPS: %d\n - Resolution: %s" % (self.source,
                int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                int(self.cap.get(cv2.CAP_PROP_FPS)),
                str(self.resolution))
        return content

    def start(self):
        self.state = "start"
        self.thread.start()

    def pause(self):
        self.state = "pause"

    def stop(self):
        self.state = "stop"
        self.thread.join()

    def read(self):
        return self.frame_queue.get()

    def size(self):
        return self.frame_queue.qsize()

    def _update(self):
        """Thread to fill the stream frame buffer"""
        while self.state != "stop":
            if not self.frame_queue.full() and self.state != "pause":
                ret, frame = self.stream.read()

                if not ret:
                    self.state = "stop"
                    continue

                frame = cv2.resize(frame, self.resolution)
                self.frame_queue.put(frame)
            else:
                # wait 10ms  for consumer consuming the frames
                time.sleep(0.1)

        self.stream.release()
