import io
import time
import zlib
import struct
import socket
import pickle
import argparse

import cv2

from multimedia import VideoStream

parser = argparse.ArgumentParser()
parser.add_argument("--capture", default="0", help="video source")
parser.add_argument("--ip", default="127.0.0.1", help="server ip to connect")
parser.add_argument("--port", default="9999", help="serivce port")


GLOBAL = {
    'tracking': {
        'topLeft': None,
        'bottomRight': None,
        'state': False,
        'clicked': False
    },
}

def reset_global():
    global GLOBAL
    GLOBAL['tracking'] = {
        'topLeft': None,
        'bottomRight': None,
        'state': False,
        'clicked': False
    }

def select_and_track(event, x, y, flags, param):

    global GLOBAL

    # if the left mouse button was clicked, record the starting (x, y) coordinate
    if event == cv2.EVENT_LBUTTONDOWN:
        reset_global()
        GLOBAL['tracking']['topLeft'] = (x, y)
        GLOBAL['tracking']['clicked'] = True

    # recording the ending (x, y) coordinate when release left mouse button
    elif event == cv2.EVENT_LBUTTONUP and not GLOBAL['tracking']['state']:
        GLOBAL['tracking']['bottomRight'] = (x, y)
        GLOBAL['tracking']['clicked'] = False
        GLOBAL['tracking']['state'] = True

    elif event == cv2.EVENT_MOUSEMOVE and GLOBAL['tracking']['clicked']:
        GLOBAL['tracking']['bottomRight'] = (x, y)
    else:
        pass

def main(args):

    # Connect to server
    # =================
    # print("Connect to {}:{}".format(args['ip'], args['port']))
    # client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # client_socket.connect((args['ip'], int(args['port'])))

    # Connect to video source
    # =======================
    print("Stream video {}".format(args['capture']))
    stream = VideoStream(args['capture'])
    stream.start()

    # Interactive interface to the client user
    # ========================================
    global GLOBAL
    name = "Single Object Tracking"
    cv2.namedWindow(name)
    cv2.setMouseCallback(name, select_and_track)

    prev_frame = None
    while stream.size() > 0 or stream.state != "stop":

        # Get previous buffered frame if it is paused
        if stream.state == "pause" and prev_frame is not None:
            frame = prev_frame.copy()

        # Get new frame if it is started
        else:
            frame = stream.read()
            prev_frame = frame

        # Show selected rectangle if user has clicked
        if GLOBAL['tracking']['clicked']:
            cv2.rectangle(frame,
                    GLOBAL['tracking']['topLeft'],
                    GLOBAL['tracking']['bottomRight'],
                    (0, 255, 0), 2)

        # Show tracking state
        # - Red border on the screen when is not in tracking
        # - Green border on the screen when is in tracking
        if GLOBAL['tracking']['state']:
            cv2.rectangle(frame, (0, 0), stream.resolution, (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (0, 0), stream.resolution, (0, 0, 255), 2)

        # Show the current frame
        cv2.imshow(name, frame)

        # Process key event
        # =================
        key = cv2.waitKey(25) & 0xFF
        if key == ord('q') or key == 27: # q or esc key
            stream.state = "stop"
            cv2.destroyAllWindows()
            break
        elif key == 32: # space key
            if stream.state == "pause":
                stream.state = "start"
            elif stream.state == "start":
                stream.state = "pause"


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(args)
