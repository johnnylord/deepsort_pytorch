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
    'stream': {
        'state': 'pause'
    },
    'tracking': {
        'topLeft': None,
        'bottomRight': None,
        'state': False
    },
    'cv2': {
        'name': None,
        'frame': None,
        'clicked': False
    }
}

def select_and_track(event, x, y, flags, param):

    global GLOBAL

    # if the left mouse button was clicked, record the starting (x, y) coordinate
    if event == cv2.EVENT_LBUTTONDOWN and not GLOBAL['tracking']['state']:
        GLOBAL['tracking']['topLeft'] = (x, y)
        GLOBAL['cv2']['clicked'] = True

    elif event == cv2.EVENT_LBUTTONUP and not GLOBAL['tracking']['state']:
        GLOBAL['tracking']['bottomRight'] = (x, y)
        GLOBAL['cv2']['clicked'] = False
        GLOBAL['tracking']['state'] = True

    elif event == cv2.EVENT_MOUSEMOVE and GLOBAL['cv2']['clicked']:
        cv2.rectangle(GLOBAL['cv2']['frame'], reference_points[0], (x, y), (0, 255, 0), 2)
        cv2.imshow(GLOBAL['cv2']['name'], GLOBAL['cv2']['frame'])
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

    # Create Opencv Window
    global GLOBAL
    name = "Single Object Tracking"
    cv2.namedWindow(name)
    cv2.setMouseCallback(name, select_object)

    GLOBAL['cv2']['name'] = name

    while stream.size() > 0 or not stream.stopped:
        if GLOBAL['stream']['state'] == "pause" and GLOBAL['cv2']['frame'] is not None:
            frame = GLOBAL['cv2']['frame'].copy()
        else:
            frame = stream.read()

        GLOBAL['cv2']['frame'] = frame
        cv2.imshow(name, frame)

        key = cv2.waitKey(25) & 0xFF
        if key == ord('q') or key == 27: # q or esc key
            stream.stopped = True
            cv2.destroyAllWindows()
            break
        elif key == 32: # space
            if GLOBAL['stream']['state'] == "pause":
                GLOBAL['stream']['state'] = "start"
            if GLOBAL['stream']['state'] == "start":
                GLOBAL['stream']['state'] = "pause"

if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(args)
