import time
import socket
import pickle
import argparse
from threading import Thread

import cv2
import numpy as np

from multimedia import VideoStream

parser = argparse.ArgumentParser()
parser.add_argument("--capture", default="0", help="video source")
parser.add_argument("--ip", default="127.0.0.1", help="server ip to connect")
parser.add_argument("--port", default="9999", help="serivce port")


HEADER_SIZE = 10

GLOBAL = {
    'tracking': {
        'topLeft': None,
        'bottomRight': None,
        # 'tlahs': [],
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

def recv_data(conn):
    full_msg = b''
    new_msg = True

    while True:
        msg = conn.recv(4096)

        if new_msg:
            msglen = int(msg[:HEADER_SIZE])
            new_msg = False

        full_msg += msg

        if len(full_msg)-HEADER_SIZE == msglen:
            data = pickle.loads(full_msg[HEADER_SIZE:])
            return data

def send_data(conn, data):
    global HEADER_SIZE

    msg = pickle.dumps(data)
    msg = bytes(f"{len(msg):<{HEADER_SIZE}}", 'utf-8')+msg
    conn.sendall(msg)

def main(args):

    # Connect to server
    # =================
    print("Connect to {}:{}".format(args['ip'], args['port']))
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((args['ip'], int(args['port'])))

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

        if GLOBAL['tracking']['clicked']:
            data = {
                'tlahs': [(0, 0, 0, 0)],
                'state': GLOBAL['tracking']['state'],
                'frame': None
            }
            send_data(client_socket, data)
            stream.state = "pause"

        # If it is tracking, then it should communicate with server
        if GLOBAL['tracking']['state']:

            # Send data to server to process frame
            # ====================================
            tl_x, tl_y = GLOBAL['tracking']['topLeft']
            br_x, br_y = GLOBAL['tracking']['bottomRight']
            cx, cy = (tl_x+br_x)/2, (tl_y+br_y)/2
            w, h = (tl_x-br_x), (br_y-tl_y)
            data = {
                'tlahs': [(cx, cy, abs(w/h), h)],
                'state': GLOBAL['tracking']['state'],
                'frame': cv2.imencode(
                                    '.jpg',
                                    frame,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), 90])[1]
            }
            send_data(client_socket, data)

            # Recv processed frame from server
            # ===============================
            data = recv_data(client_socket) # without frame information
            print(data)

            # Update tracking status
            GLOBAL['tracking']['state'] = data['state']
            if data['state']:
                cx, cy, a, h = data['tlahs'][0]
                tl_x, tl_y = int(cx-(a*h/2)), int(cy-h/2)
                br_x, br_y = int(cx+(a*h/2)), int(cy+h/2)
                GLOBAL['tracking']['topLeft'] = (tl_x, tl_y)
                GLOBAL['tracking']['bottomRight'] = (br_x, br_y)
            else:
                stream.state = "pause"

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
            cv2.rectangle(frame,
                    GLOBAL['tracking']['topLeft'],
                    GLOBAL['tracking']['bottomRight'],
                    (0 ,255, 0), 2)
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
