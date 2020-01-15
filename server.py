import time
import socket
import pickle
import argparse
from threading import Thread

import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from mot.tracker.kalman import KalmanFilter

parser = argparse.ArgumentParser()
parser.add_argument("--ip", default="127.0.0.1", help="server ip to connect")
parser.add_argument("--port", default="9999", help="server service port")

HEADER_SIZE = 10

class ClientThread(Thread):
    """Thread for handling client connection

    Each time a client connected with the server, the server will spawn a new
    thread to deal with the new client

    Here are the expected actions:
    [Client]: Send initial tracking status

                                    [Server]: Initialize kalman filter, send back
                                        the tracking result from kalman filter

    [Client]: Send tracking status <<<----------------------------------------+
    |                                                                         |
    |                               [Server]: Using Faster-RCNN to detect object
    |                                   and update kalman filter, send back the
    +----------------------------->>>>  tracking result from kalman filter
    """

    def __init__(self, conn ,addr):
        """
        Parameters:
            - conn: socket of connected client
            - addr: (ip, port) information
        """
        super().__init__()
        self.conn = conn
        self.addr = addr

        self.detector = fasterrcnn_resnet50_fpn(num_classes=91, pretrained=True)
        self.detector.eval()

        self.kalman = KalmanFilter()
        self.mean = None
        self.covariance = None

    def _recv_data(self):
        """Receive data from client in an agreed format

        Return:
            a dictionary with following format
            {
                'tlahs': [(x, y, a, h)],
                'state': True,
                'frame': # compressed frame in jpeg format
            }
        """
        full_msg = b''
        new_msg = True

        while True:
            msg = self.conn.recv(4096)

            if new_msg:
                msglen = int(msg[:HEADER_SIZE])
                new_msg = False

            full_msg += msg

            if len(full_msg)-HEADER_SIZE == msglen:
                data = pickle.loads(full_msg[HEADER_SIZE:])
                return data

    def _send_data(self, data):
        """Send data to client in an agreed format

        Data format:
            {
                'tlahs': [(x, y, a, h)],
                'state': True
            }
        """
        global HEADER_SIZE

        msg = pickle.dumps(data)
        msg = bytes(f"{len(msg):<{HEADER_SIZE}}", 'utf-8')+msg
        self.conn.sendall(msg)

    def _compute_iou(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    def run(self):

        while True:
            # Receive tracking status from client
            try:
                data = self._recv_data()
            except Exception as e:
                time.sleep(0.1)
                continue

            if data['state'] == False:
                self.mean = None
                self.covariance = None
                continue

            frame = cv2.imdecode(data['frame'], cv2.IMREAD_COLOR)

            # Run Tracking algorithm
            # ======================

            # Initialize kalman filter state
            if self.mean is None and self.covariance is None:
                self.mean, self.covariance = self.kalman.initiate(np.array(data['tlahs'][0]))

            # TODO
            # Detect objects in frame with faster-RCNN and update kalman filter
            else:
                # Using faster-rcnn to perform object detection
                input = [torch.from_numpy(frame/255.).permute(2,0,1).float()]
                prediction = self.detector(input)[0]
                boxes = prediction['boxes'].detach().numpy().tolist()
                labels = prediction['labels'].detach().numpy().tolist()
                scores = prediction['scores'].detach().numpy().tolist()

                # Filter out only bboxes with respect to person class
                people = []
                for box, label, score in zip(boxes, labels, scores):
                    if label == 1 and score > 0.7:
                        people.append(box)

                # Assign bbox based on IOU
                mean, covariance = self.kalman.predict(self.mean, self.covariance)

                ious = []
                cx, cy, a, h = mean.tolist()[:4]
                ref_tl_x, ref_tl_y = cx-(a*h/2), cy-(h/2)
                ref_br_x, ref_br_y = cx+(a*h/2), cy+(h/2)

                for person in people:
                    tl_x, tl_y, br_x, br_y = person[0], person[1], person[2], person[3]
                    iou = self._compute_iou(
                                    [ref_tl_x, ref_tl_y, ref_br_x, ref_br_y],
                                    [tl_x, tl_y, br_x, br_y])
                    ious.append(iou)

                # Update Kalman filter
                measurement = people[np.argmax(ious)]
                cx, cy = (measurement[0]+measurement[2])/2, (measurement[1]+measurement[3])/2
                a = ((measurement[2]-measurement[0])/(measurement[3]-measurement[1]))
                h = (measurement[3]-measurement[1])
                self.mean, self.covariance = self.kalman.update(
                                                mean, covariance,
                                                np.array([cx, cy, a, h]))

                print("IOU between measurement and mean:", np.max(ious))
                if np.max(ious) < 0.5:
                    data['state'] = False
                    self.mean = None
                    self.covariance = None

            # Prepare data to send to client
            del data['frame']
            data['tlahs'] = [self.mean.tolist()[:4]]
            self._send_data(data)

def main(args):

    clients = []

    # Launch Server
    # =============
    print("Launch server {}:{}".format(args['ip'], args['port']))
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((args['ip'], int(args['port'])))
    server_socket.listen(10)

    # Main thread for listening client connection
    while True:
        conn, addr = server_socket.accept()
        print("Connection from {}:{}".format(addr[0], addr[1]))
        client = ClientThread(conn, addr)
        client.start()
        clients.append(client)

if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(args)
