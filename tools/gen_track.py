import argparse
import cv2
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True, help="output file")
parser.add_argument("--wsize", default="1024,768", help="size of window")
parser.add_argument("--bsize", default="50,200", help="size of bounding box")

# Recording status
STATUS = {
    'iframe': 0,
    'drawing': False,
    'coordinates': [],
    'canvas': None
}

def draw_track(event, x, y, flags, param):
    bbox_width, bbox_height = tuple([ int(v) for v in param['bsize'].split(",") ])
    bbox_width += 10*(np.random.random()*2-1)
    bbox_height += 10*(np.random.random()*2-1)

    if event == cv2.EVENT_LBUTTONDOWN:
        STATUS['drawing'] = True

    elif STATUS['drawing'] and event == cv2.EVENT_MOUSEMOVE:
        cv2.circle(STATUS['canvas'], (x, y), 5, (0, 255, 0), -1)
        STATUS['coordinates'].append((x, y, bbox_width/bbox_height, bbox_height))

        # Flush track to output file
        if len(STATUS['coordinates']) >= 100:
            with open(param['output'], 'a+') as output:
                for idx, (x, y, a, h) in enumerate(STATUS['coordinates']):
                    output.write("{},{},{},{},{}\n".format(
                                                        STATUS['iframe']+idx,
                                                        x, y, a, h))
            STATUS['iframe'] += len(STATUS['coordinates'])
            STATUS['coordinates'] = []

    elif event == cv2.EVENT_LBUTTONUP:
        STATUS['drawing'] = False

def main(args):
    window_width, window_height = tuple([ int(v) for v in args['wsize'].split(",") ])
    bbox_width, bbox_height = tuple([ int(v) for v in args['bsize'].split(",") ])

    # Clear output file content if it already existed
    with open(args['output'], 'w') as f:
        f.write("")

    # Create Opencv Window
    cv2.namedWindow("Track")
    cv2.setMouseCallback("Track", draw_track, args)

    # Create an canvas to draw track
    canvas = np.zeros((window_height, window_width, 3), np.uint8)
    STATUS['canvas'] = canvas

    while True:
        cv2.imshow("Track", canvas)
        key = cv2.waitKey(25) & 0xFF
        if key == ord('q') or key == 27: # q or esc key
            cv2.destroyAllWindows()
            break

    if len(STATUS['coordinates']) > 0:
        with open(args['output'], 'a+') as output:
            for idx, (x,y,a,h) in enumerate(STATUS['coordinates']):
                    output.write("{},{},{},{},{}\n".format(
                                                        STATUS['iframe']+idx,
                                                        x, y, a, h))

if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(args)
