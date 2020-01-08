import os
import argparse

import cv2
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True, help="output file")
parser.add_argument("--source", required=True, help="source directory for tracks file")


def main(args):
    # Get all track files
    files = [ os.path.join(args['source'], f) for f in os.listdir(args['source']) ]

    # Keep track files' information in dictionary
    tracks = {}
    for tid, f in enumerate(files):
        with open(f, "r") as fin:
            lines = fin.readlines()
            tracks[tid] = lines

    # Decide the period of time of the video
    max_length = 100000000
    for tid, records in tracks.items():
        if max_length > len(records):
            max_length = len(records)

    # Synchronize every track file so that they are in the same length
    for tid, records in tracks.items():
        tracks[tid] = records[:max_length]

    # Merged all tracks
    merged_buffer = []
    for iframe in range(max_length):
        buf = []
        for k in tracks:
            terms = tracks[k][iframe].split(",")
            terms.insert(1, str(k))
            tracks[k][iframe] = ",".join(terms)
            buf.append(",".join(terms))

        merged_buffer.append("".join(buf))
        if len(merged_buffer) > 100:
            with open(args['output'], "a+") as output:
                output.writelines(merged_buffer)
            merged_buffer = []

    # Visualize all the tracks
    cv2.namedWindow("Merged")

    # Create an canvas to draw track
    canvas = np.zeros((768, 1024, 3), np.uint8)

    for iframe in range(max_length):

        for k in tracks:
            (_, tid, x, y, a, h) = tracks[k][iframe].strip("\n").split(",")

            if int(tid) % 3 == 0:
                cv2.circle(canvas, (int(x), int(y)), 5, (255, 0, 0), -1)
            elif int(tid) % 3 == 1:
                cv2.circle(canvas, (int(x), int(y)), 5, (0, 255, 0), -1)
            elif int(tid) % 3 == 2:
                cv2.circle(canvas, (int(x), int(y)), 5, (0, 0, 255), -1)

        cv2.imshow("Merged", canvas)

        key = cv2.waitKey(25) & 0xFF
        if key == ord('q') or key == 27: # q or esc key
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(args)
