import os
import json
import argparse

import torch

from multimedia.player import VideoPlayer
from mot.detector.models import ObjectDetector

parser = argparse.ArgumentParser("-c", "--config", default="config.json", help="configuration file")

def main(args):

    # Environment Setup
    # =================
    use_gpu = True if torch.cuda.is_available() else False

    # Load configuration file to a dictionary
    with open(args['config'], "r") as f:
        config = json.loads(f.read())

    # Construct video player
    video_player = VideoPlayer(config['video']['path'],
                        (config['video']['width'], config['video']['height']))

    # Construct object detector
    detector = ObjectDetector(backend="")

    # Construct object tracker
    tracker = None

    # Tracking Pipeline
    # =================

if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(args)
