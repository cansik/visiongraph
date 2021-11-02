from argparse import ArgumentParser, Namespace
from typing import Optional

import numpy as np

from visiongraph.input.BaseInput import BaseInput
import cv2
import logging

from visiongraph.util.TimeUtils import current_millis


class VideoCaptureInput(BaseInput):
    def __init__(self):
        super().__init__()
        self.channel = 0
        self.loop = True
        self._cap: Optional[cv2.VideoCapture] = None

    def setup(self):
        self._cap = cv2.VideoCapture(self.channel)

        if not (self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width) and
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)):
            print("Could not set media input size!")

        if not (self._cap.set(cv2.CAP_PROP_FPS, self.fps)):
            print("Could not set media framerate!")

    def release(self):
        self._cap.release()

    def read(self) -> (int, Optional[np.ndarray]):
        if not self._cap.isOpened():
            logging.critical(f"{self.__class__.__name__} is not opened")
            return None

        success, image = self._cap.read()
        time_stamp = current_millis()

        if not success:
            if self.loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # fix this behaviour to nto show (could not read frame)

            logging.warning(f"{self.__class__.__name__} could not read frame")
            return time_stamp, None

        return time_stamp, image

    def configure(self, args: Namespace):
        super().configure(args)

        if str(args.channel).isnumeric():
            self.channel = int(args.channel)
        else:
            self.channel = args.channel

    @staticmethod
    def add_params(parser: ArgumentParser):
        super(VideoCaptureInput, VideoCaptureInput).add_params(parser)
        parser.add_argument("--channel", type=str, default=0,
                            help="Input device channel (camera id, video path, image sequence).")
