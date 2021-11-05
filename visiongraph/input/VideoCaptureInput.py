import time
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
        self.fps_lock = True
        self._cap: Optional[cv2.VideoCapture] = None

        self._last_read_time = 0
        self._no_frame_count = 0
        self._no_frame_max = 3

    def setup(self):
        self._cap = cv2.VideoCapture(self.channel)

        if not (self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width) and
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)):
            logging.warning(f"{self.__class__.__name__} could not set media input size")

        if not (self._cap.set(cv2.CAP_PROP_FPS, self.fps)):
            logging.warning(f"{self.__class__.__name__} could not set media framerate")

        self.fps = self._cap.get(cv2.CAP_PROP_FPS)

        if self.fps == 0:
            logging.warning(f"{self.__class__.__name__} fps could not be read")
            self.fps = 30

    def release(self):
        self._cap.release()

    def read(self) -> (int, Optional[np.ndarray]):
        if not self._cap.isOpened():
            raise Exception(f"{self.__class__.__name__} is not opened with channel {self.channel}")

        # wait with read to match fps
        if self.fps_lock:
            fps_wait_time = (1000.0 / self.fps) - (current_millis() - self._last_read_time)
            if 1000.0 > fps_wait_time > 1:
                time.sleep(fps_wait_time / 1000.0)

        success, image = self._cap.read()
        time_stamp = current_millis()

        self._last_read_time = time_stamp

        if not success:
            if self.loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # retry getting frame
            if self._no_frame_count < self._no_frame_max:
                self._no_frame_count += 1
                self._last_read_time = 0
                return self._post_process(*self.read())

            logging.warning(f"{self.__class__.__name__} could not read frame")
            return self._post_process(time_stamp, None)

        self._no_frame_count = 0
        return self._post_process(time_stamp, image)

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
