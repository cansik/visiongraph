import time
from argparse import ArgumentParser, Namespace
from typing import Optional, Union

import numpy as np

from visiongraph.input.BaseInput import BaseInput
import cv2
import logging

from visiongraph.util.TimeUtils import current_millis


class VideoCaptureInput(BaseInput):
    def __init__(self, channel: Union[str, int] = 0, input_skip: int = -1,
                 loop: bool = True, fps_lock: bool = True):
        super().__init__()
        self.channel = channel
        self.input_skip = input_skip
        self.loop = loop
        self.fps_lock = fps_lock
        self._cap: Optional[cv2.VideoCapture] = None

        self._last_read_time = 0
        self._no_frame_count = 0
        self._no_frame_max = 3

    def setup(self):
        self._cap = cv2.VideoCapture(self.channel)

        if not self._cap.isOpened():
            logging.warning("Could not open VideoCapture, please check if channel is correct.")

        if not (self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width) and
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)):
            # if not settable-try to read
            self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # logging.warning("could not set media input size")

        if not (self._cap.set(cv2.CAP_PROP_FPS, self.fps)):
            logging.warning("could not set media framerate")

        self.fps = self._cap.get(cv2.CAP_PROP_FPS)

        if self.fps == 0:
            logging.warning("fps could not be read")
            self.fps = 30

        if self.input_skip >= 0:
            self._cap.set(cv2.CAP_PROP_POS_MSEC, self.input_skip)

    def release(self):
        self._cap.release()

    def read(self) -> (int, Optional[np.ndarray]):
        if not self._cap.isOpened():
            raise Exception(f"Could not open channel {self.channel}, please check path.")

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

            logging.warning("could not read frame")
            return self._post_process(time_stamp, None)

        self._no_frame_count = 0
        return self._post_process(time_stamp, image)

    @property
    def frame_count(self) -> int:
        if not self._cap.isOpened():
            return -1
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def configure(self, args: Namespace):
        super().configure(args)

        if str(args.channel).isnumeric():
            self.channel = int(args.channel)
        else:
            self.channel = args.channel

        self.input_skip = args.input_skip

    @staticmethod
    def add_params(parser: ArgumentParser):
        super(VideoCaptureInput, VideoCaptureInput).add_params(parser)
        parser.add_argument("--channel", type=str, default=0,
                            help="Input device channel (camera id, video path, image sequence).")
        parser.add_argument("--input-skip", type=int, default=-1,
                            help="If set the input will be skipped to the value in milliseconds.")
