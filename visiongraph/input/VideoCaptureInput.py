from typing import Optional

import numpy as np

from visiongraph.input.BaseInput import BaseInput
import cv2
import logging

from visiongraph.util.TimeUtils import current_millis


class VideoCaptureInput(BaseInput):
    def __init__(self, input_param=0):
        self.input_param = input_param
        self.loop = True
        self._cap: Optional[cv2.VideoCapture] = None

    def setup(self):
        self._cap = cv2.VideoCapture(self.input_param)

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

            logging.warning(f"{self.__class__.__name__} could not read frame")
            return time_stamp, None

        return time_stamp, image

    def configure(self, args):
        if str(args.channel).isnumeric():
            self.input_param = int(args.channel)
        else:
            self.input_param = args.channel
