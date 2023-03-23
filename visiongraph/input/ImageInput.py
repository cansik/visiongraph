import os.path
import time
from argparse import ArgumentParser, Namespace
from typing import Optional

import cv2
import numpy as np

from visiongraph.input.BaseInput import BaseInput
from visiongraph.util import CommonArgs
from visiongraph.util.TimeUtils import current_millis


class ImageInput(BaseInput):
    def __init__(self, path: Optional[str] = None, delay: float = 1.0):
        super().__init__()

        self.path: Optional[str] = path
        self.delay: float = delay

        self.image: Optional[np.ndarray] = None

    def setup(self):
        if not os.path.exists(self.path):
            raise Exception(f"Could not find input image path: '{self.path}'")

        self.image = cv2.imread(self.path)

    def read(self) -> (int, Optional[np.ndarray]):
        image = self.image.copy()
        time_stamp = current_millis()

        if self.delay > 0.0:
            time.sleep(self.delay)

        return self._post_process(time_stamp, image)

    def release(self):
        pass

    def configure(self, args: Namespace):
        if args.source is not None:
            args.input_path = args.source

        self.path = args.input_path
        self.delay = args.input_delay

    @staticmethod
    def add_params(parser: ArgumentParser):
        CommonArgs.add_source_argument(parser)

        parser.add_argument("--input-path", type=str, help="Path to the input image.")
        parser.add_argument("--input-delay", type=float, default=1.0, help="Input delay time (s).")
