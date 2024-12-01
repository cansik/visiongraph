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
    """
    A class to handle image input for processing, allowing configuration
    of image path and delay time.
    """

    def __init__(self, path: Optional[str] = None, delay: float = 1.0):
        """
        Initializes the ImageInput instance with a specified image path and delay.

        :param path: The file path to the input image.
        :param delay: The delay time before reading the image, in seconds.
        """
        super().__init__()

        self.path: Optional[str] = path
        self.delay: float = delay

        self.image: Optional[np.ndarray] = None

    def setup(self):
        """
        Sets up the image input by loading the image from the specified path.

        :raises Exception: If the specified image path does not exist.
        """
        if not os.path.exists(self.path):
            raise Exception(f"Could not find input image path: '{self.path}'")

        self.image = cv2.imread(self.path)

    def read(self) -> (int, Optional[np.ndarray]):
        """
        Reads the image and captures the current timestamp.


        :return: A tuple containing the current timestamp
        """
        image = self.image.copy()
        time_stamp = current_millis()

        if self.delay > 0.0:
            time.sleep(self.delay)

        return self._post_process(time_stamp, image)

    def release(self):
        """
        Releases any resources held by the image input.

        """
        pass

    def configure(self, args: Namespace):
        """
        Configures the image input parameters from command-line arguments.

        :param args: The command-line arguments containing input path
        """
        if args.source is not None:
            args.input_path = args.source

        self.path = args.input_path
        self.delay = args.input_delay

    @staticmethod
    def add_params(parser: ArgumentParser):
        """
        Adds command-line parameters for configuring the image input.

        :param parser: The argument parser instance to add parameters to.
        """
        CommonArgs.add_source_argument(parser)

        parser.add_argument("--input-path", type=str, help="Path to the input image.")
        parser.add_argument("--input-delay", type=float, default=1.0, help="Input delay time (s).")
