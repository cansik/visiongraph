import signal
from argparse import ArgumentParser, Namespace
from typing import Callable, Optional, Union

import cv2
import numpy as np

from visiongraph.GraphNode import GraphNode
from visiongraph.result.ImageResult import ImageResult
from visiongraph.result.ResultDict import ResultDict, DEFAULT_IMAGE_KEY

InputType = Optional[Union[np.ndarray, ResultDict, ImageResult]]


class ImagePreview(GraphNode[np.ndarray, np.ndarray]):
    """
    A graph node that displays an image and waits for user input.
    """

    def __init__(self, title: str = "Image",
                 image_key: str = DEFAULT_IMAGE_KEY,
                 wait_time: int = 1,
                 handle_key_callback: Optional[Callable[[int], None]] = None):
        """
        Initializes the ImagePreview node.

        :param title: The window title. Defaults to "Image".
        :param image_key: The key of the image result. Defaults to DEFAULT_IMAGE_KEY.
        :param wait_time: The time to wait for user input in milliseconds. Defaults to 1.
        :param handle_key_callback: A callback function to handle the pressed key. Defaults to None.
        """
        self.title = title
        self.image_key = image_key
        self.wait_time = wait_time
        self.handle_key_callback = handle_key_callback

    def setup(self):
        """
        Sets up the OpenCV window.
        """
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL or cv2.WINDOW_KEEPRATIO)

    def process(self, data: InputType) -> InputType:
        """
        Processes the input data and displays the image.

        :param data: The input data.

        :return: The processed data.
        """
        image = data

        if isinstance(data, ResultDict):
            image = data[self.image_key]

        if isinstance(data, ImageResult):
            image = data.output

        if data is None:
            return data

        cv2.imshow(self.title, image)
        key = cv2.waitKey(self.wait_time)

        if self.handle_key_callback is not None and key != 255 and key != -1:
            self.handle_key_callback(key)

        if key & 0xFF == 27:
            signal.raise_signal(signal.SIGINT)

        return data

    def release(self):
        """
        Releases the OpenCV window.
        """
        cv2.destroyWindow(self.title)

    def configure(self, args: Namespace):
        """
        Configures the node with the given arguments.

        :param args: The parsed command-line arguments.
        """
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        """
        Adds parameters to the parser.

        :param parser: The parser object.
        """
        pass
