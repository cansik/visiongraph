import signal
from argparse import ArgumentParser, Namespace
from typing import Callable, Optional, Union

import cv2
import numpy as np

from visiongraph.GraphNode import GraphNode
from visiongraph.result.ResultDict import ResultDict, DEFAULT_IMAGE_KEY


class ImagePreview(GraphNode[np.ndarray, np.ndarray]):
    def __init__(self, title: str = "Image",
                 image_key: str = DEFAULT_IMAGE_KEY,
                 wait_time: int = 1,
                 handle_key_callback: Optional[Callable[[int], None]] = None):
        self.title = title
        self.image_key = image_key
        self.wait_time = wait_time
        self.handle_key_callback = handle_key_callback

    def setup(self):
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL or cv2.WINDOW_KEEPRATIO)

    def process(self, data: Union[np.ndarray, ResultDict]) -> Union[np.ndarray, ResultDict]:
        image = data

        if isinstance(image, ResultDict):
            image = data[self.image_key]

        cv2.imshow(self.title, image)
        key = cv2.waitKey(self.wait_time)

        if self.handle_key_callback is not None and key != 255 and key != -1:
            self.handle_key_callback(key)

        if key & 0xFF == 27:
            signal.raise_signal(signal.SIGINT)

        return data

    def release(self):
        cv2.destroyWindow(self.title)

    def configure(self, args: Namespace):
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass
